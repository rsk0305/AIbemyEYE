# NASA 채널에서 신경망과 다중 시간축 점수를 정상 보정으로 결합해 평가하는 스크립트
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_anomaly_contrastive import (
    MultiModalAnomalyModel,
    Trainer,
    multimodal_pretraining_loss,
)
from src.data.nasa_loader import (
    NASADataset,
    get_nasa_channel_data,
    load_nasa_metadata,
    normalize_nasa_telemetry,
)
from src.models.multiscale_forecaster import MultiScaleProbabilisticForecaster
from src.utils.device import resolve_torch_device
from src.utils.hybrid_scoring import (
    batched_reconstruction_scores,
    evaluate_normal_only_score_fusion,
    load_score_cache,
    save_score_cache,
)
from src.utils.predictive_benchmark import combine_feature_channels, rolling_windows
from src.utils.reproducibility import seed_everything
from src.utils.robust_score_fusion import evaluate_robust_score_fusion


PRIMARY_CHANNELS = ("A-1", "D-1", "M-1")
DEVELOPMENT_HOLDOUT_CHANNELS = ("E-1", "P-1", "M-6")
MAHALANOBIS_VALIDATION_CHANNELS = ("A-5", "E-12", "S-1", "T-1")
UPPER_TAIL_VALIDATION_CHANNELS = ("A-2", "B-1", "C-1", "D-2", "F-1", "M-2")
SCORE_PIPELINE_VERSION = 4
SCORE_KEYS = (
    "neural_calibration",
    "neural_validation",
    "neural_evaluation",
    "predictive_calibration",
    "predictive_validation",
    "predictive_evaluation",
)


def _parse_csv_strings(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one channel is required")
    return values


def _parse_csv_ints(value):
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one scale is required")
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels",
        type=_parse_csv_strings,
        default=(
            PRIMARY_CHANNELS
            + DEVELOPMENT_HOLDOUT_CHANNELS
            + MAHALANOBIS_VALIDATION_CHANNELS
            + UPPER_TAIL_VALIDATION_CHANNELS
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nasa_hybrid_performance.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/cache/nasa_hybrid"),
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--eval-step", type=int, default=10)
    parser.add_argument("--scales", type=_parse_csv_ints, default=(4, 16, 64))
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument(
        "--fusion-method",
        choices=("robust_mahalanobis", "bonferroni_min_p"),
        default="robust_mahalanobis",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--time-tokens", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _normal_segments(train_dataset, window_size):
    raw_length = len(train_dataset.telemetry)
    fit_end = int(raw_length * 0.6)
    calibration_end = int(raw_length * 0.8)
    bounds = {
        "fit": (0, fit_end),
        "calibration": (fit_end, calibration_end),
        "validation": (calibration_end, raw_length),
    }
    if any(end - start < window_size for start, end in bounds.values()):
        raise ValueError("normal raw-time segments must each contain one full window")

    datasets = {}
    predictive_windows = {}
    for name, (start, end) in bounds.items():
        telemetry = train_dataset.telemetry[start:end]
        bits = train_dataset.bits[start:end]
        datasets[name] = NASADataset(telemetry, bits, window_size=window_size)
        sequence = combine_feature_channels(telemetry, bits)
        windows, _ = rolling_windows(sequence, window_size=window_size)
        if len(datasets[name]) != len(windows):
            raise RuntimeError("neural and predictive normal windows are not aligned")
        predictive_windows[name] = windows
    return datasets, predictive_windows, bounds


def _fit_reference_rescale(train_values, test_values, fit_fraction=0.6):
    return normalize_nasa_telemetry(
        train_values,
        test_values,
        reference_fraction=fit_fraction,
    )


def _evaluation_data(test_dataset, labels, window_size, eval_step):
    dataset = NASADataset(
        test_dataset.telemetry,
        test_dataset.bits,
        window_size=window_size,
        step=eval_step,
    )
    sequence = combine_feature_channels(test_dataset.telemetry, test_dataset.bits)
    windows, endpoints = rolling_windows(
        sequence,
        window_size=window_size,
        step=eval_step,
    )
    if len(dataset) != len(windows):
        raise RuntimeError("neural and predictive evaluation windows are not aligned")
    return dataset, windows, np.asarray(labels)[endpoints], endpoints


def _model_config(bit_len, args):
    return {
        "high_ch": 1,
        "mid_ch": 1,
        "low_dim": 1,
        "bit_len": int(bit_len),
        "enc_dim": 64,
        "proj_dim": 32,
        "recon_dim": 64,
        "pred_steps": 1,
        "tau": 0.07,
        "lambda_cpc": 1.0,
        "lambda_recon": 1.0,
        "time_tokens": int(args.time_tokens),
        "lr": 1e-3,
    }


def _train_neural_model(dataset, model_config, args, device):
    model = MultiModalAnomalyModel(model_config).to(device)
    trainer = Trainer(model, torch.device(device), model_config)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=_is_cuda_device(device),
        generator=generator,
    )
    last_losses = None
    processed_batches = 0
    started = perf_counter()
    model.train()
    for _ in range(args.train_epochs):
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_train_batches:
                break
            batch_a = {
                key: value.to(device, non_blocking=_is_cuda_device(device))
                for key, value in trainer._augment_batch(batch).items()
            }
            batch_b = {
                key: value.to(device, non_blocking=_is_cuda_device(device))
                for key, value in trainer._augment_batch(batch).items()
            }
            output_a = model(batch_a)
            output_b = model(batch_b)
            loss, losses = multimodal_pretraining_loss(
                output_a,
                output_b,
                batch_a,
                batch_b,
                temperature=model_config["tau"],
                reconstruction_weight=model_config["lambda_recon"],
            )
            trainer.optim.zero_grad()
            loss.backward()
            trainer.optim.step()
            processed_batches += 1
            last_losses = {
                "total": float(loss.detach().cpu()),
                "contrastive": float(losses["contrastive"].detach().cpu()),
                "reconstruction": float(losses["reconstruction"].detach().cpu()),
            }
    _synchronize(device)
    return model, {
        "seconds": float(perf_counter() - started),
        "processed_batches": processed_batches,
        "last_losses": last_losses,
    }


def _build_scores(normal_datasets, normal_windows, evaluation_dataset, evaluation_windows, model_config, args, device):
    model, training = _train_neural_model(
        normal_datasets["fit"],
        model_config,
        args,
        device,
    )

    neural_started = perf_counter()
    neural_scores = {
        "neural_calibration": batched_reconstruction_scores(
            model,
            normal_datasets["calibration"],
            device=device,
            batch_size=args.score_batch_size,
        ),
        "neural_validation": batched_reconstruction_scores(
            model,
            normal_datasets["validation"],
            device=device,
            batch_size=args.score_batch_size,
        ),
        "neural_evaluation": batched_reconstruction_scores(
            model,
            evaluation_dataset,
            device=device,
            batch_size=args.score_batch_size,
        ),
    }
    _synchronize(device)
    neural_scoring_seconds = perf_counter() - neural_started

    predictive_started = perf_counter()
    predictor = MultiScaleProbabilisticForecaster(
        scales=args.scales,
        ridge=args.ridge,
    ).fit(normal_windows["fit"])
    predictive_scores = {
        "predictive_calibration": predictor.score_windows(normal_windows["calibration"]),
        "predictive_validation": predictor.score_windows(normal_windows["validation"]),
        "predictive_evaluation": predictor.score_windows(evaluation_windows),
    }
    predictive_seconds = perf_counter() - predictive_started
    return {
        **neural_scores,
        **predictive_scores,
        "training_seconds": np.asarray(training["seconds"]),
        "neural_scoring_seconds": np.asarray(neural_scoring_seconds),
        "predictive_fit_and_scoring_seconds": np.asarray(predictive_seconds),
        "processed_train_batches": np.asarray(training["processed_batches"]),
        "last_total_loss": np.asarray(
            training["last_losses"]["total"] if training["last_losses"] else 0.0
        ),
        "last_reconstruction_loss": np.asarray(
            training["last_losses"]["reconstruction"] if training["last_losses"] else 0.0
        ),
    }


def _cache_metadata(channel, model_config, bounds, normalization, args, device):
    train_path = Path("dataset/NASA/data/data/train") / f"{channel}.npy"
    test_path = Path("dataset/NASA/data/data/test") / f"{channel}.npy"
    metadata = {
        "schema_version": 1,
        "score_pipeline_version": SCORE_PIPELINE_VERSION,
        "channel": channel,
        "seed": args.seed,
        "device": _device_identity(device),
        "window_size": args.window_size,
        "eval_step": args.eval_step,
        "normalization": normalization,
        "normal_split": {name: list(value) for name, value in bounds.items()},
        "model_config": model_config,
        "training": {
            "epochs": args.train_epochs,
            "max_batches": args.max_train_batches,
            "batch_size": args.train_batch_size,
            "score_batch_size": args.score_batch_size,
            "torch_threads": args.torch_threads,
        },
        "predictive": {"scales": list(args.scales), "ridge": args.ridge},
        "data_sha256": {
            "train": _sha256_file(train_path),
            "test": _sha256_file(test_path),
        },
        "code_sha256": {
            "neural": _sha256_file(Path("main_anomaly_contrastive.py")),
            "predictive": _sha256_file(Path("src/models/multiscale_forecaster.py")),
            "scoring": _sha256_file(Path("src/utils/hybrid_scoring.py")),
            "nasa_loader": _sha256_file(Path("src/data/nasa_loader.py")),
            "windowing": _sha256_file(Path("src/utils/predictive_benchmark.py")),
        },
        "cache_assumption": "seeded_training_with_exact_config_and_data",
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return metadata, cache_key


def _valid_cached_scores(scores, normal_datasets, evaluation_dataset):
    if scores is None or not set(SCORE_KEYS).issubset(scores):
        return False
    expected_lengths = {
        "neural_calibration": len(normal_datasets["calibration"]),
        "neural_validation": len(normal_datasets["validation"]),
        "neural_evaluation": len(evaluation_dataset),
        "predictive_calibration": len(normal_datasets["calibration"]),
        "predictive_validation": len(normal_datasets["validation"]),
        "predictive_evaluation": len(evaluation_dataset),
    }
    return all(np.asarray(scores[name]).size == length for name, length in expected_lengths.items())


def evaluate_channel(channel, metadata, existing_results, args, device):
    started = perf_counter()
    train_dataset, test_dataset, labels, bit_len = get_nasa_channel_data(
        channel,
        window_size=args.window_size,
        normalization_reference_fraction=0.6,
    )
    normalization = train_dataset.normalization_summary
    normal_datasets, normal_windows, bounds = _normal_segments(
        train_dataset,
        args.window_size,
    )
    evaluation_dataset, evaluation_windows, evaluation_labels, endpoints = _evaluation_data(
        test_dataset,
        labels,
        args.window_size,
        args.eval_step,
    )
    model_config = _model_config(bit_len, args)
    cache_metadata, cache_key = _cache_metadata(
        channel,
        model_config,
        bounds,
        normalization,
        args,
        device,
    )
    cache_path = args.cache_dir / f"{channel}_{cache_key}.npz"
    scores = None if args.force_recompute else load_score_cache(cache_path, cache_metadata)
    cache_hit = _valid_cached_scores(scores, normal_datasets, evaluation_dataset)
    if not cache_hit:
        seed_everything(args.seed)
        scores = _build_scores(
            normal_datasets,
            normal_windows,
            evaluation_dataset,
            evaluation_windows,
            model_config,
            args,
            device,
        )
        save_score_cache(cache_path, scores, cache_metadata)

    evaluate_fusion = (
        evaluate_robust_score_fusion
        if args.fusion_method == "robust_mahalanobis"
        else evaluate_normal_only_score_fusion
    )
    score_report = evaluate_fusion(
        {
            "neural_reconstruction": scores["neural_calibration"],
            "predictive_state": scores["predictive_calibration"],
        },
        {
            "neural_reconstruction": scores["neural_validation"],
            "predictive_state": scores["predictive_validation"],
        },
        {
            "neural_reconstruction": scores["neural_evaluation"],
            "predictive_state": scores["predictive_evaluation"],
        },
        evaluation_labels,
        quantile=args.quantile,
    )
    role = (
        "prior_comparison"
        if channel in PRIMARY_CHANNELS
        else "predeclared_holdout"
        if channel in DEVELOPMENT_HOLDOUT_CHANNELS
        else "mahalanobis_predeclared_validation"
        if channel in MAHALANOBIS_VALIDATION_CHANNELS
        else "upper_tail_predeclared_validation"
        if channel in UPPER_TAIL_VALIDATION_CHANNELS
        else "additional"
    )
    report = {
        "dataset": "NASA SMAP/MSL",
        "channel": channel,
        "spacecraft": metadata[channel]["spacecraft"],
        "role": role,
        **score_report,
        "protocol_details": {
            "window_size": args.window_size,
            "sample_step": args.eval_step,
            "evaluation_grain": "rolling_window_endpoint",
            "normal_raw_split_fractions": [0.6, 0.2, 0.2],
            "normalization": normalization,
            "normal_raw_segment_bounds": {name: list(value) for name, value in bounds.items()},
            "fit_windows": len(normal_datasets["fit"]),
            "calibration_windows": len(normal_datasets["calibration"]),
            "validation_windows": len(normal_datasets["validation"]),
            "calibration_reliability": {
                "minimum_windows": 200,
                "sufficient": bool(
                    len(normal_datasets["calibration"]) >= 200
                    and len(normal_datasets["validation"]) >= 200
                ),
            },
            "evaluation_windows": len(evaluation_dataset),
            "first_evaluation_endpoint": int(endpoints[0]),
            "last_evaluation_endpoint": int(endpoints[-1]),
            "test_labels_used_for_training": False,
            "test_labels_used_for_score_fusion": False,
            "test_labels_used_for_threshold": False,
            "test_labels_used_for_metrics_only": True,
            "point_adjusted_scope": "diagnostic_on_sampled_label_stream",
            "event_metric_scope": "sampled_label_stream_internal_regression",
        },
        "runtime_seconds": {
            "neural_training": float(np.asarray(scores.get("training_seconds", 0.0))),
            "neural_batch_scoring": float(
                np.asarray(scores.get("neural_scoring_seconds", 0.0))
            ),
            "predictive_fit_and_scoring": float(
                np.asarray(scores.get("predictive_fit_and_scoring_seconds", 0.0))
            ),
            "wall_including_loading": float(perf_counter() - started),
        },
        "training": {
            "seed": args.seed,
            "epochs": args.train_epochs,
            "max_batches_per_epoch": args.max_train_batches,
            "processed_batches": int(np.asarray(scores.get("processed_train_batches", 0))),
            "last_total_loss": float(np.asarray(scores.get("last_total_loss", 0.0))),
            "last_reconstruction_loss": float(
                np.asarray(scores.get("last_reconstruction_loss", 0.0))
            ),
        },
        "cache": {
            "hit": bool(cache_hit),
            "path": str(cache_path),
            "key": cache_key,
        },
    }
    existing_channel = existing_results.get("channels", {}).get(channel)
    if existing_channel:
        existing_point_f1 = float(existing_channel["pt_f1"])
        report["existing_ada_comparison"] = {
            "point_f1": existing_point_f1,
            "point_adjusted_f1": float(existing_channel["pa_f1"]),
            "hybrid_minus_existing_point_f1": float(
                report["fusion"]["point"]["f1"] - existing_point_f1
            ),
            "comparison_note": "same sampled endpoints, different normal-only calibration protocol",
        }
    return report


def _aggregate(channel_reports, selected_channels):
    reports = [channel_reports[channel] for channel in selected_channels if channel in channel_reports]
    if not reports:
        return None
    algorithms = {
        "hybrid": lambda report: report["fusion"],
        "neural_reconstruction": lambda report: report["components"]["neural_reconstruction"],
        "predictive_state": lambda report: report["components"]["predictive_state"],
    }
    aggregate = {"channels": [report["channel"] for report in reports]}
    for name, select in algorithms.items():
        aggregate[name] = {
            "average_precision": float(
                np.mean([select(report)["ranking"]["average_precision"] for report in reports])
            ),
            "average_point_f1": float(
                np.mean([select(report)["point"]["f1"] for report in reports])
            ),
            "average_point_adjusted_f1": float(
                np.mean([select(report)["point_adjusted"]["f1"] for report in reports])
            ),
        }
    aggregate["hybrid_minus_neural_point_f1"] = float(
        aggregate["hybrid"]["average_point_f1"]
        - aggregate["neural_reconstruction"]["average_point_f1"]
    )
    aggregate["hybrid_minus_predictive_point_f1"] = float(
        aggregate["hybrid"]["average_point_f1"]
        - aggregate["predictive_state"]["average_point_f1"]
    )
    comparable = [report for report in reports if "existing_ada_comparison" in report]
    if comparable:
        existing_average = float(
            np.mean([report["existing_ada_comparison"]["point_f1"] for report in comparable])
        )
        hybrid_comparable_average = float(
            np.mean([report["fusion"]["point"]["f1"] for report in comparable])
        )
        aggregate["existing_ada_comparable_channels"] = [
            report["channel"] for report in comparable
        ]
        aggregate["hybrid_comparable_average_point_f1"] = hybrid_comparable_average
        aggregate["existing_ada_average_point_f1"] = existing_average
        aggregate["hybrid_minus_existing_ada_point_f1"] = float(
            hybrid_comparable_average - existing_average
        )
    return aggregate


def _reliable_channels(channel_reports, selected_channels):
    return tuple(
        channel
        for channel in selected_channels
        if channel in channel_reports
        and channel_reports[channel]["protocol_details"]["calibration_reliability"][
            "sufficient"
        ]
    )


def _load_existing_results():
    path = Path("results/nasa_performance.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_identity(device):
    torch_device = torch.device(device)
    identity = {
        "type": torch_device.type,
        "index": torch_device.index,
        "torch_version": torch.__version__,
    }
    if torch_device.type == "cuda":
        index = torch_device.index if torch_device.index is not None else 0
        identity.update(
            {
                "name": torch.cuda.get_device_name(index),
                "cuda_version": torch.version.cuda,
            }
        )
    return identity


def _synchronize(device):
    if _is_cuda_device(device):
        torch.cuda.synchronize(torch.device(device))


def _is_cuda_device(device):
    return torch.device(device).type == "cuda"


def main():
    args = parse_args()
    if args.window_size <= max(args.scales):
        raise ValueError("window size must be larger than every predictive scale")
    if args.eval_step < 1 or args.train_epochs < 1 or args.max_train_batches < 1:
        raise ValueError("eval step, train epochs, and train batches must be positive")
    if args.device == "auto":
        device = resolve_torch_device()
    else:
        device = args.device.lower()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = str(torch_device)
    if torch_device.type == "cpu":
        torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or 1)))

    metadata = load_nasa_metadata()
    unknown = set(args.channels) - set(metadata)
    if unknown:
        raise ValueError(f"NASA channels not found: {sorted(unknown)}")
    existing_results = _load_existing_results()
    seed_everything(args.seed)
    print(f"Device: {device}; channels: {', '.join(args.channels)}")

    channel_reports = {}
    total_started = perf_counter()
    for channel in args.channels:
        channel_started = perf_counter()
        report = evaluate_channel(channel, metadata, existing_results, args, device)
        channel_reports[channel] = report
        print(
            f"{channel}: hybrid AP={report['fusion']['ranking']['average_precision']:.4f}, "
            f"F1={report['fusion']['point']['f1']:.4f}; "
            f"neural F1={report['components']['neural_reconstruction']['point']['f1']:.4f}; "
            f"predictive F1={report['components']['predictive_state']['point']['f1']:.4f}; "
            f"cache={'hit' if report['cache']['hit'] else 'miss'}; "
            f"wall={perf_counter() - channel_started:.2f}s"
        )

    result = {
        "schema_version": 1,
        "algorithm": f"normal_calibrated_neural_predictive_{args.fusion_method}",
        "generated_at": datetime.now().astimezone().isoformat(),
        "device": _device_identity(device),
        "seed": args.seed,
        "selection_policy": {
            "fusion_rule": args.fusion_method,
            "rule_selected_before_all_reported_channel_evaluation": bool(
                args.fusion_method == "bonferroni_min_p"
            ),
            "development_channel_labels_informed_rule_iteration": bool(
                args.fusion_method == "robust_mahalanobis"
            ),
            "rule_frozen_before_upper_tail_validation": True,
            "normalization_protocol_corrected_after_initial_upper_tail_run": True,
            "upper_tail_channels_used_for_rejection_not_rule_selection": True,
            "prior_comparison_channels": list(PRIMARY_CHANNELS),
            "development_holdout_channels": list(DEVELOPMENT_HOLDOUT_CHANNELS),
            "mahalanobis_validation_channels": list(MAHALANOBIS_VALIDATION_CHANNELS),
            "upper_tail_predeclared_validation_channels": list(
                UPPER_TAIL_VALIDATION_CHANNELS
            ),
            "test_labels_used_for_score_fit_or_threshold": False,
        },
        "channels": channel_reports,
        "summary": {
            "all": _aggregate(channel_reports, args.channels),
            "prior_comparison": _aggregate(channel_reports, PRIMARY_CHANNELS),
            "development_holdout": _aggregate(
                channel_reports,
                DEVELOPMENT_HOLDOUT_CHANNELS,
            ),
            "mahalanobis_validation": _aggregate(
                channel_reports,
                MAHALANOBIS_VALIDATION_CHANNELS,
            ),
            "mahalanobis_validation_reliable": _aggregate(
                channel_reports,
                _reliable_channels(channel_reports, MAHALANOBIS_VALIDATION_CHANNELS),
            ),
            "upper_tail_validation": _aggregate(
                channel_reports,
                UPPER_TAIL_VALIDATION_CHANNELS,
            ),
            "upper_tail_validation_reliable": _aggregate(
                channel_reports,
                _reliable_channels(channel_reports, UPPER_TAIL_VALIDATION_CHANNELS),
            ),
        },
        "runtime_seconds": {"total_wall": float(perf_counter() - total_started)},
    }
    final_validation = result["summary"]["upper_tail_validation_reliable"]
    final_delta = final_validation["hybrid_minus_neural_point_f1"]
    result["decision"] = {
        "status": (
            "candidate_for_additional_validation"
            if final_delta > 0.0
            else "rejected_as_universal_nasa_replacement"
        ),
        "basis_group": "upper_tail_validation_reliable",
        "primary_metric": "average_point_f1",
        "hybrid_minus_neural_point_f1": float(final_delta),
        "operational_default_changed": False,
        "operational_default": "existing_ada",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
