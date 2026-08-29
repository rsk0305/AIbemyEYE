# NASA와 SWaT에서 경량 다중 시간축 예측 모델을 동일 규약으로 평가하는 스크립트
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.nasa_loader import get_nasa_channel_data, load_nasa_metadata
from src.data.swat_loader import get_swat_data
from src.utils.predictive_benchmark import (
    combine_feature_channels,
    disjoint_warmup_windows,
    evaluate_multiscale_forecaster,
    rolling_windows,
)
from src.utils.reproducibility import seed_everything


def evaluate_nasa(args):
    metadata = load_nasa_metadata()
    channel_reports = {}
    for channel in args.nasa_channels:
        if channel not in metadata:
            raise ValueError(f"NASA channel not found: {channel}")

        started = perf_counter()
        train_ds, test_ds, labels, _ = get_nasa_channel_data(
            channel,
            window_size=args.window_size,
        )
        train_sequence = combine_feature_channels(train_ds.telemetry, train_ds.bits)
        test_sequence = combine_feature_channels(test_ds.telemetry, test_ds.bits)
        train_windows, _ = rolling_windows(
            train_sequence,
            window_size=args.window_size,
        )
        evaluation_windows, endpoints = rolling_windows(
            test_sequence,
            window_size=args.window_size,
            step=args.nasa_eval_step,
        )
        report = evaluate_multiscale_forecaster(
            train_windows,
            evaluation_windows,
            labels[endpoints],
            scales=args.scales,
            quantile=args.quantile,
            ridge=args.ridge,
            partition_purge=args.window_size - 1,
        )
        report["dataset"] = "NASA SMAP/MSL"
        report["channel"] = channel
        report["protocol"].update(
            {
                "window_size": args.window_size,
                "sample_step": args.nasa_eval_step,
                "evaluation_grain": "rolling_window_endpoint",
                "point_adjusted_scope": "diagnostic_on_sampled_label_stream",
                "event_metric_scope": "sampled_label_stream_internal_regression",
            }
        )
        report["wall_seconds_including_loading"] = float(perf_counter() - started)
        channel_reports[channel] = report
        print(
            f"NASA {channel}: AP={report['ranking']['average_precision']:.4f}, "
            f"Pt-F1={report['point']['f1']:.4f}, "
            f"PA-F1={report['point_adjusted']['f1']:.4f}, "
            f"score={report['runtime_seconds']['evaluation_scoring']:.3f}s"
        )

        if args.nasa_warmup_fraction > 0.0:
            reference_windows, calibration_windows, warmup_end = disjoint_warmup_windows(
                test_sequence,
                window_size=args.window_size,
                warmup_fraction=args.nasa_warmup_fraction,
            )

            post_warmup = endpoints >= warmup_end + args.window_size - 1
            post_warmup_static = evaluate_multiscale_forecaster(
                train_windows,
                evaluation_windows[post_warmup],
                labels[endpoints[post_warmup]],
                scales=args.scales,
                quantile=args.quantile,
                ridge=args.ridge,
                partition_purge=args.window_size - 1,
            )
            post_warmup_static["dataset"] = "NASA SMAP/MSL"
            post_warmup_static["channel"] = channel
            post_warmup_static["protocol"].update(
                {
                    "window_size": args.window_size,
                    "sample_step": args.nasa_eval_step,
                    "evaluation_grain": "post_warmup_rolling_window_endpoint",
                    "warmup_fraction": args.nasa_warmup_fraction,
                    "warmup_raw_points": warmup_end,
                    "comparison_role": "post_warmup_static_control",
                    "point_adjusted_scope": "diagnostic_on_sampled_label_stream",
                    "event_metric_scope": "sampled_label_stream_internal_regression",
                }
            )
            adapted = evaluate_multiscale_forecaster(
                train_windows,
                evaluation_windows[post_warmup],
                labels[endpoints[post_warmup]],
                scales=args.scales,
                quantile=args.quantile,
                ridge=args.ridge,
                deployment_reference_windows=reference_windows,
                deployment_calibration_windows=calibration_windows,
                partition_purge=args.window_size - 1,
            )
            adapted["dataset"] = "NASA SMAP/MSL"
            adapted["channel"] = channel
            adapted["protocol"].update(
                {
                    "window_size": args.window_size,
                    "sample_step": args.nasa_eval_step,
                    "evaluation_grain": "post_warmup_rolling_window_endpoint",
                    "warmup_fraction": args.nasa_warmup_fraction,
                    "warmup_raw_points": warmup_end,
                    "warmup_assumption": "fixed_initial_unlabeled_normal_period",
                    "warmup_positive_labels": int(np.sum(labels[:warmup_end])),
                    "warmup_assumption_satisfied": bool(
                        np.sum(labels[:warmup_end]) == 0
                    ),
                    "comparison_role": "deployment_adapted",
                    "point_adjusted_scope": "diagnostic_on_sampled_label_stream",
                    "event_metric_scope": "sampled_label_stream_internal_regression",
                }
            )
            adapted["valid_for_comparison"] = bool(
                adapted["protocol"]["warmup_assumption_satisfied"]
            )
            report["post_warmup_static"] = post_warmup_static
            report["deployment_adapted"] = adapted
            print(
                f"NASA {channel} post-warmup static/adapted AP="
                f"{post_warmup_static['ranking']['average_precision']:.4f}/"
                f"{adapted['ranking']['average_precision']:.4f}, "
                f"Pt-F1={adapted['point']['f1']:.4f}, "
                f"PA-F1={adapted['point_adjusted']['f1']:.4f}"
            )
        report["channel_wall_seconds"] = float(perf_counter() - started)

    nasa_report = {
        "channels": channel_reports,
        "average_precision": _mean_metric(channel_reports, "ranking", "average_precision"),
        "average_point_f1": _mean_metric(channel_reports, "point", "f1"),
        "average_point_adjusted_f1": _mean_metric(channel_reports, "point_adjusted", "f1"),
        "total_wall_seconds": float(
            sum(report["channel_wall_seconds"] for report in channel_reports.values())
        ),
    }
    if args.nasa_warmup_fraction > 0.0:
        post_warmup_reports = {
            channel: report["post_warmup_static"]
            for channel, report in channel_reports.items()
        }
        adapted_reports = {
            channel: report["deployment_adapted"]
            for channel, report in channel_reports.items()
            if report["deployment_adapted"]["valid_for_comparison"]
        }
        nasa_report["post_warmup_static_average_precision"] = _mean_metric(
            post_warmup_reports,
            "ranking",
            "average_precision",
        )
        nasa_report["post_warmup_static_average_point_f1"] = _mean_metric(
            post_warmup_reports,
            "point",
            "f1",
        )
        nasa_report["deployment_adapted_valid_channels"] = sorted(adapted_reports)
        nasa_report["deployment_adapted_average_precision"] = (
            _mean_metric(adapted_reports, "ranking", "average_precision")
            if adapted_reports
            else None
        )
        nasa_report["deployment_adapted_average_point_f1"] = (
            _mean_metric(adapted_reports, "point", "f1")
            if adapted_reports
            else None
        )
        nasa_report["deployment_adapted_average_point_adjusted_f1"] = (
            _mean_metric(adapted_reports, "point_adjusted", "f1")
            if adapted_reports
            else None
        )
    return nasa_report


def evaluate_swat(args):
    started = perf_counter()
    train_ds, test_ds, labels, _, _ = get_swat_data(
        window_size=args.window_size,
        train_subsample_step=args.swat_train_stride,
        test_subsample_step=args.swat_test_stride,
    )
    report = evaluate_multiscale_forecaster(
        train_ds.windows,
        test_ds.windows,
        labels,
        scales=args.scales,
        quantile=args.quantile,
        ridge=args.ridge,
        augment_targets=True,
        target_step=args.target_step,
    )
    report["dataset"] = "SWaT"
    report["protocol"].update(
        {
            "window_size": args.window_size,
            "train_window_stride": args.swat_train_stride,
            "test_window_stride": args.swat_test_stride,
            "sample_step": args.swat_test_stride,
            "evaluation_grain": "sampled_window_endpoint",
            "point_adjusted_scope": "diagnostic_on_sampled_label_stream",
            "event_metric_scope": "sampled_label_stream_internal_regression",
        }
    )
    report["wall_seconds_including_loading"] = float(perf_counter() - started)
    print(
        f"SWaT: AP={report['ranking']['average_precision']:.4f}, "
        f"Pt-F1={report['point']['f1']:.4f}, "
        f"PA-F1={report['point_adjusted']['f1']:.4f}, "
        f"score={report['runtime_seconds']['evaluation_scoring']:.3f}s"
    )
    return report


def _mean_metric(reports, section, metric):
    return float(np.mean([report[section][metric] for report in reports.values()]))


def _parse_csv_ints(value):
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _parse_csv_strings(value):
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=_parse_csv_strings, default=("nasa", "swat"))
    parser.add_argument("--nasa-channels", type=_parse_csv_strings, default=("A-1", "D-1", "M-1"))
    parser.add_argument("--output", type=Path, default=Path("results/multiscale_forecaster_performance.json"))
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--scales", type=_parse_csv_ints, default=(4, 16, 64))
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--target-step", type=int, default=4)
    parser.add_argument("--nasa-eval-step", type=int, default=10)
    parser.add_argument("--nasa-warmup-fraction", type=float, default=0.1)
    parser.add_argument("--swat-train-stride", type=int, default=3000)
    parser.add_argument("--swat-test-stride", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = set(args.datasets) - {"nasa", "swat"}
    if unknown:
        raise ValueError(f"unsupported datasets: {sorted(unknown)}")
    if not 0.0 <= args.nasa_warmup_fraction < 0.5:
        raise ValueError("nasa warmup fraction must be in [0, 0.5)")

    seed = seed_everything(42)
    result = {
        "schema_version": 1,
        "algorithm": "multiscale_probabilistic_forecaster",
        "generated_at": datetime.now().astimezone().isoformat(),
        "seed": seed,
        "datasets": {},
    }
    if "nasa" in args.datasets:
        result["datasets"]["nasa"] = evaluate_nasa(args)
    if "swat" in args.datasets:
        result["datasets"]["swat"] = evaluate_swat(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
