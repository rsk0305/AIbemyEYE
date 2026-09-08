# MSTS 전체 DB에서 공통 GT01 채널로 시험 실행 단위 알고리즘 검증을 수행하는 스크립트
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.msts_run_benchmark import run_msts_run_benchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("results/msts_gt01_core_1hz.npz"),
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=Path("results/msts_full_quality.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/msts_full_db_benchmark.json"),
    )
    return parser.parse_args()


def _load_canonical_runs(path, minimum_length=65):
    with np.load(path, allow_pickle=False) as source:
        values = np.asarray(source["values"], dtype=np.float64)
        starts = np.asarray(source["starts"], dtype=np.int64)
        lengths = np.asarray(source["lengths"], dtype=np.int64)
        run_ids = source["run_ids"].astype(str)
        containers = source["containers"].astype(str)
        paths = source["paths"].astype(str)
        core_channels = source["core_channels"].astype(str).tolist()
    candidates = {}
    discarded = []
    for start, length, run_id, container, item_path in zip(
        starts,
        lengths,
        run_ids,
        containers,
        paths,
    ):
        if length < minimum_length:
            discarded.append(
                {
                    "run_id": run_id,
                    "container": container,
                    "path": item_path,
                    "reason": "shorter_than_window",
                    "length": int(length),
                }
            )
            continue
        candidate = {
            "values": values[start : start + length],
            "container": container,
            "path": item_path,
            "length": int(length),
        }
        prior = candidates.get(run_id)
        if prior is None or _canonical_sort_key(candidate) < _canonical_sort_key(prior):
            if prior is not None:
                discarded.append(
                    {"run_id": run_id, **_record_without_values(prior), "reason": "duplicate_run"}
                )
            candidates[run_id] = candidate
        else:
            discarded.append(
                {"run_id": run_id, **_record_without_values(candidate), "reason": "duplicate_run"}
            )
    return candidates, discarded, core_channels


def _canonical_sort_key(record):
    path = record["path"].replace("\\", "/")
    if "_Converted/" in path:
        priority = 0
    elif "Converted_scaled" in path:
        priority = 1
    elif record["container"] != "filesystem" and "nonscaled" not in path.lower():
        priority = 2
    elif record["container"] == "filesystem":
        priority = 3
    else:
        priority = 4
    return priority, -record["length"], record["container"], path


def _record_without_values(record):
    return {
        "container": record["container"],
        "path": record["path"],
        "length": record["length"],
    }


def _scenario_partitions(canonical):
    train = sorted(
        run_id
        for run_id, record in canonical.items()
        if record["container"].startswith("RAWDATA")
    )
    calibration = sorted(
        run_id
        for run_id, record in canonical.items()
        if record["container"].startswith("210513")
    )
    normal_test = sorted(
        run_id
        for run_id, record in canonical.items()
        if record["container"].startswith("210517")
    )
    pcu_faults = sorted(
        run_id
        for run_id, record in canonical.items()
        if record["container"].startswith("210303")
    )
    ins_faults = sorted(
        run_id
        for run_id, record in canonical.items()
        if "BMON4=0008" in record["path"]
    )
    labels = {run_id: 0 for run_id in normal_test}
    labels.update({run_id: 1 for run_id in pcu_faults + ins_faults})
    all_selected = set(train + calibration + list(labels))
    unlabeled = sorted(set(canonical) - all_selected)
    return train, calibration, labels, unlabeled, {
        "normal": normal_test,
        "pcu_abort_associated": pcu_faults,
        "ins_bmon4_0008": ins_faults,
    }


def main():
    args = parse_args()
    canonical, discarded, core_channels = _load_canonical_runs(args.samples)
    train, calibration, labels, unlabeled, scenario_runs = _scenario_partitions(canonical)
    benchmark = run_msts_run_benchmark(
        {run_id: record["values"] for run_id, record in canonical.items()},
        train_run_ids=train,
        calibration_run_ids=calibration,
        labeled_test_labels=labels,
        unlabeled_run_ids=unlabeled,
    )
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "dataset": "MSTS full DB",
        "generated_at": datetime.now().astimezone().isoformat(),
        "core_channels": core_channels,
        "input_coverage": {
            "telemetry_files_scanned": quality["summary"]["telemetry_entry_count"],
            "telemetry_rows_scanned": quality["summary"]["row_count"],
            "sampled_gt01_files": 105,
            "canonical_runs": len(canonical),
            "discarded_samples": discarded,
        },
        "scenario_labels": {
            "label_grain": "run_level_weak_scenario_label",
            "normal_source": "210517 explicit normal test archive",
            "fault_sources": {
                "pcu_abort_associated": "210303 archive name",
                "ins_bmon4_0008": "210512 inner path",
            },
            "scenario_runs": scenario_runs,
            "unlabeled_runs_scored_without_metrics": len(unlabeled),
        },
        **benchmark,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictor = result["algorithms"]["multiscale_predictor"]
    print(
        f"runs={len(canonical)}; labeled={len(labels)}; "
        f"predictor_f1={predictor['run_metrics']['f1']:.4f}; "
        f"predictor_ap={predictor['run_average_precision']:.4f}"
    )


if __name__ == "__main__":
    main()
