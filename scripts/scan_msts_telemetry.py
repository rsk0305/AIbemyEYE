# MSTS 전체 텔레메트리를 스트리밍 검사하고 GT01 공통 채널 표본을 저장하는 스크립트
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.msts_quality import scan_msts_telemetry


CORE_CHANNELS = (
    "GCU_PWR28P",
    "EBAT_VOLT",
    "GCU_PWR5P",
    "GCU_PWR28P_CUR",
    "GCU_TEMP",
    "ACU_EFIN_FB1",
    "ACU_EFIN_FB2",
    "ACU_EFIN_FB3",
    "ACU_EFIN_FB4",
    "ACU_EFIN_CUR1",
    "ACU_EFIN_CUR2",
    "ACU_EFIN_CUR3",
    "ACU_EFIN_CUR4",
    "MINS_Heading",
    "MINS_Pitch",
    "MINS_Roll",
    "MINS_Vn_L",
    "MINS_Ve_L",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\db_msts"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/msts_full_manifest.json"),
    )
    parser.add_argument(
        "--quality-output",
        type=Path,
        default=Path("results/msts_full_quality.json"),
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=Path("results/msts_gt01_core_1hz.npz"),
    )
    parser.add_argument("--sample-stride", type=int, default=100)
    return parser.parse_args()


def _save_samples(path: Path, samples):
    values = []
    starts = []
    lengths = []
    run_ids = []
    containers = []
    paths = []
    offset = 0
    for sample in samples:
        sequence = sample["values"]
        values.append(sequence)
        starts.append(offset)
        lengths.append(len(sequence))
        run_ids.append(sample["run_id"])
        containers.append(sample["container"])
        paths.append(sample["path"])
        offset += len(sequence)
    matrix = (
        np.concatenate(values, axis=0)
        if values
        else np.empty((0, len(CORE_CHANNELS)), dtype=np.float64)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=matrix,
        starts=np.asarray(starts, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
        run_ids=np.asarray(run_ids, dtype=str),
        containers=np.asarray(containers, dtype=str),
        paths=np.asarray(paths, dtype=str),
        core_channels=np.asarray(CORE_CHANNELS, dtype=str),
    )


def main():
    args = parse_args()
    if args.sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    quality, samples = scan_msts_telemetry(
        args.root,
        manifest,
        core_channels=CORE_CHANNELS,
        sample_stride=args.sample_stride,
    )
    quality["generated_at"] = datetime.now().astimezone().isoformat()
    quality["samples_output"] = str(args.samples_output)
    quality["sampled_run_count"] = len(samples)
    args.quality_output.parent.mkdir(parents=True, exist_ok=True)
    args.quality_output.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_samples(args.samples_output, samples)
    summary = quality["summary"]
    print(
        f"files={summary['telemetry_entry_count']}; rows={summary['row_count']}; "
        f"samples={len(samples)}"
    )


if __name__ == "__main__":
    main()
