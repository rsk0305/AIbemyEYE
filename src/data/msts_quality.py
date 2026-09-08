# MSTS 텔레메트리 전체 파일을 스트리밍으로 품질 검사하고 공통 채널을 추출하는 도구
from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def scan_msts_telemetry(
    root: Path,
    manifest: Dict[str, object],
    core_channels: Sequence[str],
    sample_stride: int = 100,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Stream every telemetry file and retain only downsampled complete GT01_1 runs."""
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    channels = tuple(core_channels)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("core_channels must be non-empty and unique")

    source = Path(root)
    telemetry_entries = [
        entry for entry in manifest["entries"] if entry["file_type"] == "txt"
    ]
    quality_entries = []
    sampled_runs = []
    zip_groups = defaultdict(list)
    direct_entries = []
    for entry in telemetry_entries:
        if entry["container_type"] == "zip":
            zip_groups[entry["container"]].append(entry)
        else:
            direct_entries.append(entry)

    for container, entries in sorted(zip_groups.items()):
        with zipfile.ZipFile(source / container) as archive:
            for entry in entries:
                with archive.open(entry["path"]) as stream:
                    quality, sample = _scan_stream(
                        stream,
                        entry,
                        channels,
                        sample_stride,
                    )
                quality_entries.append(quality)
                if sample is not None:
                    sampled_runs.append(sample)

    for entry in direct_entries:
        with (source / entry["path"]).open("rb") as stream:
            quality, sample = _scan_stream(
                stream,
                entry,
                channels,
                sample_stride,
            )
        quality_entries.append(quality)
        if sample is not None:
            sampled_runs.append(sample)

    return {
        "schema_version": 1,
        "core_channels": list(channels),
        "sample_stride": sample_stride,
        "entries": quality_entries,
        "summary": _quality_summary(quality_entries),
    }, sampled_runs


def _scan_stream(stream, entry, core_channels, sample_stride):
    header = stream.readline()
    columns = _fields(header.lstrip(b"#"))
    core_indices = _core_indices(columns, core_channels)
    core_complete = core_indices is not None
    is_gt01_primary = Path(entry["path"]).name.startswith("GT01_1_")
    row_count = 0
    width_mismatches = 0
    invalid_time_rows = 0
    time_regressions = 0
    invalid_core_samples = 0
    last_time = None
    sampled_values = []
    for raw_line in stream:
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        row_count += 1
        if len(fields) != len(columns):
            width_mismatches += 1
        current_time = _float_at(fields, 0)
        if current_time is None:
            invalid_time_rows += 1
        elif last_time is not None and current_time < last_time:
            time_regressions += 1
        if current_time is not None:
            last_time = current_time

        if (
            core_complete
            and is_gt01_primary
            and (row_count - 1) % sample_stride == 0
        ):
            values = [_float_at(fields, index) for index in core_indices]
            if any(value is None or not np.isfinite(value) for value in values):
                invalid_core_samples += 1
            else:
                sampled_values.append(values)

    missing_core = [channel for channel in core_channels if channel not in columns]
    quality = {
        "container": entry["container"],
        "container_type": entry["container_type"],
        "path": entry["path"],
        "run_id": entry["run_id"],
        "telemetry_group": entry["telemetry_group"],
        "row_count": row_count,
        "width_mismatches": width_mismatches,
        "invalid_time_rows": invalid_time_rows,
        "time_regressions": time_regressions,
        "core_complete": core_complete,
        "missing_core_channels": missing_core,
        "sample_count": len(sampled_values),
        "invalid_core_samples": invalid_core_samples,
    }
    sample = None
    if sampled_values:
        sample = {
            "container": entry["container"],
            "path": entry["path"],
            "run_id": entry["run_id"],
            "values": np.asarray(sampled_values, dtype=np.float64),
        }
    return quality, sample


def _quality_summary(entries: Iterable[Dict[str, object]]) -> Dict[str, int]:
    records = list(entries)
    return {
        "telemetry_entry_count": len(records),
        "row_count": int(sum(record["row_count"] for record in records)),
        "width_mismatch_count": int(sum(record["width_mismatches"] for record in records)),
        "invalid_time_row_count": int(sum(record["invalid_time_rows"] for record in records)),
        "time_regression_count": int(sum(record["time_regressions"] for record in records)),
        "core_complete_entry_count": int(sum(record["core_complete"] for record in records)),
        "sampled_entry_count": int(sum(record["sample_count"] > 0 for record in records)),
    }


def _fields(line: bytes) -> List[str]:
    return line.decode("ascii", errors="backslashreplace").split()


def _core_indices(columns: Sequence[str], core_channels: Sequence[str]):
    index_by_name = {name: index for index, name in enumerate(columns)}
    if any(channel not in index_by_name for channel in core_channels):
        return None
    return tuple(index_by_name[channel] for channel in core_channels)


def _float_at(fields: Sequence[bytes], index: int):
    if index >= len(fields):
        return None
    try:
        return float(fields[index])
    except ValueError:
        return None
