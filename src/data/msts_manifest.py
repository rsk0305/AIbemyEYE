# MSTS 압축 텔레메트리 파일의 재현 가능한 매니페스트를 생성하는 도구
from __future__ import annotations

import zipfile
import re
from pathlib import Path
from typing import Dict, List, Optional


def build_msts_manifest(root: Path) -> Dict[str, object]:
    """Index every archive entry and direct file without extracting the dataset."""
    source = Path(root)
    archives = sorted(path for path in source.glob("*.zip") if path.is_file())
    entries = []
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                entries.append(_archive_entry(archive_path, archive, info))

    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".zip":
            continue
        entries.append(_direct_file_entry(source, path))

    entries.sort(key=lambda entry: (entry["container"], entry["path"]))
    telemetry_entries = [entry for entry in entries if entry["file_type"] == "txt"]
    return {
        "schema_version": 1,
        "root": str(source),
        "entries": entries,
        "summary": {
            "archive_count": len(archives),
            "entry_count": len(entries),
            "telemetry_entry_count": len(telemetry_entries),
            "telemetry_groups": sorted(
                {entry["telemetry_group"] for entry in telemetry_entries}
                - {None}
            ),
        },
    }


def _archive_entry(archive_path: Path, archive: zipfile.ZipFile, info: zipfile.ZipInfo):
    file_type = Path(info.filename).suffix.lower().lstrip(".") or None
    entry = {
        "container": archive_path.name,
        "container_type": "zip",
        "path": info.filename,
        "file_type": file_type,
        "compressed_bytes": int(info.compress_size),
        "uncompressed_bytes": int(info.file_size),
        "telemetry_group": _telemetry_group(info.filename) if file_type == "txt" else None,
        "run_id": _run_id(info.filename),
        "column_count": None,
        "columns": None,
        "sample_interval_seconds": None,
        "time_status": None,
    }
    if file_type == "txt":
        entry.update(_inspect_telemetry(archive, info))
    return entry


def _direct_file_entry(root: Path, path: Path):
    file_type = path.suffix.lower().lstrip(".") or None
    entry = {
        "container": "filesystem",
        "container_type": "filesystem",
        "path": str(path.relative_to(root)),
        "file_type": file_type,
        "compressed_bytes": None,
        "uncompressed_bytes": int(path.stat().st_size),
        "telemetry_group": _telemetry_group(path.name) if file_type == "txt" else None,
        "run_id": _run_id(str(path.relative_to(root))),
        "column_count": None,
        "columns": None,
        "sample_interval_seconds": None,
        "time_status": None,
    }
    if file_type == "txt":
        with path.open("rb") as source:
            entry.update(_inspect_telemetry_lines(_first_nonempty_lines(source)))
    return entry


def _inspect_telemetry(archive: zipfile.ZipFile, info: zipfile.ZipInfo):
    with archive.open(info) as source:
        lines = _first_nonempty_lines(source)
    return _inspect_telemetry_lines(lines)


def _first_nonempty_lines(source) -> List[bytes]:
    lines = []
    while len(lines) < 3:
        line = source.readline()
        if not line:
            break
        if line.strip():
            lines.append(line)
    return lines


def _inspect_telemetry_lines(lines: List[bytes]):
    if not lines:
        return {
            "column_count": 0,
            "columns": [],
            "sample_interval_seconds": None,
            "time_status": "empty",
        }

    columns = _ascii_fields(lines[0].lstrip(b"#"))
    times = [_first_float(line) for line in lines[1:]]
    valid_times = [value for value in times if value is not None]
    interval = None
    status = "unavailable"
    if len(valid_times) >= 2:
        interval = valid_times[1] - valid_times[0]
        status = "increasing" if interval > 0 else "constant" if interval == 0 else "decreasing"
    return {
        "column_count": len(columns),
        "columns": columns,
        "sample_interval_seconds": interval,
        "time_status": status,
    }


def _ascii_fields(line: bytes) -> List[str]:
    return line.decode("ascii", errors="backslashreplace").split()


def _first_float(line: bytes) -> Optional[float]:
    fields = _ascii_fields(line)
    if not fields:
        return None
    try:
        return float(fields[0])
    except ValueError:
        return None


def _telemetry_group(path: str) -> Optional[str]:
    name = Path(path).name
    prefix = name.split("_", 1)[0]
    return prefix or None


def _run_id(path: str) -> Optional[str]:
    match = re.search(r"(20\d{6})_(\d{6})", path)
    return "_".join(match.groups()) if match else None
