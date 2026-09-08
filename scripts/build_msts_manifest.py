# MSTS 전체 DB의 압축 및 추출 파일 매니페스트를 생성하는 스크립트
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.msts_manifest import build_msts_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\db_msts"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/msts_full_manifest.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_msts_manifest(args.root)
    manifest["generated_at"] = datetime.now().astimezone().isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = manifest["summary"]
    print(
        f"archives={summary['archive_count']}; entries={summary['entry_count']}; "
        f"telemetry={summary['telemetry_entry_count']}"
    )


if __name__ == "__main__":
    main()
