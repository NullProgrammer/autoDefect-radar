from __future__ import annotations

import shutil
from pathlib import Path

from src import config
from src.models import VehicleRecords
from src.normalizer import normalize_records


def build_knowledge_base(
    records: VehicleRecords,
    output_dir: Path = config.KNOWLEDGE_BASE_DIR,
    clear_existing: bool = True,
) -> list[Path]:
    if clear_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for document in normalize_records(records):
        path = document.path_under(output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document.content + "\n", encoding="utf-8")
        paths.append(path)
    return paths
