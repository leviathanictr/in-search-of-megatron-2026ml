"""
Полный пайплайн: данные → признаки → модель → submission.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from load_data import main as load_main
from features import main as feats_main
from train import main as train_main


def main():
    here = Path(__file__).resolve().parent.parent
    data = here / "data" / "data_set_1.csv"
    artifacts = here / "artifacts"
    if not data.exists():
        sys.exit(f"data_set_1.csv not found at {data}\nSee data/README.md")

    if not (artifacts / "features_basic.parquet").exists():
        load_main(data, artifacts)
    if not (artifacts / "features_full.parquet").exists() or not (artifacts / "features_extra.parquet").exists():
        feats_main(artifacts)

    train_main(artifacts)


if __name__ == "__main__":
    main()
