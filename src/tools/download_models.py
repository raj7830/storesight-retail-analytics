"""Fetch every public model the pipeline uses, so later runs work offline.

    python -m src.tools.download_models
"""
from __future__ import annotations

import shutil
from pathlib import Path

import open_clip
from ultralytics import YOLO

from ..pipeline import load_config


def main():
    cfg = load_config("configs/default.yaml")
    for w in (cfg["model"]["weights"], cfg["model"]["roi_detector"]["weights"]):
        p = Path(w)
        if not p.exists():
            YOLO(p.name)                       # downloads into the working directory
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(p.name, p)
        print("ok", p)
    open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    print("ok CLIP ViT-B-32 laion2b_s34b_b79k")


if __name__ == "__main__":
    main()
