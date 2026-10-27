"""Dump person crops from a video to label for staff enrolment.

    python -m src.tools.dump_crops Input_Video/sample1.mp4 --every 2.0 --out data/unlabelled
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from ..pipeline import load_config, pick_device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--min-height", type=int, default=60)
    ap.add_argument("--out", default="data/unlabelled")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    model = YOLO(cfg["model"]["weights"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) * a.every))
    stem, i, n = Path(a.video).stem, 0, 0
    while cap.grab():
        if i % step == 0:
            _, frame = cap.retrieve()
            r = model(frame, imgsz=cfg["model"]["imgsz"], conf=0.4, classes=[0], verbose=False,
                      device=pick_device(cfg["runtime"]["device"]))[0]
            for k, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                if y2 - y1 >= a.min_height:
                    cv2.imwrite(str(out / f"{stem}_f{i:05d}_{k}.jpg"), frame[max(y1, 0):y2, max(x1, 0):x2])
                    n += 1
        i += 1
    print(f"wrote {n} crops to {out}")


if __name__ == "__main__":
    main()
