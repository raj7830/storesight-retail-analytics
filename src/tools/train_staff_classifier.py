"""(Re)train the staff-uniform classifier from labelled person crops.

    python -m src.tools.train_staff_classifier \
        --data data/staff_enrollment --out models/staff_classifier.pkl

data/staff_enrollment/{staff,customer}/*.jpg are person crops (any size).
File names start with "<video>_f<frame>_" so cross-validation can hold out
whole minutes of footage at a time (crops of the same person a few frames
apart would otherwise leak between folds and inflate the score).

To enrol a new store: dump crops with `python -m src.tools.dump_crops`,
sort them into the two folders, and run this script. Takes seconds.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict

from ..pipeline import pick_device
from ..staff import StaffClassifier

CLIP_ARCH, CLIP_WEIGHTS = "ViT-B-32", "laion2b_s34b_b79k"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/staff_enrollment")
    ap.add_argument("--out", default="models/staff_classifier.pkl")
    ap.add_argument("--C", type=float, default=2.0, help="inverse L2 regularisation strength")
    a = ap.parse_args(argv)

    files, labels = [], []
    for label, sub in ((1, "staff"), (0, "customer")):
        for f in sorted((Path(a.data) / sub).glob("*.jpg")):
            files.append(f)
            labels.append(label)
    y = np.array(labels)
    print(f"{y.sum()} staff / {len(y) - y.sum()} customer crops")

    # Embedding only needs the CLIP backbone: build the extractor with a
    # placeholder linear head.
    tmp = Path(a.out).with_suffix(".tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(dict(coef=np.zeros(512, np.float32), intercept=np.zeros(1, np.float32),
                         clip=CLIP_ARCH, pretrained=CLIP_WEIGHTS), f)
    emb = StaffClassifier(str(tmp), pick_device("auto"))
    tmp.unlink()
    crops = [cv2.imread(str(f)) for f in files]
    X = np.vstack([emb.embed(crops[i:i + 64]) for i in range(0, len(crops), 64)])

    # Group = video + minute, so folds never share near-identical frames.
    groups = []
    for f in files:
        m = re.match(r"(.+?)_f(\d+)_", f.name)
        groups.append(f"{m.group(1)}_{int(m.group(2)) // 1800}" if m else f.name)
    make = lambda: LogisticRegression(C=a.C, class_weight="balanced", max_iter=3000)
    p = cross_val_predict(make(), X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    print("Grouped 5-fold cross-validation (per crop):")
    print(classification_report(y, p > 0.5, target_names=["customer", "staff"], digits=3))

    clf = make().fit(X, y)
    with open(a.out, "wb") as f:
        pickle.dump(dict(coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
                         clip=CLIP_ARCH, pretrained=CLIP_WEIGHTS), f)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
