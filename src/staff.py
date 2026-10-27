"""Staff-vs-customer appearance classifier.

CLIP (ViT-B/32) turns each person crop into a 512-d embedding; a logistic
regression trained on labelled crops of this store's uniform
(src/tools/train_staff_classifier.py) scores it. Running only a frozen
embedding + linear layer makes "enrolling" a new store's uniform a matter of
dropping ~100 crops per class in data/staff_enrollment and retraining in
seconds - no detector fine-tuning needed.
"""
from __future__ import annotations

import pickle

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image


class StaffClassifier:
    def __init__(self, path: str, device: str):
        with open(path, "rb") as f:
            m = pickle.load(f)
        self.coef = m["coef"].reshape(-1)
        self.intercept = float(np.asarray(m["intercept"]).reshape(-1)[0])
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            m["clip"], pretrained=m["pretrained"])
        self.model.eval().to(device)

    @torch.no_grad()
    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        ims = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
            for c in crops_bgr]).to(self.device)
        e = self.model.encode_image(ims).float()
        e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()

    def predict(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Probability that each crop shows a staff member."""
        if not crops_bgr:
            return np.zeros(0)
        z = self.embed(crops_bgr) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))
