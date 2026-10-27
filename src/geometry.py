"""Small 2-D geometry helpers (image coordinates, y grows downwards)."""
from __future__ import annotations

import math

import cv2
import numpy as np


def as_poly(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def in_poly(poly: np.ndarray, p, margin: float = 0.0) -> bool:
    """True if p lies inside poly, or within `margin` px of its border."""
    d = cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True)
    return d >= -margin


def poly_center(poly: np.ndarray) -> np.ndarray:
    return poly.reshape(-1, 2).mean(axis=0)


class Line:
    """Directed entrance line with an 'inside' side."""

    def __init__(self, a, b, inside_point):
        self.a = np.asarray(a, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        d = self.b - self.a
        self.length = float(np.linalg.norm(d))
        self.dir = d / self.length
        n = np.array([-self.dir[1], self.dir[0]], dtype=np.float32)
        # Orient the normal so that it points into the store.
        if np.dot(np.asarray(inside_point, np.float32) - self.a, n) < 0:
            n = -n
        self.normal = n

    def signed_dist(self, p) -> float:
        """> 0 inside the store, < 0 outside (pixels)."""
        return float(np.dot(np.asarray(p, np.float32) - self.a, self.normal))

    def along(self, p) -> float:
        """Projection of p on the line, 0 at a and 1 at b."""
        return float(np.dot(np.asarray(p, np.float32) - self.a, self.dir)) / self.length


def unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-6 else np.asarray(v, np.float32) / n


def angle_deg(u, v) -> float:
    """Unsigned angle between two (unit) vectors in degrees."""
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(c))
