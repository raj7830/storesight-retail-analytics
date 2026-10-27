"""Attention direction from COCO-17 body keypoints.

The camera is mounted high and looks down at an angle, so "facing direction"
is estimated in the image plane:

* Head direction (preferred, it is what "looking" means): vector from the
  centre of the visible ears to the nose. Frontal face -> points down the
  image (towards the camera); profile -> points left/right.
* Body direction (fallback when the face is hidden, e.g. seen from behind):
  perpendicular of the shoulder line. Pose models label left/right shoulders
  from appearance, so the perpendicular's sign gives front vs. back: with the
  person facing the camera their LEFT shoulder appears on the image RIGHT.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import unit

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = range(7)


@dataclass
class Orientation:
    facing: np.ndarray | None   # unit vector in image coords, None if unknown
    frontal: bool               # face turned towards the camera


def estimate(kps: np.ndarray, bbox_h: float, kpt_conf: float) -> Orientation:
    vis = kps[:, 2] >= kpt_conf
    xy = kps[:, :2]

    frontal = bool(vis[NOSE] and vis[L_EYE] and vis[R_EYE])
    if frontal and vis[L_SH] and vis[R_SH]:
        # Both eyes visible but shoulders mirrored -> the model is guessing.
        frontal = xy[L_SH, 0] > xy[R_SH, 0] - 0.02 * bbox_h

    facing = None
    if vis[NOSE] and (vis[L_EAR] or vis[R_EAR]):
        ears = xy[[i for i in (L_EAR, R_EAR) if vis[i]]].mean(axis=0)
        v = xy[NOSE] - ears
        # Ignore degenerate vectors (head ~1-2 px across at the back of the mall).
        if np.linalg.norm(v) > 0.015 * bbox_h:
            facing = unit(v)
    if facing is None and vis[L_SH] and vis[R_SH]:
        s = xy[L_SH] - xy[R_SH]
        # Shoulders collapse to a point when side-on; direction is then unreliable.
        if np.linalg.norm(s) > 0.10 * bbox_h:
            facing = unit(np.array([-s[1], s[0]]))
    return Orientation(facing=facing, frontal=frontal)
