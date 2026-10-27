"""Detection + tracking + person identity.

YOLO-pose detects people and their keypoints, ByteTrack links detections
frame-to-frame, and IdentityManager sits on top of the tracker to merge track
fragments that belong to the same person (see `identity` in the config).
Everything is causal: a decision at time t only uses frames <= t.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from . import pose
from .geometry import as_poly, in_poly


@dataclass
class Detection:
    track_id: int
    bbox: np.ndarray        # x1, y1, x2, y2
    conf: float
    kps: np.ndarray         # 17 x 3 (x, y, conf)

    @property
    def foot(self) -> np.ndarray:
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y2], np.float32)

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class _TrackerInput:
    """Minimal Results-like container accepted by Ultralytics' BYTETracker."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy, self.conf = xyxy.reshape(-1, 4), conf.reshape(-1)
        self.cls = np.zeros_like(self.conf)

    @property
    def xywh(self):
        b = self.xyxy
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                         b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, m):
        return _TrackerInput(self.xyxy[m], self.conf[m])


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between box sets a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-6)


class Detector:
    """Pose model on the full frame + a plain detector on the mall walkway.

    Passers-by on the far side of the glass are only ~20x50 px and washed out
    by reflections: the pose model misses them entirely, while a detection
    model on an up-scaled crop of that strip finds them reliably. The two
    sets are merged (pose detections win on overlap, since they carry
    keypoints) and fed to one ByteTrack instance.
    """

    def __init__(self, cfg: dict, device: str):
        m = cfg["model"]
        self.pose = YOLO(m["weights"])
        self.kw = dict(imgsz=m["imgsz"], conf=m["conf"], iou=m["iou"], classes=[0],
                       device=device, verbose=False)
        r = m.get("roi_detector")
        self.roi = None
        if r:
            self.roi_model = YOLO(r["weights"])
            self.roi = [int(v) for v in r["box"]]           # x1, y1, x2, y2
            self.roi_kw = dict(self.kw, imgsz=r["imgsz"], conf=r["conf"])
        with open(m["tracker"]) as f:
            self.tracker = BYTETracker(SimpleNamespace(**yaml.safe_load(f)))
        self.ignore = [as_poly(z) for z in cfg["scene"].get("ignore_zones", [])]

    def _detect(self, frame):
        r = self.pose(frame, **self.kw)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        kps = r.keypoints.data.cpu().numpy() if len(boxes) else np.zeros((0, 17, 3), np.float32)
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            rr = self.roi_model(frame[y1:y2, x1:x2], **self.roi_kw)[0]
            rb = rr.boxes.xyxy.cpu().numpy() + np.array([x1, y1, x1, y1], np.float32)
            rc = rr.boxes.conf.cpu().numpy()
            keep = _iou(rb, boxes).max(axis=1) < 0.4 if len(boxes) else np.ones(len(rb), bool)
            if keep.any():
                boxes = np.concatenate([boxes, rb[keep]])
                conf = np.concatenate([conf, rc[keep]])
                kps = np.concatenate([kps, np.zeros((int(keep.sum()), 17, 3), np.float32)])
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1) if len(boxes) else np.zeros((0, 2))
        ok = np.array([not any(in_poly(z, f) for z in self.ignore) for f in feet], bool)
        return boxes[ok], conf[ok], kps[ok]

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        boxes, conf, kps = self._detect(frame)
        tracks = self.tracker.update(_TrackerInput(boxes, conf), frame)
        if len(tracks) == 0 or len(boxes) == 0:
            return []
        # Map each confirmed track back to the detection it was matched with
        # this frame (highest IoU) to recover its raw box and keypoints.
        iou = _iou(tracks[:, :4], boxes)
        out = []
        for ti, tr in enumerate(tracks):
            j = int(iou[ti].argmax())
            if iou[ti, j] < 0.5:
                continue
            out.append(Detection(int(tr[4]), boxes[j], float(conf[j]), kps[j]))
        return out


def appearance_hist(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """Hue-saturation histogram of the upper 60% of the body (clothes, hair)."""
    x1, y1, x2, y2 = bbox.astype(int)
    y2 = y1 + int(0.6 * (y2 - y1))
    crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


@dataclass
class Person:
    gid: int
    first_t: float
    last_t: float
    det: Detection
    track_ids: set = field(default_factory=set)
    hist: deque = field(default_factory=lambda: deque(maxlen=40))   # (t, foot)
    appearance: np.ndarray | None = None
    app_ref: np.ndarray | None = None      # slow-moving reference look (ID-swap check)
    mismatch: int = 0
    orient: pose.Orientation | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    speed_bh: float = 0.0
    speed_valid: bool = False
    staff_probs: list = field(default_factory=list)
    staff_locked: bool | None = None
    next_staff_sample_t: float = 0.0
    active: bool = True

    @property
    def foot(self):
        return self.det.foot

    @property
    def h(self):
        return max(self.det.height, 1.0)

    @property
    def staff_score(self) -> float:
        return float(np.mean(self.staff_probs)) if self.staff_probs else 0.0

    def role(self, min_samples: int) -> str:
        if self.staff_locked is not None:
            return "staff" if self.staff_locked else "customer"
        if len(self.staff_probs) < min_samples:
            return "unknown"
        return "staff" if self.staff_score > 0.5 else "customer"

    @property
    def life(self) -> float:
        return self.last_t - self.first_t


class IdentityManager:
    def __init__(self, cfg: dict):
        c = cfg["identity"]
        self.max_gap = c["stitch_max_gap_s"]
        self.max_dist = c["stitch_max_dist_bh"]
        self.min_app = c["stitch_min_appearance"]
        self.split_sim = c["split_max_appearance"]
        self.split_frames = c["split_frames"]
        self.splits = 0
        self.kpt_conf = cfg["model"]["kpt_conf"]
        self.tid2gid: dict[int, int] = {}
        self.persons: dict[int, Person] = {}
        self.finished: dict[int, Person] = {}
        self._next = 1

    def _stitch(self, t: float, det: Detection, app, busy: set) -> int | None:
        best, best_score = None, 0.0
        for p in self.persons.values():
            gap = t - p.last_t
            if p.active or p.gid in busy or gap > self.max_gap:
                continue
            # Allowed jump grows with the gap (people keep walking while hidden).
            dist = np.linalg.norm(det.foot - p.foot) / max(p.h, det.height)
            if dist > self.max_dist * (1.0 + gap):
                continue
            sim = 1.0
            if app is not None and p.appearance is not None:
                sim = 1.0 - cv2.compareHist(app, p.appearance, cv2.HISTCMP_BHATTACHARYYA)
            if sim < self.min_app:
                continue
            score = sim - 0.2 * dist
            if score > best_score:
                best, best_score = p.gid, score
        return best

    def update(self, t: float, frame: np.ndarray, dets: list[Detection]):
        """Returns (active persons, persons that left the scene this step)."""
        seen = set()
        for det in dets:
            app = appearance_hist(frame, det.bbox)
            gid = self.tid2gid.get(det.track_id)
            if gid is None:
                gid = self._stitch(t, det, app, seen)
                if gid is None:
                    gid = self._next
                    self._next += 1
                    self.persons[gid] = Person(gid, t, t, det)
                self.tid2gid[det.track_id] = gid
            if gid in seen:      # two boxes claim one identity -> keep the first
                continue
            if gid not in self.persons:
                # The tracker kept this track alive longer than our stitch
                # window: same track id means same person, so revive it.
                self.persons[gid] = self.finished.pop(gid)
            if self._swapped(self.persons[gid], app):
                # The tracker's box has jumped onto someone else (typical when
                # two people cross). Start a new identity for the track; the
                # old person goes "lost" and may be stitched back later.
                gid = self._next
                self._next += 1
                self.splits += 1
                self.persons[gid] = Person(gid, t, t, det)
                self.tid2gid[det.track_id] = gid
            seen.add(gid)
            p = self.persons[gid]
            p.det, p.last_t, p.active = det, t, True
            p.track_ids.add(det.track_id)
            if app is not None:
                p.appearance = app if p.appearance is None else 0.8 * p.appearance + 0.2 * app
                if p.app_ref is None:
                    p.app_ref = app.copy()
                elif p.mismatch == 0:
                    p.app_ref = 0.95 * p.app_ref + 0.05 * app
            p.hist.append((t, det.foot))
            p.orient = pose.estimate(det.kps, det.height, self.kpt_conf)
            self._kinematics(p)

        left = []
        for gid, p in list(self.persons.items()):
            if gid not in seen:
                p.active = False
                if t - p.last_t > self.max_gap:
                    left.append(self.persons.pop(gid))
        self.finished.update({p.gid: p for p in left})
        return [p for p in self.persons.values() if p.active], left

    def _swapped(self, p: Person, app) -> bool:
        """True after `split_frames` consecutive frames that look nothing like
        the person's reference appearance."""
        if app is None or p.app_ref is None:
            return False
        sim = 1.0 - cv2.compareHist(app, p.app_ref, cv2.HISTCMP_BHATTACHARYYA)
        p.mismatch = p.mismatch + 1 if sim < self.split_sim else 0
        if p.mismatch >= self.split_frames:
            p.mismatch = 0
            return True
        return False

    @staticmethod
    def _kinematics(p: Person):
        # Velocity = displacement between the median foot position of the
        # latest 0.4 s and of 1.0-1.4 s ago. Medians ignore single-frame box
        # jumps (occlusion, a brief ID swap) that would spike a plain
        # frame-to-frame or least-squares estimate.
        now = p.last_t
        recent = [f for tt, f in p.hist if now - tt <= 0.4]
        old = [f for tt, f in p.hist if 1.0 <= now - tt <= 1.4]
        if not recent or not old:
            p.speed_valid = False
            return
        old_t = np.median([tt for tt, _ in p.hist if 1.0 <= now - tt <= 1.4])
        new_t = np.median([tt for tt, _ in p.hist if now - tt <= 0.4])
        p.velocity = ((np.median(recent, axis=0) - np.median(old, axis=0)) / max(new_t - old_t, 1e-3)).astype(np.float32)
        p.speed_bh = float(np.linalg.norm(p.velocity) / p.h)
        p.speed_valid = True

    def get(self, gid: int) -> Person | None:
        return self.persons.get(gid) or self.finished.get(gid)

    def all_persons(self) -> list[Person]:
        return list(self.finished.values()) + list(self.persons.values())
