"""Single-pass, causal analytics pipeline.

    frame ─► YOLO-pose + ByteTrack ─► identity stitching ─► staff classifier
          ─► store interest / shelf interest / interactions ─► overlay + events

`process()` is called once per analysed frame, in order, and never looks
ahead, so the same object serves an uploaded file and a live RTSP camera.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import yaml

from .analytics.interaction import Interactions
from .analytics.shelf import ShelfInterest
from .analytics.store import StoreInterest
from .render import Renderer
from .staff import StaffClassifier
from .tracking import Detector, IdentityManager


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Models:
    """Heavy models, loaded once and shared between jobs (web server)."""

    def __init__(self, cfg: dict):
        self.device = pick_device(cfg["runtime"]["device"])
        self.staff = StaffClassifier(cfg["staff"]["classifier"], self.device)
        self.cfg = cfg


class Pipeline:
    def __init__(self, cfg: dict, video_id: str, models: Models):
        self.cfg, self.video_id = cfg, video_id
        self.detector = Detector(cfg, models.device)      # fresh tracker state per video
        self.staff_clf = models.staff
        self.ids = IdentityManager(cfg)
        self.store = StoreInterest(cfg)
        self.shelf = ShelfInterest(cfg)
        self.inter = Interactions(cfg)
        self.renderer = Renderer(cfg)
        self.events: list[dict] = []
        self.t = 0.0
        self.frames = 0
        self.proc_time = 0.0
        sc = cfg["staff"]
        self.min_samples, self.lock_samples, self.lock_margin = sc["min_samples"], sc["lock_samples"], sc["lock_margin"]
        self.min_life = cfg["identity"]["min_track_life_s"]

    # -- roles -----------------------------------------------------------
    def role(self, gid: int) -> str:
        p = self.ids.get(gid)
        return p.role(self.min_samples) if p else "unknown"

    def _classify_staff(self, t, frame, persons):
        sc = self.cfg["staff"]
        due = [p for p in persons
               if p.staff_locked is None and t >= p.next_staff_sample_t
               and p.det.height >= sc["min_crop_height_px"]]
        if not due:
            return
        crops = []
        for p in due:
            x1, y1, x2, y2 = p.det.bbox.astype(int)
            crops.append(frame[max(y1, 0):y2, max(x1, 0):x2])
        for p, prob in zip(due, self.staff_clf.predict(crops)):
            p.staff_probs.append(float(prob))
            p.next_staff_sample_t = t + sc["sample_every_s"]
            if len(p.staff_probs) >= self.lock_samples and abs(p.staff_score - 0.5) >= self.lock_margin:
                p.staff_locked = p.staff_score > 0.5

    # -- main step -------------------------------------------------------
    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        t0 = time.perf_counter()
        dt = t - self.t if self.frames else 0.0
        self.t = t
        self.frames += 1

        dets = self.detector(frame)
        persons, left = self.ids.update(t, frame, dets)
        left_gids = {p.gid for p in left}
        self._classify_staff(t, frame, persons)

        new_events: list[dict] = []
        not_staff = [p for p in persons if p.role(self.min_samples) != "staff"]
        self.store.update(t, dt, not_staff, new_events)
        self.shelf.update(t, dt, [p for p in persons if p.role(self.min_samples) == "customer"],
                          left_gids, new_events)
        self.inter.update(t, dt, persons, left_gids, self.role, new_events)
        for e in new_events:
            e["video_id"] = self.video_id
        self.events.extend(new_events)

        out = self.renderer.draw(frame, t, persons, self, self.live_metrics())
        self.proc_time += time.perf_counter() - t0
        return out

    # -- metrics ---------------------------------------------------------
    def _valid(self, gid: int) -> bool:
        p = self.ids.get(gid)
        return p is not None and p.life >= self.min_life

    def _is_customer(self, gid: int) -> bool:
        return self._valid(gid) and self.role(gid) != "staff"

    def staff_instances(self) -> dict[int, int]:
        """Map every staff identity (gid) to a staff *instance* id.

        A staff member who drops out of detection for a few seconds (bending
        behind the counter, hidden by a shelf) gets a new track, but has not
        left the camera view - it is the same instance. Staff
        identities that follow each other within `staff_merge_gap_s` and look
        alike are therefore merged; longer absences start a new instance
        (re-identification after leaving the view is not required)."""
        ic = self.cfg["identity"]
        staff = sorted((p for p in self.ids.all_persons()
                        if p.life >= self.min_life and p.role(self.min_samples) == "staff"),
                       key=lambda p: p.first_t)
        groups: list[dict] = []
        out: dict[int, int] = {}
        for p in staff:
            best, best_gap = None, None
            for g in groups:
                gap = p.first_t - g["last_t"]
                if not 0 <= gap <= ic["staff_merge_gap_s"]:
                    continue
                if p.appearance is not None and g["app"] is not None and 1.0 - cv2.compareHist(
                        p.appearance, g["app"], cv2.HISTCMP_BHATTACHARYYA) < ic["staff_merge_min_appearance"]:
                    continue
                if best is None or gap < best_gap:
                    best, best_gap = g, gap
            if best is None:
                best = dict(id=p.gid, last_t=p.last_t, app=p.appearance)
                groups.append(best)
            best["last_t"] = max(best["last_t"], p.last_t)
            out[p.gid] = best["id"]
        return out

    def live_metrics(self) -> dict:
        staff = self.staff_instances()
        per = self.inter.per_staff(staff, self._is_customer)
        n_staff = len(set(staff.values()))
        m = dict(video_id=self.video_id, t=round(self.t, 1), **self.store.counts(self._is_customer))
        m["staff_instances"] = n_staff
        m["interaction_sessions"] = sum(per.values())
        m["avg_customer_interactions_per_staff"] = round(sum(per.values()) / n_staff, 2) if n_staff else 0.0
        m["per_staff"] = {str(k): v for k, v in per.items()}
        m["shelves"] = self.shelf.counts(self._is_customer)
        m["people_now"] = len([p for p in self.ids.persons.values() if p.active])
        m["fps"] = round(self.frames / self.proc_time, 1) if self.proc_time else 0.0
        return m

    def video_row(self) -> dict:
        m = self.live_metrics()
        return {k: m[k] for k in ("video_id", "store_interest_count", "entered_count", "passed_count",
                                  "staff_instances", "avg_customer_interactions_per_staff")}

    def shelf_rows(self) -> list[dict]:
        return [dict(video_id=self.video_id, shelf_id=k, interest_events=v)
                for k, v in self.shelf.counts(self._is_customer).items()]
