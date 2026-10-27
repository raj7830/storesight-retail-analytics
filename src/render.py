"""Annotated-frame renderer (boxes, IDs, roles, interest + interaction cues, live HUD)."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import as_poly, poly_center

STAFF = (0, 140, 255)       # orange (BGR)
CUSTOMER = (90, 210, 90)    # green
UNKNOWN = (180, 180, 180)
STORE = (0, 230, 255)       # yellow
SHELF = (255, 80, 255)      # magenta
INTERACT = (255, 200, 0)    # cyan-blue
LINE = (255, 128, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _tag(img, text, org, color, scale=0.45, fg=(0, 0, 0)):
    (w, h), b = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - b - 2), (x + w + 4, y + 1), color, -1)
    cv2.putText(img, text, (x + 2, y - b), FONT, scale, fg, 1, cv2.LINE_AA)
    return y - h - b - 3


def _center(b):
    return int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)


class Renderer:
    def __init__(self, cfg: dict):
        s = cfg["scene"]
        self.line = tuple(map(tuple, s["entrance_line"]))
        self.shelves = {k: as_poly(v).astype(np.int32) for k, v in s["shelves"].items()}
        self.centers = {k: poly_center(v) for k, v in self.shelves.items()}
        self.min_samples = cfg["staff"]["min_samples"]
        self.blur_faces = cfg.get("privacy", {}).get("blur_faces", False)

    @staticmethod
    def _pixelate_head(img, bbox):
        # Head = top ~22% of the body box (widened a little): coarse
        # pixelation hides identity while keeping the scene readable.
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = y2 - y1, x2 - x1
        hx1, hx2 = max(x1 - w // 10, 0), min(x2 + w // 10, img.shape[1])
        hy1, hy2 = max(y1 - h // 20, 0), min(y1 + int(0.22 * h), img.shape[0])
        roi = img[hy1:hy2, hx1:hx2]
        if roi.size == 0:
            return
        k = max(2, min(roi.shape[:2]) // 5)
        small = cv2.resize(roi, (max(1, roi.shape[1] // k), max(1, roi.shape[0] // k)), interpolation=cv2.INTER_AREA)
        img[hy1:hy2, hx1:hx2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    def draw(self, frame, t, persons, pl, metrics) -> np.ndarray:
        img = frame.copy()
        if self.blur_faces:
            for p in persons:
                self._pixelate_head(img, p.det.bbox)
        active_shelf = pl.shelf.active()
        busy = {s for s in active_shelf.values()}

        # Scene geometry.
        overlay = img.copy()
        for name, poly in self.shelves.items():
            if name in busy:
                cv2.fillPoly(overlay, [poly], SHELF)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        for name, poly in self.shelves.items():
            col = SHELF if name in busy else (0, 0, 230)
            cv2.polylines(img, [poly], True, col, 2)
            x, y = poly.reshape(-1, 2).min(axis=0)
            _tag(img, name.replace("_", " "), (x + 2, y + 18), col, 0.5, (255, 255, 255))
        cv2.line(img, self.line[0], self.line[1], LINE, 2, cv2.LINE_AA)
        _tag(img, "ENTRANCE", (self.line[1][0] - 70, self.line[1][1] - 4), LINE, 0.4, (255, 255, 255))

        by_gid = {p.gid: p for p in persons}

        # Interaction links under the boxes.
        for g1, g2 in pl.inter.active_pairs():
            a, b = by_gid.get(g1), by_gid.get(g2)
            if a is None or b is None:
                continue
            pa = _center(a.det.bbox)
            pb = _center(b.det.bbox)
            cv2.line(img, pa, pb, INTERACT, 3, cv2.LINE_AA)
            mid = ((pa[0] + pb[0]) // 2 - 40, (pa[1] + pb[1]) // 2)
            _tag(img, "INTERACTION", mid, INTERACT, 0.45)

        for p in persons:
            role = p.role(self.min_samples)
            col = STAFF if role == "staff" else CUSTOMER if role == "customer" else UNKNOWN
            x1, y1, x2, y2 = p.det.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            tags = [(f"STAFF {p.gid}" if role == "staff" else f"ID {p.gid}", col, (0, 0, 0))]
            st = pl.store.status(p.gid)
            if st == "entered":
                tags.append(("ENTERED", STORE, (0, 0, 0)))
            elif st == "interested":
                tags.append(("STORE INTEREST", STORE, (0, 0, 0)))
            elif st == "looking":
                tags.append(("looking...", (200, 255, 255), (0, 0, 0)))
            if p.gid in active_shelf:
                s = active_shelf[p.gid]
                tags.append((f"INTEREST {s.replace('_', ' ')}", SHELF, (255, 255, 255)))
                c = self.centers[s]
                cv2.line(img, tuple(p.foot.astype(int)), (int(c[0]), int(c[1])), SHELF, 1, cv2.LINE_AA)
            # Stack tags above the box, or below it when the box touches the top edge.
            if y1 > 18 * len(tags) + 4:
                y = y1
                for text, bg, fg in tags:
                    y = _tag(img, text, (x1, y), bg, 0.45, fg)
            else:
                y = y2 + 16
                for text, bg, fg in tags:
                    _tag(img, text, (x1, y), bg, 0.45, fg)
                    y += 17

            # Facing arrow from the head: makes the attention estimate auditable.
            if p.orient is not None and p.orient.facing is not None:
                head = np.array([(x1 + x2) / 2, y1 + 0.12 * (y2 - y1)])
                tip = head + p.orient.facing * max(18, 0.35 * (y2 - y1))
                cv2.arrowedLine(img, tuple(head.astype(int)), tuple(tip.astype(int)), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)

        self._hud(img, t, metrics)
        return img

    @staticmethod
    def _hud(img, t, m):
        lines = [
            f"{m['video_id']}   t={t:6.1f}s   {m['fps']:.1f} fps",
            f"Store interest: {m['store_interest_count']}  (entered {m['entered_count']} / passed {m['passed_count']})",
            "Shelf interest: " + "  ".join(f"{k.replace('_', ' ')}={v}" for k, v in m["shelves"].items()),
            f"Staff: {m['staff_instances']}   sessions: {m['interaction_sessions']}   avg/staff: {m['avg_customer_interactions_per_staff']:.2f}",
        ]
        w, lh = 470, 20
        x0, y0 = 8, img.shape[0] - lh * len(lines) - 18   # bottom-left: wall, never people
        roi = img[y0:y0 + lh * len(lines) + 10, x0:x0 + w]
        cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, roi)
        for i, s in enumerate(lines):
            cv2.putText(img, s, (x0 + 8, y0 + 18 + i * lh), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
