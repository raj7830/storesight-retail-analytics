"""Task 2 - per-shelf customer interest episodes."""
from __future__ import annotations

import numpy as np

from ..geometry import angle_deg, as_poly, in_poly, poly_center, unit
from .episode import EpisodeTracker


class ShelfInterest:
    def __init__(self, cfg: dict):
        c = cfg["shelf_interest"]
        self.c = c
        self.shelves = {k: as_poly(v) for k, v in cfg["scene"]["shelves"].items()}
        self.centers = {k: poly_center(v) for k, v in self.shelves.items()}
        self.ep = EpisodeTracker(c["min_dwell_s"], c["max_gap_s"], c["end_gap_s"])
        self.events: list[dict] = []          # confirmed episodes (gid, shelf, t)

    def _engaged_shelf(self, p) -> str | None:
        c = self.c
        if not p.speed_valid or p.speed_bh > c["max_speed_bh_s"]:
            return None
        best, best_angle = None, 1e9
        for name, poly in self.shelves.items():
            if not in_poly(poly, p.foot, c["zone_margin_px"]):
                continue
            to_shelf = unit(self.centers[name] - p.foot)
            facing = p.orient.facing if p.orient else None
            # Unknown orientation (e.g. head occluded by the shelf itself) is
            # accepted: proximity + standing still is then the evidence.
            ang = 0.0 if facing is None or to_shelf is None else angle_deg(facing, to_shelf)
            if ang <= c["max_facing_angle_deg"] and ang < best_angle:
                best, best_angle = name, ang
        return best

    def update(self, t, dt, customers, left_gids, events: list):
        on = set()
        for p in customers:
            s = self._engaged_shelf(p)
            if s is not None:
                on.add((p.gid, s))
        confirmed, _ = self.ep.step(t, dt, on)
        self.ep.close_where(lambda k: k[0] in left_gids)
        for (gid, shelf), ep in confirmed:
            e = dict(t=t, type="shelf_interest", gid=gid, shelf=shelf, start=ep.start)
            self.events.append(e)
            events.append(e)

    def active(self) -> dict[int, str]:
        return {gid: shelf for (gid, shelf) in self.ep.active_confirmed()}

    def counts(self, is_customer) -> dict[str, int]:
        out = {k: 0 for k in self.shelves}
        for e in self.events:
            if is_customer(e["gid"]):
                out[e["shelf"]] += 1
        return out
