"""Task 1 - store interest and conversion (passer-by -> interested -> entered/passed)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..geometry import Line, as_poly, in_poly


@dataclass
class Visitor:
    passer_by: bool                       # first seen on the mall walkway
    evidence: deque = field(default_factory=deque)   # (t, dt) attention samples
    interested_t: float | None = None
    interest_reason: str = ""
    was_outside: bool = False
    cross_t: float | None = None          # pending crossing awaiting confirmation
    entered_t: float | None = None
    looking: bool = False                 # attention evidence in the current frame


class StoreInterest:
    def __init__(self, cfg: dict):
        s, c, e = cfg["scene"], cfg["store_interest"], cfg["entry"]
        self.line = Line(*s["entrance_line"], s["inside_point"])
        self.outside = as_poly(s["outside_zone"])
        self.c, self.e = c, e
        self.v: dict[int, Visitor] = {}

    def update(self, t: float, dt: float, persons, events: list):
        for p in persons:
            v = self.v.get(p.gid)
            if v is None:
                v = self.v[p.gid] = Visitor(passer_by=in_poly(self.outside, p.foot))
            if not v.passer_by:
                continue
            sd = self.line.signed_dist(p.foot)
            if v.entered_t is None:
                self._attention(t, dt, p, v, sd, events)
                self._entry(t, p, v, sd, events)

    def _attention(self, t, dt, p, v: Visitor, sd, events):
        c = self.c
        v.looking = False
        if sd < 0:
            o = p.orient
            # Facing the store = facing vector has a component towards "inside".
            toward = o is not None and o.facing is not None and float(np.dot(o.facing, self.line.normal)) > 0
            frontal = o is not None and o.frontal
            # Standing still only means "window shopping" right at the glass;
            # someone waiting across the atrium is not looking at this store.
            stopped = (p.speed_valid and p.speed_bh < c["stationary_speed_bh_s"]
                       and -sd <= c["storefront_band_px"] and (o is None or o.facing is None or toward))
            v.looking = frontal or stopped
            if v.looking:
                v.evidence.append((t, dt))
        while v.evidence and t - v.evidence[0][0] > c["attention_window_s"]:
            v.evidence.popleft()
        if v.interested_t is None and sum(d for _, d in v.evidence) >= c["min_attention_s"]:
            v.interested_t, v.interest_reason = t, "attention"
            events.append(dict(t=t, type="store_interest", gid=p.gid))

    def _entry(self, t, p, v: Visitor, sd, events):
        e = self.e
        if sd < 0:
            v.was_outside, v.cross_t = True, None
            return
        along = self.line.along(p.foot)
        m = e["line_extent_margin"]
        if v.was_outside and v.cross_t is None and sd >= e["min_depth_px"] and -m <= along <= 1 + m:
            v.cross_t = t
        if v.cross_t is not None and t - v.cross_t >= e["confirm_s"]:
            v.entered_t = v.cross_t
            if v.interested_t is None and self.c["entering_implies_interest"]:
                v.interested_t, v.interest_reason = v.cross_t, "entered"
                events.append(dict(t=t, type="store_interest", gid=p.gid, reason="entered"))
            events.append(dict(t=t, type="store_entered", gid=p.gid))

    def status(self, gid: int) -> str | None:
        v = self.v.get(gid)
        if v is None or not v.passer_by:
            return None
        if v.entered_t is not None:
            return "entered"
        if v.interested_t is not None:
            return "interested"
        return "looking" if v.looking else None

    def counts(self, is_customer) -> dict:
        """Unique-person counts. `is_customer(gid)` filters out staff."""
        interested = [g for g, v in self.v.items()
                      if v.passer_by and v.interested_t is not None and is_customer(g)]
        entered = [g for g in interested if self.v[g].entered_t is not None]
        return dict(store_interest_count=len(interested), entered_count=len(entered),
                    passed_count=len(interested) - len(entered))
