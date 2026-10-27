"""Task 3 - staff-customer interaction sessions."""
from __future__ import annotations

import itertools

import numpy as np

from ..geometry import angle_deg, unit
from .episode import EpisodeTracker


class Interactions:
    def __init__(self, cfg: dict):
        c = cfg["interaction"]
        self.c = c
        # Candidate gaps use the same tolerance as the end gap: a conversation
        # that pauses while staff fetch a box is still being established.
        self.ep = EpisodeTracker(c["min_duration_s"], c["end_gap_s"], c["end_gap_s"])
        self.sessions: list[dict] = []

    def _faces(self, a, b) -> bool:
        f = a.orient.facing if a.orient else None
        d = unit(b.foot - a.foot)
        return f is not None and d is not None and angle_deg(f, d) <= self.c["max_facing_angle_deg"]

    def engaged(self, a, b) -> bool:
        c = self.c
        if not (a.speed_valid and b.speed_valid) or max(a.speed_bh, b.speed_bh) > c["max_speed_bh_s"]:
            return False
        dist = np.linalg.norm(a.foot - b.foot) / ((a.h + b.h) / 2)
        if dist > c["max_distance_bh"]:
            return False
        return self._faces(a, b) or self._faces(b, a)

    def update(self, t, dt, persons, left_gids, role, events: list):
        """Pairs are evaluated between every two people that are not both
        customers, so a pair whose staff label only settles later is still
        tracked; final metrics keep only staff-customer pairs."""
        on = set()
        for a, b in itertools.combinations(persons, 2):
            if role(a.gid) == "customer" and role(b.gid) == "customer":
                continue
            if self.engaged(a, b):
                on.add((min(a.gid, b.gid), max(a.gid, b.gid)))
        confirmed, _ = self.ep.step(t, dt, on)
        self.ep.close_where(lambda k: k[0] in left_gids or k[1] in left_gids)
        for (g1, g2), ep in confirmed:
            e = dict(t=t, type="interaction", pair=[g1, g2], start=ep.start)
            self.sessions.append(e)
            events.append(e)

    def active_pairs(self):
        return list(self.ep.active_confirmed())

    def per_staff(self, staff: dict[int, int], is_customer) -> dict[int, int]:
        """Sessions per staff instance. `staff` maps staff gid -> instance id;
        only staff-customer pairs count (staff-staff chats are not customer
        interactions)."""
        out = {i: 0 for i in set(staff.values())}
        for s in self.sessions:
            g1, g2 = s["pair"]
            if g1 in staff and g2 not in staff and is_customer(g2):
                out[staff[g1]] += 1
            elif g2 in staff and g1 not in staff and is_customer(g1):
                out[staff[g2]] += 1
        return out
