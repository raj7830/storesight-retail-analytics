"""Debounced on/off episode used by shelf interest and staff interactions.

A frame-level condition (e.g. "customer is at shelf A and looking at it") is
noisy: occlusions, a turned head or one missed detection would split one
visit into many. An Episode therefore:

1. starts as a *candidate* the first time the condition is true,
2. is *confirmed* (= counted once) when the condition has been true for
   `min_on` seconds, tolerating gaps up to `max_gap` while still a candidate,
3. once confirmed, stays open through gaps up to `end_gap` and is *closed*
   after a longer absence. A later return opens a NEW episode.
"""
from __future__ import annotations


class Episode:
    __slots__ = ("start", "last_on", "on_time", "confirmed", "confirm_t")

    def __init__(self, t: float):
        self.start = t
        self.last_on = t
        self.on_time = 0.0
        self.confirmed = False
        self.confirm_t = None


class EpisodeTracker:
    def __init__(self, min_on: float, max_gap: float, end_gap: float):
        self.min_on, self.max_gap, self.end_gap = min_on, max_gap, end_gap
        self.open: dict = {}

    def step(self, t: float, dt: float, on_keys: set):
        """Advance one analysed frame. Returns (newly confirmed, closed) keys."""
        confirmed, closed = [], []
        for k in on_keys:
            ep = self.open.get(k)
            if ep is None:
                ep = self.open[k] = Episode(t)
            else:
                ep.on_time += min(dt, t - ep.last_on)
            ep.last_on = t
            if not ep.confirmed and ep.on_time >= self.min_on:
                ep.confirmed, ep.confirm_t = True, t
                confirmed.append((k, ep))
        for k, ep in list(self.open.items()):
            if k in on_keys:
                continue
            idle = t - ep.last_on
            if (not ep.confirmed and idle > self.max_gap) or idle > self.end_gap:
                del self.open[k]
                if ep.confirmed:
                    closed.append((k, ep))
        return confirmed, closed

    def close_where(self, pred):
        """Force-close episodes (e.g. a participant left the scene)."""
        closed = []
        for k, ep in list(self.open.items()):
            if pred(k):
                del self.open[k]
                if ep.confirmed:
                    closed.append((k, ep))
        return closed

    def active_confirmed(self):
        return {k: ep for k, ep in self.open.items() if ep.confirmed}
