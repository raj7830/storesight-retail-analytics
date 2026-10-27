"""Frame sources, annotated-video writer and the job loop shared by CLI and web app."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2

from .pipeline import Models, Pipeline


class FileSource:
    """Deterministic: every k-th frame of a file, timestamps from the frame index.
    Re-running gives identical results (used for the reported metrics)."""

    def __init__(self, path: str, analysis_fps: float):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.stride = max(1, round(self.fps / analysis_fps))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.out_fps = self.fps / self.stride

    def __iter__(self):
        i = 0
        while True:
            if i % self.stride == 0:
                ok, frame = self.cap.read()
                if not ok:
                    break
                yield frame, i / self.fps
            elif not self.cap.grab():
                break
            i += 1
        self.cap.release()


class LiveSource:
    """Live camera semantics (RTSP/HTTP/webcam, or a file played at real speed).
    A reader thread always holds the newest frame; the analyser takes the
    latest one when it is free, so it never falls behind real time - if the
    machine is slow it simply analyses fewer frames per second."""

    def __init__(self, src, analysis_fps: float, pace_file: bool = False):
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {src}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if pace_file else 0
        self.period = 1.0 / analysis_fps
        self.out_fps = analysis_fps
        self.pace = pace_file
        self.latest, self.done, self.lock = None, False, threading.Lock()
        self.t0 = time.time()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        n = 0
        while not self.done:
            ok, frame = self.cap.read()
            if not ok:
                break
            ts = n / self.fps if self.pace else time.time() - self.t0
            with self.lock:
                self.latest = (frame, ts)
            n += 1
            if self.pace:   # emulate a camera: deliver frames at their real rate
                time.sleep(max(0.0, self.t0 + n / self.fps - time.time()))
        self.done = True

    def __iter__(self):
        last_t = -1.0
        next_due = time.time()
        while True:
            with self.lock:
                item = self.latest
            if item is not None and item[1] > last_t:
                last_t = item[1]
                yield item
                next_due += self.period
                time.sleep(max(0.0, next_due - time.time()))
            elif self.done:
                break
            else:
                time.sleep(0.005)
        self.cap.release()

    def stop(self):
        self.done = True


class VideoWriter:
    """H.264 via an ffmpeg pipe (plays in browsers), else OpenCV mp4v."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self.proc = self.cv = None
        w, h = size
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.cv:
            self.cv.release()


def run_job(cfg: dict, models: Models, source, video_id: str, out_dir: str,
            on_frame=None, on_update=None, stop_event: threading.Event | None = None) -> Pipeline:
    """Single pass over `source`. Streams annotated frames to `on_frame`,
    and metrics + new events to `on_update`, as they are produced."""
    out = Path(out_dir)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    (out / "events").mkdir(parents=True, exist_ok=True)
    pl = Pipeline(cfg, video_id, models)
    writer = None
    ev_file = open(out / "events" / f"{video_id}_events.jsonl", "w")
    n_events = 0
    try:
        for frame, t in source:
            if stop_event is not None and stop_event.is_set():
                break
            annotated = pl.process(frame, t)
            if cfg["output"]["write_video"]:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = VideoWriter(str(out / "annotated" / f"{video_id}_annotated.mp4"),
                                         source.out_fps, (w, h))
                writer.write(annotated)
            new = pl.events[n_events:]
            n_events = len(pl.events)
            for e in new:
                ev_file.write(json.dumps(e) + "\n")
            if on_frame:
                on_frame(annotated)
            if on_update:
                on_update(pl.live_metrics(), new)
    finally:
        ev_file.close()
        if writer:
            writer.close()
    return pl


def _upsert_csv(path: Path, rows: list[dict], key):
    """Merge rows into an existing CSV, replacing rows with the same key."""
    if not rows:
        return
    old = []
    if path.exists():
        with open(path) as f:
            old = [r for r in csv.DictReader(f)]
    new_keys = {key(r) for r in rows}
    merged = [r for r in old if key(r) not in new_keys] + rows
    merged.sort(key=key)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(merged)


def write_results(pl: Pipeline, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _upsert_csv(out / "video_metrics.csv", [pl.video_row()], key=lambda r: r["video_id"])
    _upsert_csv(out / "shelf_metrics.csv", pl.shelf_rows(), key=lambda r: (r["video_id"], r["shelf_id"]))
    detail = pl.live_metrics()
    detail["people"] = [
        dict(gid=p.gid, role=p.role(pl.min_samples), staff_score=round(p.staff_score, 3),
             first_t=round(p.first_t, 1), last_t=round(p.last_t, 1), track_ids=sorted(p.track_ids),
             store=pl.store.status(p.gid))
        for p in pl.ids.all_persons() if p.life >= pl.min_life]
    with open(out / "events" / f"{pl.video_id}_summary.json", "w") as f:
        json.dump(detail, f, indent=1)
