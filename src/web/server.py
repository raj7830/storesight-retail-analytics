"""Web app: upload a CCTV clip (or give a camera URL) and watch the analysis live.

    uvicorn src.web.server:app --host 0.0.0.0 --port 8000

Endpoints
  POST /api/jobs                  multipart: file=<video> | source=<rtsp/http url>
  GET  /api/jobs/{id}             status + latest metrics
  GET  /api/jobs/{id}/stream      MJPEG of annotated frames, as they are produced
  GET  /api/jobs/{id}/events      Server-Sent Events: metrics + new behavioural events
  POST /api/jobs/{id}/stop        stop a running job (results so far are kept)
  GET  /api/jobs/{id}/files/{f}   video_metrics.csv, shelf_metrics.csv, annotated.mp4, events.jsonl

Jobs run one at a time (a single GPU/CPU is shared); extra jobs wait in a queue.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from ..pipeline import Models, load_config
from ..runner import FileSource, LiveSource, run_job, write_results

CONFIG = os.environ.get("BA_CONFIG", "configs/default.yaml")
WORK = Path(os.environ.get("BA_WORKDIR", "outputs/web"))
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Retail Behavioural Analytics")
cfg = load_config(CONFIG)
models: Models | None = None
gpu = threading.Lock()
jobs: dict[str, "Job"] = {}


class Job:
    def __init__(self, jid: str, name: str, source: str, live: bool, is_upload: bool):
        self.id, self.name, self.source, self.live, self.is_upload = jid, name, source, live, is_upload
        self.status = "queued"
        self.error = None
        self.metrics: dict = {}
        self.events: list[dict] = []
        self.jpeg: bytes | None = None
        self.frame_no = 0
        self.progress = 0.0
        self.stop = threading.Event()
        self.dir = WORK / jid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.duration = 0.0          # seconds of video, 0 for open-ended streams

    def summary(self):
        return dict(id=self.id, name=self.name, status=self.status, error=self.error, live=self.live,
                    started=self.started, duration=round(self.duration, 1),
                    progress=round(self.progress, 3), metrics=self.metrics, n_events=len(self.events))


def _video_id(name: str) -> str:
    stem = Path(name).stem or "stream"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:40]


def _run(job: Job):
    global models
    with gpu:
        if job.stop.is_set():
            job.status = "stopped"
            return
        job.status = "running"
        try:
            if models is None:
                models = Models(cfg)
            fps = cfg["runtime"]["analysis_fps"]
            src = LiveSource(job.source, fps, pace_file=True) if job.live and job.is_upload \
                else LiveSource(job.source, fps) if job.live else FileSource(job.source, fps)
            total_s = (src.total / src.fps) if src.total else 0
            job.duration = total_s

            def on_frame(img):
                small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    job.jpeg = buf.tobytes()
                    job.frame_no += 1

            def on_update(m, new):
                job.metrics = m
                job.events.extend(new)
                if total_s:
                    job.progress = min(1.0, m["t"] / total_s)

            pl = run_job(cfg, models, src, _video_id(job.name), str(job.dir),
                         on_frame=on_frame, on_update=on_update, stop_event=job.stop)
            write_results(pl, str(job.dir))
            job.metrics = pl.live_metrics()
            job.status = "stopped" if job.stop.is_set() else "done"
            job.progress = 1.0 if job.status == "done" else job.progress
        except Exception as e:  # surfaced to the UI
            traceback.print_exc()
            job.status, job.error = "error", f"{type(e).__name__}: {e}"


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.post("/api/jobs")
async def create_job(file: UploadFile | None = File(None), source: str = Form(""),
                     live: bool = Form(False)):
    jid = uuid.uuid4().hex[:10]
    if file is not None and file.filename:
        job = Job(jid, file.filename, "", live, True)
        dst = job.dir / ("input" + (Path(file.filename).suffix or ".mp4"))
        with open(dst, "wb") as f:
            await asyncio.to_thread(shutil.copyfileobj, file.file, f, 4 << 20)
        job.source = str(dst)
    elif source.strip():
        # Camera URLs are live by nature; a server-side path can be either.
        s = source.strip()
        is_path = Path(s).exists()
        job = Job(jid, Path(s).name if is_path else s, s, live or not is_path, is_path)
    else:
        raise HTTPException(400, "upload a video file or give a source URL")
    jobs[jid] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job.summary()


def _job(jid: str) -> Job:
    if jid not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[jid]


@app.get("/api/jobs")
def list_jobs():
    return sorted((j.summary() for j in jobs.values()), key=lambda j: -j["started"])


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _job(jid).summary()


@app.post("/api/jobs/{jid}/stop")
def stop_job(jid: str):
    _job(jid).stop.set()
    return {"ok": True}


@app.get("/api/jobs/{jid}/stream")
async def stream(jid: str):
    job = _job(jid)
    boundary = "frame"

    async def gen():
        last = -1
        while True:
            if job.jpeg is not None and job.frame_no != last:
                last = job.frame_no
                yield (b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(job.jpeg)).encode() + b"\r\n\r\n" + job.jpeg + b"\r\n")
            elif job.status in ("done", "error", "stopped"):
                break
            await asyncio.sleep(0.03)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@app.get("/api/jobs/{jid}/events")
async def events(jid: str):
    job = _job(jid)

    async def gen():
        sent = 0
        while True:
            new = job.events[sent:]
            sent += len(new)
            payload = dict(job.summary(), events=new)
            yield f"data: {json.dumps(payload, default=float)}\n\n"
            if job.status in ("done", "error", "stopped"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{jid}/files/{name}")
def files(jid: str, name: str):
    job = _job(jid)
    vid = _video_id(job.name)
    paths = {
        "video_metrics.csv": job.dir / "video_metrics.csv",
        "shelf_metrics.csv": job.dir / "shelf_metrics.csv",
        "annotated.mp4": job.dir / "annotated" / f"{vid}_annotated.mp4",
        "events.jsonl": job.dir / "events" / f"{vid}_events.jsonl",
    }
    p = paths.get(name)
    if p is None or not p.exists():
        raise HTTPException(404, "not available yet")
    return FileResponse(p, filename=f"{vid}_{name}")
