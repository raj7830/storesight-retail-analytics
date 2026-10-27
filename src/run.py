"""Command-line entry point.

    python -m src.run Input_Video/sample1.mp4 Input_Video/sample3.mp4
    python -m src.run rtsp://user:pass@camera/stream --live --video-id cam_01
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .pipeline import Models, load_config
from .runner import FileSource, LiveSource, run_job, write_results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Retail behavioural analytics (single pass, causal).")
    ap.add_argument("sources", nargs="+", help="video files, RTSP/HTTP URLs or webcam index")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--video-id", action="append", help="id per source (default: file stem)")
    ap.add_argument("--live", action="store_true",
                    help="live-camera mode: analyse the newest frame, drop frames if behind "
                         "(files are played back at real speed)")
    ap.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.no_video:
        cfg["output"]["write_video"] = False
    models = Models(cfg)
    print(f"device: {models.device}", file=sys.stderr)

    for i, src in enumerate(a.sources):
        vid = a.video_id[i] if a.video_id and i < len(a.video_id) else Path(str(src)).stem
        fps_target = cfg["runtime"]["analysis_fps"]
        is_file = Path(str(src)).exists()
        source = LiveSource(src, fps_target, pace_file=is_file) if a.live or not is_file \
            else FileSource(src, fps_target)
        total = source.total
        t0 = time.time()
        last = [0.0]

        def progress(m, new):
            now = time.time()
            for e in new:
                print(f"  [{e['t']:7.1f}s] {e['type']:15s} " +
                      " ".join(f"{k}={v}" for k, v in e.items() if k not in ('t', 'type', 'video_id')),
                      file=sys.stderr)
            if now - last[0] > 5:
                last[0] = now
                pct = f"{100 * m['t'] * source.fps / total:5.1f}%" if total else ""
                print(f"{vid} {pct} t={m['t']}s  {m['fps']} fps  interest={m['store_interest_count']} "
                      f"entered={m['entered_count']} shelves={m['shelves']} staff={m['staff_instances']} "
                      f"sessions={m['interaction_sessions']}", file=sys.stderr)

        pl = run_job(cfg, models, source, vid, a.out, on_update=progress)
        write_results(pl, a.out)
        el = time.time() - t0
        print(f"{vid}: done in {el:.0f}s for {pl.t:.0f}s of video ({pl.t / el:.2f}x real time)", file=sys.stderr)
        print(pl.video_row())
        for r in pl.shelf_rows():
            print(r)


if __name__ == "__main__":
    main()
