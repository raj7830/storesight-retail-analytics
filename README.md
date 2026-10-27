# StoreSight: Retail Behavioural Analytics

<p align="center">
  <img src="docs/demo.gif" alt="StoreSight dashboard analysing CCTV footage live: annotated feed, conversion funnel, event timeline and staff metrics" width="100%">
</p>
<p align="center"><b><a href="docs/demo.mp4">▶ Watch the full-quality demo (MP4, 66 s)</a></b> · Live analysis shown at 2× speed · faces pixelated with the built-in privacy mode</p>

Real-time, single-pass video analytics for retail CCTV. From one fixed camera overlooking the storefront and the interior, it measures:

1. **Store interest and conversion**: passers-by who show interest in the store, and how many of them enter or walk on.
2. **Per-shelf interest**: distinct interest episodes for Shelf A and Shelf B.
3. **Staff-customer interaction**: average number of interaction sessions per staff member.

Every frame is processed once, in order, and never revisited, so the same code serves an uploaded video and a live RTSP camera. Results appear while the video plays. All inference runs locally on public pretrained weights.

> **Footage and privacy.** No raw CCTV footage, people images or analysis outputs are included in this repository; the demo above has every tracked person's head pixelated. Put your own videos in `Input_Video/` (git-ignored). Set `privacy.blur_faces: true` (or pass `--blur-faces`) to pixelate heads in your own annotated output.

---

## Contents

- [Quick start](#quick-start)
- [Setup](#setup)
- [Running](#running)
- [Docker](#docker)
- [Outputs](#outputs)
- [Technical approach](#technical-approach)
- [Behavioural definitions and thresholds](#behavioural-definitions-and-thresholds)
- [Project structure](#project-structure)
- [Re-enrolling a new store / camera](#re-enrolling-a-new-store--camera)
- [Limitations](#limitations)

---

## Quick start

```bash
cd Behavioral_Analytics
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.tools.download_models

# CSVs + annotated videos for both snippets → outputs/
python -W ignore -m src.run Input_Video/sample1.mp4 Input_Video/sample3.mp4

# or the web app: upload a video and watch the analysis live
uvicorn src.web.server:app --port 8000      # open http://localhost:8000
```

---

## Setup

**Requirements:** Python 3.11 and `ffmpeg` on the PATH (used to write browser-playable H.264 video; without it the tool falls back to OpenCV's mp4v writer).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # exact versions used to produce outputs/
python -m src.tools.download_models      # one-off: fetches all model weights
```

### Models

| Model | Role | Source | Location |
|---|---|---|---|
| YOLO11s-pose | People + 17 body keypoints, full frame | Ultralytics release asset (COCO) | `models/yolo11s-pose.pt` |
| YOLO11s | Small passers-by on the mall strip | Ultralytics release asset (COCO) | `models/yolo11s.pt` |
| CLIP ViT-B/32 (`laion2b_s34b_b79k`) | Image embedding for staff uniform | OpenCLIP / Hugging Face | HF cache (downloaded by the script above) |
| Staff classifier head | Logistic regression on CLIP embeddings | Trained here from `data/staff_enrollment/` | `models/staff_classifier.pkl` |

The trained classifier head is included in `models/`; the YOLO and CLIP weights are fetched by `download_models`. The labelled person crops used to train the head are not published (they show identifiable people). To retrain for your own store, see [Re-enrolling a new store / camera](#re-enrolling-a-new-store--camera).

**Device:** with `runtime.device: auto` the tool uses CUDA if available, then Apple MPS, then CPU.

---

## Running

All commands are run from the project root with the venv active.

### 1. Process the provided snippets (CSVs + annotated videos)

```bash
python -W ignore -m src.run Input_Video/sample1.mp4 Input_Video/sample3.mp4
```

This writes `outputs/video_metrics.csv`, `outputs/shelf_metrics.csv`, `outputs/annotated/<id>_annotated.mp4` and `outputs/events/`. Progress and every detected event are printed while the video is processed. Expect roughly one minute of processing per minute of video on an Apple M1.

Useful options:

```bash
python -W ignore -m src.run Input_Video/sample1.mp4 --video-id video_01   # custom id in the CSV
python -W ignore -m src.run Input_Video/sample1.mp4 --no-video            # metrics only
python -W ignore -m src.run Input_Video/sample1.mp4 --blur-faces          # pixelate heads in the annotated video
python -W ignore -m src.run Input_Video/sample1.mp4 --config configs/default.yaml --out outputs
```

Videos are processed independently; identities do not carry over between snippets. Re-running a video replaces its rows in the CSVs. File mode is deterministic, so repeated runs give identical numbers.

### 2. Web app (upload and watch live)

```bash
uvicorn src.web.server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>, drop in a video (or enter a camera URL) and press **Analyse**. The page shows:

- the annotated feed as it is produced;
- live KPI cards and an event feed;
- download links for the CSVs, annotated MP4 and event log when the run finishes.

Each job's files are stored in `outputs/web/<job_id>/`.

### 3. Live CCTV camera

```bash
python -W ignore -m src.run rtsp://USER:PASS@CAMERA_IP/stream --live --video-id cam_01
python -W ignore -m src.run 0 --live --video-id webcam
```

In live mode a reader thread always holds the newest frame. If the machine is slower than the camera, fewer frames are analysed, but processing never falls behind real time. Press Ctrl-C to stop.

---

## Docker

The image is CPU-only for portability (Apple MPS is not available inside containers). All weights are baked in at build time, so it runs offline.

```bash
docker build -t retail-analytics .

# web app
docker run --rm -p 8000:8000 retail-analytics

# batch processing of the provided snippets
docker run --rm \
  -v "$PWD/Input_Video:/app/Input_Video" \
  -v "$PWD/outputs:/app/outputs" \
  retail-analytics \
  python -W ignore -m src.run Input_Video/sample1.mp4 Input_Video/sample3.mp4
```

For an NVIDIA GPU, install the CUDA build of PyTorch in the Dockerfile and run with `--gpus all`.

---

## Outputs

| File | Contents |
|---|---|
| `outputs/video_metrics.csv` | `video_id, store_interest_count, entered_count, passed_count, staff_instances, avg_customer_interactions_per_staff` |
| `outputs/shelf_metrics.csv` | `video_id, shelf_id, interest_events` |
| `outputs/annotated/<id>_annotated.mp4` | Annotated video (H.264, 10 fps) |
| `outputs/events/<id>_events.jsonl` | Every confirmed event with its timestamp: `store_interest`, `store_entered`, `shelf_interest`, `interaction` |
| `outputs/events/<id>_summary.json` | Final metrics plus every person (role, staff score, lifetime, tracker IDs, store status), for auditing |

**What the annotated video shows:**

- **Boxes and roles:** green box + `ID n` is a customer, orange `STAFF n` is a staff member, grey means the role is not decided yet.
- **Facing:** a white arrow from the head shows the estimated facing direction.
- **Store interest:** `looking...` (gathering evidence), then `STORE INTEREST` or `ENTERED`.
- **Shelf interest:** `INTEREST Shelf X`, with the shelf zone filled magenta.
- **Interaction:** a thick blue line labelled `INTERACTION` between the staff member and the customer.
- **Scene and counts:** the entrance line and shelf boxes are drawn, and a bottom-left panel shows the live counts.

---

## Technical approach

```
frame → YOLO11s-pose (full frame) + YOLO11s (mall strip, 1.5× up-scaled)
      → ByteTrack → identity layer (stitch / split / revive)
      → kinematics (median-based velocity) + facing direction from keypoints
      → staff classifier (CLIP ViT-B/32 + logistic regression, per-person voting)
      → Task 1 / Task 2 / Task 3 state machines → events
      → annotated frame + live metrics → MP4 / JSONL / CSV / web stream
```

- **Pose instead of plain detection.** The rules need to know where people face. Facing comes from the ear-to-nose vector, or from the shoulder-line perpendicular when the face is hidden. The pose model labels left/right shoulders, so the sign separates front from back.
- **Second detector for the mall.** Passers-by behind the glass are about 20×50 px. The pose model missed them entirely in testing; YOLO11s on an up-scaled crop of that strip finds them, adding about 30 ms per frame. Boxes are merged, and the pose detection wins where they overlap.
- **Identity layer above ByteTrack.**
  - *Stitching:* re-attaches a person who re-appears within 2.5 s near where they were lost and with similar clothing colours, so occlusions don't double-count.
  - *Splitting:* when the tracker moves an ID onto someone else after people cross, the track's colours stop matching for 0.6 s and it becomes a new person.
- **Staff by uniform.** The uniform is a black apron: bib with a round white logo at the front, crossed straps at the back, worn over the staff member's own top. A colour rule failed because customers wear black tops and rust patterned dresses. Instead, CLIP embeddings feed a logistic-regression head trained on 1,244 hand-labelled crops from this store. It scores **95.7% per crop** under cross-validation with whole minutes of footage held out. Each person is classified twice per second, and their role is the mean over samples (locked once decisive).
- **Perspective-invariant thresholds.** Distances and speeds are measured in body heights (bh), so the same values work near and far from the camera (1 bh is about 1.65 m).
- **Debounced episodes.** Shelf interest and interactions use a candidate → confirmed → closed state machine (`src/analytics/episode.py`). A continuous episode is counted once, and a later return is a new event.

### Performance

On an Apple M1 (MPS) the pipeline runs at about 8–9 analysed frames per second, with analysis at 10 fps (about 0.85–0.9× real time). To reach real time on the M1, set `runtime.analysis_fps: 8` or use `--live`; a CUDA GPU runs faster than real time.

---

## Behavioural definitions and thresholds

All values live in [`configs/default.yaml`](configs/default.yaml), each with its reasoning in a comment. The **foot point** (bottom-centre of the box) is a person's ground position.

### Task 1: store interest and conversion

| Concept | Definition |
|---|---|
| **Passer-by** | A person first seen in the mall walkway zone (outside the entrance line). People already inside at the start, and staff, are excluded. |
| **Attention evidence (per frame, while outside)** | (a) **Looking:** frontal face visible, i.e. nose and both eyes, shoulders not mirrored. The camera is inside the store looking out, so facing the camera = facing the store. Or (b) **window shopping:** speed < **0.25 bh/s** within **110 px** of the storefront and not facing away. |
| **Interested** | At least **0.8 s** of evidence within a **3 s** sliding window: more than a glance (0.3–0.5 s), less than a stop. Walking past side-on never qualifies. |
| **Entered** | Foot point crosses the entrance line from outside (within its extent ±10%), gets **≥ 20 px** past it and stays inside **1.0 s**. This hysteresis stops box jitter and walkers along the glass from counting. |
| **Entering implies interest** | Every passer-by who enters counts as interested, since entering is the strongest evidence of attention. |
| **Passed by** | Interested and never entered. Unique people: interested = entered + passed. |

### Task 2: per-shelf interest (customers only)

| Concept | Definition |
|---|---|
| **Engaged (per frame)** | Foot point inside the shelf zone (expanded by **40 px** to include the aisle), speed ≤ **0.35 bh/s** (normal walking is about 0.8), and facing within **80°** of the shelf centre. Unknown orientation, e.g. head hidden by the shelf, is accepted. Where zones overlap, the shelf faced most directly wins. |
| **Interest event** | Counted **once** after **2.0 s** of engagement. While still being confirmed, gaps up to **1.0 s** are tolerated. |
| **Same episode vs. return** | A confirmed episode survives absences up to **3.0 s** (picking up a shoe, a head turn). A longer absence closes it, and a later return is a **new** event. |

### Task 3: staff-customer interaction

| Concept | Definition |
|---|---|
| **Engaged (per frame)** | Feet ≤ **1.0 bh** apart (about 1.6 m, including across the counter), both moving ≤ **0.5 bh/s**, and at least one facing the other within **70°**. |
| **Session** | Confirmed after **2.0 s** of engagement. It continues through pauses up to **4.0 s** (staff turns to the till); a longer separation ends it, and re-engaging later is a new session. Staff-staff conversations are excluded. |
| **Staff instance** | One staff identity while in view. Identities that follow each other within **10 s** and look alike are merged, e.g. when someone briefly drops out of detection behind the counter. Someone who leaves the view and returns later counts as a new instance (no re-identification across exits). |
| **Metric** | `avg_customer_interactions_per_staff = staff-customer sessions / staff instances`, including staff with zero sessions. |

### Identity and noise

| Setting | Value | Why |
|---|---|---|
| `stitch_max_gap_s` / `stitch_min_appearance` | 2.5 s / 0.55 | Re-attach briefly hidden people without merging different people |
| `split_max_appearance` / `split_frames` | 0.30 / 6 frames | Detect tracker ID swaps; far below the stitch bar so lighting changes never trigger it |
| `min_track_life_s` | 1.0 s | Flickering false positives are never reported |
| `ignore_zones` | wall mirror | Reflections are not people |

---

## Project structure

```
Behavioral_Analytics/
├── README.md
├── Dockerfile  .dockerignore  requirements.txt
├── configs/
│   ├── default.yaml            scene geometry + all thresholds (commented)
│   └── bytetrack.yaml          tracker settings
├── models/                     YOLO weights + staff_classifier.pkl
├── data/staff_enrollment/      your labelled crops: staff/ and customer/ (git-ignored)
├── src/
│   ├── run.py                  CLI entry point
│   ├── runner.py               frame sources (file / live), H.264 writer, job loop, CSV output
│   ├── pipeline.py             orchestration, roles, staff instances, metrics
│   ├── tracking.py             detector fusion, ByteTrack, identity layer, kinematics
│   ├── pose.py                 facing direction from keypoints
│   ├── staff.py                CLIP + linear staff classifier
│   ├── render.py               annotated frames + HUD
│   ├── geometry.py
│   ├── analytics/              store.py (T1)  shelf.py (T2)  interaction.py (T3)  episode.py
│   ├── web/                    server.py (FastAPI)  static/index.html
│   └── tools/                  download_models.py  dump_crops.py  train_staff_classifier.py
├── Input_Video/                your CCTV videos (git-ignored)
└── outputs/                    CSVs, annotated/, events/, web/ (git-ignored)
```

---

## Re-enrolling a new store / camera

**New uniform.** Only a small linear head is trained, so this takes minutes:

```bash
python -m src.tools.dump_crops path/to/video.mp4 --every 2 --out data/unlabelled
# sort the crops into data/staff_enrollment/staff/ and data/staff_enrollment/customer/
python -m src.tools.train_staff_classifier      # prints cross-validated accuracy, saves the model
```

**New camera view.** Edit `scene` in `configs/default.yaml`: `entrance_line`, `inside_point`, `outside_zone`, `shelves`, `ignore_zones`, and `model.roi_detector.box` (the mall strip).

---

## Limitations

- **Distant passers-by:** people far down the mall are too small for face keypoints, so their interest relies on stopping at the storefront rather than head turns.
- **Approximate distances:** geometry is in image space (no floor homography), so distances are approximate and the storefront band is in pixels.
- **Look-alike people:** re-identification uses colour histograms, so similar-looking people in a crowd can still be confused.
- **Store-specific staff classifier:** it was trained on this store's footage and must be re-enrolled for a different uniform.
- **No ground truth:** there are no ground-truth annotations for the clips, so the numbers are the system's estimates under the definitions above. The event log and summary JSON trace every count back to a person and a second for manual review.
