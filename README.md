# Local AI Safety Monitoring System

I built a full-stack camera monitoring system. A lightweight open-source model
runs **locally on CPU**. I turn its output into a **restricted-zone safety event
with an automatic rule**, then persist and manage those events from a live web
dashboard.

No paid cloud AI API. No GPU. No training.

**Demo video (3–5 min):** <https://drive.google.com/file/d/1tuV7ao_2C7YuwjzxGjkyrKpLLmKiejYk/view?usp=sharing>

```
Browser (Next.js 16 + TypeScript)
  ├── webcam  or  synthetic sample source
  ├── frame pump ──► 640 px JPEG ──► WS /ws/inference
  ├── canvas overlay: boxes, track ids, foot points, ROI, FPS/latency HUD
  └── event panel ◄── WS /ws/events
        ▼
FastAPI (Python 3.12)
  decode ─► SSD-MobileNetV1 via ONNX Runtime (CPU) ─► person filter
    ─► IoU tracker ─► rule engine (dwell + hysteresis + cooldown)
    ─► persist event + annotated snapshot ─► broadcast
        ▼
SQLite   (events · event_status_history · zones)
```

---

## System requirements

- Python **3.12** (3.13/3.14 have no `onnxruntime` wheels)
- Node 18+
- SQLite (no database server)

---

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r backend/requirements.txt

python backend/scripts/fetch_model.py
python backend/scripts/make_fixtures.py

cp backend/.env.example backend/.env
cd backend && alembic upgrade head && cd ..

cd backend && uvicorn app.main:app --port 8000
# new terminal
cd frontend && npm install && npm run dev
```

Open <http://localhost:3000>. API docs: <http://localhost:8000/docs>.

`fetch_model.py` downloads the ~29 MB ONNX weights (SHA-256 verified, not
committed). `make_fixtures.py` builds the evaluation clips and the dashboard
Sample image locally — I do not commit either.

---

## Model

| | |
|---|---|
| **Model** | SSD-MobileNetV1, COCO (I use class 1, `person`) |
| **Source** | [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1) |
| **Licence** | Apache-2.0 |
| **Format** | ONNX opset 10, 29 MB |
| **Input** | `image_tensor:0` — uint8 NHWC RGB, any resolution |
| **Output** | normalised boxes (`ymin,xmin,ymax,xmax`), classes, scores |
| **Runtime** | ONNX Runtime 1.20, `CPUExecutionProvider` |

---

## Safety event

The primary event is `restricted_zone_entry`, generated automatically from model
output — not from a test button.

I anchor each person to the **foot point** (bottom-centre of the box) and test
that point against a configurable ROI polygon every frame.

| | Default |
|---|---|
| Confidence threshold | `0.45` |
| Timing window (continuous presence) | `2.0` s dwell |
| Duplicate-alert cooldown | `15` s per track, `10` s per zone |

A walk-past shorter than the dwell window does not fire. The same tracked person
cannot re-alert until cooldown lapses.

---

## Evaluation

```bash
python backend/scripts/make_fixtures.py   # once
python backend/scripts/evaluate.py
cd backend && pytest
```

Two clips go through the real detector, tracker and rule engine:

| Scenario | Expect | Result |
|---|---|---|
| Positive: person enters the zone and stays ~4 s | ≥ 1 event | fires 1 |
| Negative: person moves for 7 s but never enters the zone | 0 events | fires 0 |

The negative clip is longer than the 2 s dwell window, so a pass means the zone
test is doing the work, not that the clip was too short. I synthesise both from
a public detection sample image. I did not record real people and I do not use
patient data.

### Observed performance (Windows 11 laptop, CPU only)

| Path | Measured |
|---|---|
| Model inference — offline replay (`evaluate.py`) | p50 **30–81 ms**, p95 34–137 ms |
| Model inference — live, both dev servers running | p50 **95–150 ms** |
| Browser round-trip (capture → overlay) | **120–165 ms** |
| End-to-end throughput | **4–7 fps** |

These are ranges, not single figures. The spread is dominated by what else the
CPU is doing: the offline replay hits its low end on an idle machine and its
high end with the dev servers up. `evaluate.py` reprints the inference numbers
on every run, so quote that run rather than this table.

---

## Limitations

**False positives.** A poster, photo or reflection of a person in the ROI will
fire. A zone drawn across a walkway re-alerts on ordinary traffic.

**False negatives.** The rule tests the foot point, so occluded or out-of-frame
feet can miss someone standing in the zone. Small or distant people, motion blur
at 4–7 fps, tracker id switches, and low light all reduce recall.

### How I would reduce false alerts with more time

1. N-of-M frame voting instead of wall-clock dwell.
2. Per-track confidence EMA, so one lucky frame cannot carry an alert.
3. Kalman tracking to cut id switches during crossings.
4. ROI erosion buffer so standing on the line does not flicker.
5. Calibrate thresholds on a labelled clip set.
6. A second anchor (box centre or pose ankles) when the lower body is occluded.
