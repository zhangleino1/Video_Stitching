# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**LightGlueStick** — a fast joint point-line feature matcher (ICCV Workshops 2025) used both as a standalone library and as the backbone for a real-time multi-camera video stitching + person-counting application.

## Commands

### Install
```bash
python -m venv venv && source venv/bin/activate
pip install .
# or for development:
pip install -r requirements.txt
```

### Run the demo (two-image matching)
```bash
python -m lightgluestick.run -img1 assets/img1.jpg -img2 assets/img2.jpg
# With adaptive early exit:
python -m lightgluestick.run -img1 assets/img1.jpg -img2 assets/img2.jpg --depth_confidence 0.95
```

### Run the multi-camera stitching app
```bash
python app.py
```
Edit `config.yaml` first to set RTSP URLs and ROI polygon.

### No test suite exists in this repo. Training lives in the separate GlueFactory repo.

## Architecture

### Two main concerns

**1. LightGlueStick library** (`lightgluestick/`)  
Feature matching pipeline using a 7-layer transformer. Given two images, it returns matched keypoints and line segments plus the confidence scores.

**2. Stitching application** (`pipeline.py`, `app.py`)  
Uses the matcher to compute a homography (once, then reused across frames), warps live RTSP frames, and runs YOLOv8 to count persons inside a configurable polygon ROI.

### Data flow — library

```
Images → WireframeExtractor
             ├─ SuperPoint  → keypoints + 256-d descriptors
             └─ LSD         → line segments → Wireframe (DBSCAN junction clustering)
         → LightGlueStick (transformer, 7 layers)
             ├─ SelfBlock    (intra-image self-attention)
             ├─ LineLayer    (line-to-line attention)
             └─ CrossBlock   (cross-image attention)
         → matched kpts / lines
         → cv2.findHomography (RANSAC)
```

### Data flow — stitching app

```
VideoThread (RTSP) → frames → StitchingThread
    → StitchingPipeline.calculate_homography() [once]
    → StitchingPipeline.stitch_frames()         [per frame, warpPerspective]
    → StitchingPipeline.detect_and_count()      [YOLOv8, ROI polygon test]
    → PyQt5 signal → UI display
```

### Key files

| File | Role |
|------|------|
| `lightgluestick/lightgluestick.py` | Core transformer matcher (`TransformerLayer`, `SelfBlock`, `LineLayer`, `CrossBlock`, `TokenConfidence`) |
| `lightgluestick/two_view_pipeline.py` | Orchestrates extractor → matcher → filter → solver |
| `lightgluestick/wireframe.py` | Clusters line endpoints (DBSCAN) into junctions, builds adjacency mask |
| `lightgluestick/superpoint.py` | CNN keypoint + descriptor extractor |
| `lightgluestick/lsd.py` | Thin wrapper around `pytlsd` |
| `lightgluestick/base_model.py` | `BaseModel` abstract class; `MetaModel` metaclass wires config inheritance |
| `lightgluestick/utils.py` | `get_model()` factory, image tensor helpers |
| `pipeline.py` | `StitchingPipeline`: homography estimation + stitching + YOLO counting |
| `app.py` | PyQt5 app, `VideoThread`, `StitchingThread` |
| `config.yaml` | RTSP URLs, ROI polygon, display dimensions |

### Model loading pattern

All components are loaded dynamically via `get_model(name, conf)` (in `utils.py`). The string `name` maps to a module path; `conf` is an `OmegaConf`/dict merged with defaults from each class's `default_conf`.

### Device handling

The pipeline auto-selects `cuda` → `mps` → `cpu`. Models are moved to device in `BaseModel.__init__`. Keep tensors on the same device; avoid mixing `.to()` calls.

### Homography reuse

`StitchingPipeline` computes homography only once (or on demand) and stores it in `self.H`. It does **not** recompute every frame — this is intentional for real-time performance.

## Configuration

`config.yaml` controls:
- `cameras[].url` — RTSP stream URLs (one entry per camera)
- `region_of_interest` — list of `[x, y]` vertices defining the counting polygon
- `display.width` / `display.height` — output canvas size

Matcher depth/confidence is set programmatically in `pipeline.py` inside the `conf` dict passed to `TwoViewPipeline`.

## Dependencies of note

- `pytlsd` — LSD line detector (C++ bindings)
- `kornia` — geometric transforms used in SuperPoint
- `omegaconf` — config merging for model defaults
- `ultralytics` — YOLOv8 (`YOLO` class in `pipeline.py`)
- `PyQt5` — GUI framework for `app.py`
- Pre-trained weights: `yolo26n.pt` (YOLOv8n) must be present at repo root; SuperPoint/LightGlueStick weights are embedded in the package.

## License note

SuperPoint uses a **non-commercial license**. Deployments must respect this constraint.
