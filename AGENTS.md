# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**LightGlueStick** — a fast joint point-line feature matcher (ICCV Workshops 2025). The repo
holds the library plus **two independent applications** built on top of it:

| Directory | Application |
|-----------|-------------|
| `lightgluestick/` | The matcher library (shared by both apps) |
| `video_stitching/` | Horizontal panorama stitching of N RTSP cameras + RF-DETR object detection / ROI person counting |
| `video_virtual/` | Bird's-eye-view (BEV) fusion: per-camera ground-plane calibration in physical units, then mosaic |

The two applications are separate — they share only `lightgluestick/` and are launched by
different entry points. Each owns its own config file.

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
```

### Run the panorama stitching app
```bash
python video_stitching/app.py
```
Edit `video_stitching/config.yaml` first to set RTSP URLs and the ROI polygon.

### Run the BEV fusion app
```bash
python video_virtual/main.py
```
Calibration is done interactively in the GUI and saved to `video_virtual/config.json`.

### Tests
```bash
python video_stitching/test_layout_units.py
python video_stitching/test_detector_units.py
python video_virtual/test_fusion_units.py
```
Cover the pure-computation parts of each app: spanning-tree layout, homography sanity checks and
exposure-gain solving for the stitching app; rigid estimation, calibration residuals and the
fusion cache for the BEV app. GUI code and the live RTSP paths have no automated tests.
Training lives in the separate GlueFactory repo.

## Path convention

Both apps are run as scripts (`python <dir>/<entry>.py`), which puts their own directory on
`sys.path` — that is why sibling modules import flat (`from pipeline import ...`). Each entry
module inserts the repo root into `sys.path` so `lightgluestick` resolves, and every file/asset
path is derived from `Path(__file__)`, so the apps run correctly from any working directory.
Do not reintroduce CWD-relative paths.

## Architecture

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

### Data flow — panorama stitching app (`video_stitching/`)

```
VideoThread (RTSP) → frames → StitchingThread
    → StitchingPipeline.calculate_homographies() [once: all pairs → gate → spanning tree]
    → StitchingPipeline.stitch_frames(frames)    [per frame, warpPerspective + gain + feather]
    → StitchingPipeline.detect_and_count()       [RF-DETR, ROI polygon test on foot point]
    → PyQt5 signal → UI display
```

### Data flow — BEV fusion app (`video_virtual/`)

```
VideoThread (RTSP) → FrameStore (lock-protected latest frame per camera)
                          ↓ 8 Hz snapshot
                     FusionWorker (background QThread)
                          → cv2.remap per camera (single pre-composed map)
                          → weighted accumulate (pre-normalised feather weights)
                          → mosaic_ready signal → GUI displays only

RegistrationThread (on demand)
    → pairwise LightGlueStick registration between local BEVs
    → quality gate → maximum spanning tree → global layout
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
| `video_stitching/app.py` | PyQt5 app, `VideoThread`, `StitchingThread`; entry point |
| `video_stitching/pipeline.py` | `StitchingPipeline`: N-camera homography estimation + stitching/blending + detection |
| `video_stitching/detector.py` | `RFDETRDetector` + `annotate()` — RF-DETR inference and drawing, decoupled from stitching |
| `video_stitching/widgets.py` | `StitchedView` — zoom/pan + interactive ROI polygon drawing |
| `video_stitching/main_window.ui` | Qt Designer layout (references `widgets` for the custom widget) |
| `video_stitching/config.yaml` | RTSP URLs, ROI polygon, display dimensions |
| `video_virtual/main.py` | PyQt5 calibration/fusion GUI; entry point |
| `video_virtual/calibration.py` | `CameraCalibration` — ground-plane homography from clicked points + physical coords, reprojection residuals |
| `video_virtual/fusion.py` | Pairwise BEV registration, rigid estimation, `solve_layout` spanning tree |
| `video_virtual/fusion_cache.py` | Pre-computed remap tables + normalised feather weights; runtime `fuse()` |
| `video_virtual/fusion_worker.py` | `FrameStore` + `FusionWorker` background thread |

### Model loading pattern

All components are loaded dynamically via `get_model(name, conf)` (in `utils.py`). The string `name` maps to a module path; `conf` is an `OmegaConf`/dict merged with defaults from each class's `default_conf`.

### Device handling

The pipeline auto-selects `cuda` → `mps` → `cpu`. Models are moved to device in `BaseModel.__init__`. Keep tensors on the same device; avoid mixing `.to()` calls.

### Multi-camera stitching (N cameras)

`cameras` in `video_stitching/config.yaml` may list any number of streams. **The list order does
not have to be the spatial order** — `calculate_homographies()` matches every camera pair and
infers adjacency from the RANSAC inlier count, so a mis-ordered config still stitches correctly.

Each pair must clear a quality gate (`MIN_INLIERS`, `MIN_INLIER_RATIO`, plus
`_is_sane_homography` which rejects degenerate, mirrored or diverging warps). Surviving edges are
fed to `solve_layout()`, which builds a maximum spanning tree weighted by inlier count and roots
it at the **tree centre** — the topology-independent generalisation of the old `ref = (n-1)//2`,
so distortion still accumulates only half-way out to each side. Without the gate a non-adjacent
pair (measured: ~20 inliers versus 100+ for a real overlap) would be accepted and drag one camera
into a badly wrong pose.

`self.homographies[i]` maps camera `i` into the root camera's frame; `None` means that camera
could not be linked to the largest connected component and it is dropped from the mosaic.
`self.layout_info` records the chosen root, the accepted edges, the rejected pairs and any
cameras pruned for divergence.

**Divergence is a chaining effect, not a per-pair one.** `_is_sane_homography` only inspects one
hop, so a chain of individually-plausible homographies can still blow a camera up several times
over — measured at FOV 0: cam2 composed to 3.1× area, which pushed the canvas past both caps and
clipped a *different* camera out of the picture entirely. `_prune_diverged()` therefore re-checks
the **composed** transforms and drops the worst-scaled non-root camera until things settle. Its
two triggers are deliberately distinct: a single camera exceeding `MAX_COMPOSED_AREA_RATIO`, or a
total span beyond `SPAN_SLACK ×` the canvas caps. Mild cap overflow is *not* pruned — clipping a
few edge pixels is the accepted old behaviour and far cheaper than losing a whole camera.

### FOV / cylindrical warp

The cylindrical warp is **required**, not cosmetic. It pre-projects each view so adjacent
homographies stay close to translations; without it the chain diverges as above.

`_cyl_maps` samples the output uniformly in **arc length** (θ uniform, output width `f·fov_rad`).
An earlier version reused the input width, which is equivalent to sampling uniformly in `tan θ`;
that makes the edge stretch `1/cos(tan(fov/2))`, which **diverges at fov = 2·atan(π/2) =
115.04°** — 2.7× at 100°, 7.0× at 110°. Feature matching collapsed well before that, which looked
like a scene property but was purely a parameterisation error. Correct sampling caps the stretch
at `1/cos(fov/2)`, which only diverges at 180°. If you touch this function, keep the θ-uniform
sampling; the mask must then be looked up by **output** shape (`_cyl_mask_by_out`), since output
width no longer equals input width.

Measured range after the fix: 40–110 places all three cameras (110 previously failed outright);
115+ starts dropping cameras as the arc-length output gets too narrow to match reliably. The
slider is clamped to 40–110, defaulting to 65. Note that releasing the slider triggers a full
recalibration, which is an all-pairs match rather than N−1 adjacent ones.

**How generic are these numbers?** The gain sigmas are Brown & Lowe's published values and the
convexity/orientation check is pure geometry, so those transfer. `MIN_INLIERS = 40` (≈2 % of the
2048-keypoint budget), `MAX_COMPOSED_AREA_RATIO`, `SPAN_SLACK` and the `MAX_WIDTH`/`MAX_HEIGHT`
caps were all fitted to one three-camera 1080p indoor capture and should be revisited for
low-texture scenes, other resolutions, more cameras, or non-strip layouts. `solve_layout` itself
is topology-agnostic; `_prune_diverged` is not — its trigger is a global bounding-box span while
its victim is picked by per-camera area, which only coincide for a horizontal strip.

### Exposure compensation

Cameras rarely agree on exposure, and the resulting brightness step dominates the visible seam
(measured on the three live cameras: a seam with mean-absolute-difference 56 was 51 of pure
brightness bias). `_solve_exposure_gains()` measures each overlap's mean intensity and solves the
Brown & Lowe gain system (`solve_gains`) for one gain per camera, applied during accumulation.
The prior term pulls gains toward 1, so seams are strongly reduced rather than perfectly
flattened — end-to-end this took the worst seam from 55.4 to 34.4. Gains are computed once with
the canvas and invalidated together with it; set `pipeline.exposure_compensation = False` to
disable.

Homographies are computed **once** and reused — not recomputed per frame. The
canvas geometry and per-camera feather-blend weights are cached alongside them
(`self._canvas`), so a frame costs only N `warpPerspective` calls plus a weighted
accumulate. Setting `pipeline.homographies = None` (what the 重新校准 button does)
invalidates both.

### BEV fusion (video_virtual)

Each camera is calibrated independently: the user clicks ground points and enters their
physical coordinates, giving an image→ground homography at a shared `px/m` scale. Because all
local BEVs use the same scale, two BEVs can only differ by a **rigid** transform — scale is
therefore locked to 1 during registration, and a similarity fit whose scale deviates from 1 by
more than 8 % is rejected as a sign that the two calibrations disagree on physical scale.

Registration tries every camera pair, keeps only edges passing the quality gate, and builds a
maximum spanning tree (weighted by inlier count) so a single failed pair cannot break the whole
layout. Every run clears all previous transforms first — reusing a stale transform would anchor
later cameras to an inconsistent frame.

`FusionCache` pre-computes, per camera, one composed `image → canvas` remap table and a
cross-camera-normalised feather weight. Runtime cost per frame is one `cv2.remap` plus a
multiply-accumulate; nothing content-independent is recomputed. The cache rebuilds automatically
when its signature (calibration, transforms, frame shape) changes.

### Object detection (RF-DETR)

`video_stitching/detector.py` wraps RF-DETR and is deliberately independent of the stitching
code, so the detector can be swapped or tested on its own. The model is loaded lazily on the
first frame so the GUI starts immediately, and the checkpoint is cached by `rfdetr` itself under
`~/.roboflow/models` — nothing is stored in the repo.

**Class-index trap:** RF-DETR returns **1-based COCO** `class_id` (`person == 1`), whereas
`model.class_names` is a 0-based list — the two are off by one, so `class_names[class_id]`
silently yields the wrong label. Always resolve names through
`detections.data['class_name']`, falling back to the (also 1-based) `COCO_CLASSES` dict; that is
what `class_names_of()` does.

Detection runs on every stitched frame regardless of whether an ROI exists, so objects are always
annotated. All detected classes are drawn; only `count_class` is counted, and membership uses the
box's **bottom-centre (foot) point** rather than its centre, because the foot point is what
actually lies on the floor plane. Measured on the three live cameras: ~44 ms per frame for
detection plus annotation on a 3149×2000 mosaic (RF-DETR nano, MPS).

## Configuration

`video_stitching/config.yaml` controls:
- `cameras[].url` — RTSP stream URLs (one entry per camera; order need not be spatial)
- `region_of_interest` — list of `[x, y]` vertices defining the counting polygon
- `display.width` / `display.height` — output canvas size
- `detection.model` — RF-DETR size: `nano` / `small` / `medium` / `base` / `large`
- `detection.threshold` — confidence threshold
- `detection.classes` — allow-list of class names to draw; `null` draws all 80 COCO classes
- `detection.count_class` — which class the ROI counter reports (default `person`)
- `detection.show_labels` / `detection.optimize` — label rendering, and `torch.compile` warm-up

`video_virtual/config.json` (written by the GUI) holds per-camera calibration points, physical
coordinates, homographies, the shared `scale` (px/m) and `canvas_padding_px`.

Matcher depth/confidence is set programmatically in each app's pipeline inside the `conf` dict
passed to `TwoViewPipeline`.

## Dependencies of note

- `pytlsd` — LSD line detector (C++ bindings)
- `kornia` — geometric transforms used in SuperPoint
- `omegaconf` — config merging for model defaults
- `rfdetr` + `supervision` — RF-DETR detector used by `video_stitching/detector.py`
- `PyQt5` — GUI framework for both apps
- Pre-trained weights: RF-DETR downloads and caches its own checkpoint to `~/.roboflow/models` on first use (nano ≈ 349 MB); SuperPoint/LightGlueStick weights are embedded in the package.

## License note

SuperPoint uses a **non-commercial license**. Deployments must respect this constraint.
