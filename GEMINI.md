# LightGlueStick Video Stitching & Monitoring

This project is a real-time video stitching application that combines dual RTSP streams into a single panorama. It integrates **LightGlueStick** for robust feature matching and **YOLOv8** for person detection and counting within a user-defined Region of Interest (ROI).

## Project Overview

-   **Purpose:** Seamlessly stitch two video streams and monitor a specific area for person counting.
-   **Core Technologies:**
    -   **LightGlueStick:** A fast and robust joint point-line matching model (ICCV 2025).
    -   **YOLOv8:** Person detection backbone (using `yolo26n.pt`).
    -   **PyQt5:** GUI framework for the desktop application.
    -   **OpenCV:** Image processing, perspective warping, and video stream handling.
    -   **PyTorch:** Deep learning execution backend.

## Architecture

1.  **GUI Layer (`app.py`, `widgets.py`, `main_window.ui`):**
    -   `app.py`: Manages the main event loop and multi-threaded processing.
    -   `VideoThread`: Handles concurrent RTSP stream capture using OpenCV.
    -   `StitchingThread`: Processes frames through the pipeline to keep the UI responsive.
    -   `StitchedView`: A custom PyQt widget for interactive ROI drawing.
2.  **Processing Pipeline (`pipeline.py`):**
    -   `StitchingPipeline`: Orchestrates homography calculation (via LightGlueStick), image warping, and object detection (via YOLO).
3.  **Model Layer (`lightgluestick/`):**
    -   Contains the implementation of the LightGlueStick matching algorithm, including SuperPoint and LSD extractors.

## Building and Running

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA/MPS support recommended)
- RTSP camera streams or video files

### Installation
```bash
# Clone the repository (if not already present)
git clone <repository_url>
cd Video_Stitching

# Install dependencies
pip install -r requirements.txt
pip install .
```

### Running the Application
```bash
python app.py
```

### Configuration
Edit `config.yaml` to specify your camera URLs and default monitoring region:
```yaml
cameras:
  - url: "rtsp://camera1_ip:port/stream"
  - url: "rtsp://camera2_ip:port/stream"

region_of_interest:
  - [x1, y1]
  - [x2, y2]
  - ...
```

## Development Conventions

-   **Multi-threading:** Always perform heavy computation (stitching, detection) and I/O (video capture) in separate `QThread` instances to avoid freezing the GUI.
-   **Feature Matching:** The project defaults to using LightGlueStick for computing homography. If stitching fails, it falls back to displaying the primary frame.
-   **ROI Persistence:** ROI changes made in the GUI are automatically saved back to `config.yaml`.
-   **Styling:** GUI styling is managed via a CSS-like string in `app.py`.

## Testing
There is currently no automated test suite for the GUI or the pipeline. Manual verification is performed by:
1. Running `python app.py` with valid RTSP streams or video files.
2. Drawing an ROI on the stitched view and verifying person counting.
3. Triggering a "Recalibrate" to test the LightGlueStick homography calculation.

## Key Files
- `app.py`: Application entry point.
- `pipeline.py`: Core stitching and detection logic.
- `widgets.py`: Custom GUI components.
- `config.yaml`: System configuration.
- `lightgluestick/`: Model implementation package.
- `yolo26n.pt`: YOLOv8 model weights.
