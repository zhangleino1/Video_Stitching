import sys
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightgluestick.utils import batch_to_np, numpy_image_to_torch
from lightgluestick.viz2d import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from lightgluestick.two_view_pipeline import TwoViewPipeline

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"

def capture_frame(rtsp_url, num_frames_to_skip=5):
    print(f"Connecting to: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open stream: {rtsp_url}")
        return None
    
    # Read a few frames to clear any initial buffers and get a stable image
    for _ in range(num_frames_to_skip):
        ret, frame = cap.read()
        if not ret:
            break
            
    cap.release()
    if not ret:
        print(f"Failed to read frame from: {rtsp_url}")
        return None
    return frame

def main():
    # Load config
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading config.yaml: {e}")
        return
        
    cameras = config.get("cameras", [])
    if len(cameras) < 2:
        print("Need at least 2 cameras defined in config.yaml")
        return
        
    url1 = cameras[0]["url"]
    url2 = cameras[1]["url"]
    
    # Capture frames
    frame0_bgr = capture_frame(url1)
    frame1_bgr = capture_frame(url2)
    
    if frame0_bgr is None or frame1_bgr is None:
        print("Failed to capture frames from RTSP streams. Exiting.")
        return
    
    # Save the captured frames for reference
    cv2.imwrite(str(HERE / "rtsp_frame1.jpg"), frame0_bgr)
    cv2.imwrite(str(HERE / "rtsp_frame2.jpg"), frame1_bgr)
    print("Saved captured frames to rtsp_frame1.jpg and rtsp_frame2.jpg")
    
    # Convert to grayscale for LightGlueStick
    gray0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
    
    # Evaluation config (same as run.py)
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            "name": "wireframe",
            "point_extractor": {
                "name": "superpoint",
                "trainable": False,
                "dense_outputs": True,
                "max_num_keypoints": 2048,
                "force_num_keypoints": False,
            },
            "line_extractor": {
                "name": "lsd",
                "trainable": False,
                "max_num_lines": 250,
                "force_num_lines": False,
                "min_length": 15,
            },
            "wireframe_params": {
                "merge_points": True,
                "merge_line_endpoints": True,
                "nms_radius": 3,
            },
        },
        'matcher': {
            'name': 'lightgluestick',
            'depth_confidence': -1.0,
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    pipeline_model = TwoViewPipeline(conf).to(device).eval()

    torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
    x = {'view0': {"image": torch_gray0}, 'view1': {"image": torch_gray1}}
    
    print("Running LightGlueStick matching...")
    with torch.no_grad():
        pred = pipeline_model(x)

    pred = batch_to_np(pred)
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]

    line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
    line_matches = pred["line_matches0"]

    valid_matches = m0 != -1
    match_indices = m0[valid_matches]
    matched_kps0 = kp0[valid_matches]
    matched_kps1 = kp1[match_indices]

    valid_matches = line_matches != -1
    match_indices = line_matches[valid_matches]
    matched_lines0 = line_seg0[valid_matches]
    matched_lines1 = line_seg1[match_indices]

    # Plot the matches
    img0 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    
    print("Saving match visualizations...")
    plot_images([img0, img1], ['Image 1 - detected lines', 'Image 2 - detected lines'], dpi=200, pad=2.0)
    plot_lines([line_seg0, line_seg1], ps=4, lw=2)
    plt.gcf().canvas.manager.set_window_title('Detected Lines')
    plt.savefig(str(HERE / "rtsp_detected_lines.png"))

    plot_images([img0, img1], ['Image 1 - detected points', 'Image 2 - detected points'], dpi=200, pad=2.0)
    plot_keypoints([kp0, kp1], colors='c')
    plt.gcf().canvas.manager.set_window_title('Detected Points')
    plt.savefig(str(HERE / "rtsp_detected_points.png"))

    plot_images([img0, img1], ['Image 1 - line matches', 'Image 2 - line matches'], dpi=200, pad=2.0)
    plot_color_line_matches([matched_lines0, matched_lines1], lw=2)
    plt.gcf().canvas.manager.set_window_title('Line Matches')
    plt.savefig(str(HERE / "rtsp_line_matches.png"))

    plot_images([img0, img1], ['Image 1 - point matches', 'Image 2 - point matches'], dpi=200, pad=2.0)
    plot_matches(matched_kps0, matched_kps1, 'green', lw=1, ps=0)
    plt.gcf().canvas.manager.set_window_title('Point Matches')
    plt.savefig(str(HERE / "rtsp_point_matches.png"))
    
    print("Test completed successfully! Check the rtsp_*.png files for results.")

if __name__ == '__main__':
    main()
