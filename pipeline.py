import cv2
import numpy as np
import torch
from lightgluestick.utils import batch_to_np, numpy_image_to_torch
from lightgluestick.two_view_pipeline import TwoViewPipeline
from ultralytics import YOLO

class StitchingPipeline:
    def __init__(self, depth_confidence=-1.0):
        # Setup LightGlueStick Model
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

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
                'depth_confidence': depth_confidence,
                'trainable': False,
            },
            'ground_truth': {
                'from_pose_depth': False,
            }
        }

        self.pipeline_model = TwoViewPipeline(conf).to(self.device).eval()
        self.homography = None

        # Setup YOLOv8 for person detection
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_model.to(self.device)

    def calculate_homography(self, frame1, frame2):
        gray0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
        torch_gray0, torch_gray1 = torch_gray0.to(self.device)[None], torch_gray1.to(self.device)[None]

        with torch.no_grad():
            x = {'view0': {"image": torch_gray0}, 'view1': {"image": torch_gray1}}
            pred = self.pipeline_model(x)

        pred = batch_to_np(pred)

        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]

        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]

        if len(matched_kps0) >= 4:
            H, status = cv2.findHomography(matched_kps1, matched_kps0, cv2.RANSAC, 5.0)
            self.homography = H
            return True
        else:
            return False

    def stitch_frames(self, frame1, frame2):
        if self.homography is None:
            # We must calculate Homography
            success = self.calculate_homography(frame1, frame2)
            if not success:
                return frame1 # Fallback, return frame 1 if unable to stitch

        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        # Using computed homography to warp frame2 to frame1 space
        # Here we do a simple stitching where we combine the images.
        # Determine the canvas size
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        try:
            pts2_ = cv2.perspectiveTransform(pts2, self.homography)
            pts = np.concatenate((pts1, pts2_), axis=0)
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

            t = [-xmin, -ymin]
            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

            result = cv2.warpPerspective(frame2, Ht.dot(self.homography), (xmax-xmin, ymax-ymin))
            result[t[1]:h1+t[1], t[0]:w1+t[0]] = frame1
            return result
        except Exception as e:
            print(f"Homography error: {e}")
            return frame1

    def detect_and_count(self, image, polygon):
        """
        Runs YOLOv8 person detection and counts people within the polygon.
        """
        results = self.yolo_model(image, classes=[0], verbose=False) # 0 is person class in COCO

        count = 0
        img_with_boxes = image.copy()

        # Draw the ROI polygon
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_with_boxes, [pts], True, (255, 0, 0), 2)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Check if center of the box is inside the polygon
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # cv2.pointPolygonTest returns > 0 if inside, 0 if on contour, < 0 if outside
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    count += 1
                    color = (0, 255, 0) # Green for inside
                else:
                    color = (0, 0, 255) # Red for outside

                # Draw bounding box and center point
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.circle(img_with_boxes, (cx, cy), 3, color, -1)

        return img_with_boxes, count
