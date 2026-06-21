import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightgluestick.two_view_pipeline import TwoViewPipeline
from lightgluestick.utils import batch_to_np, numpy_image_to_torch


@dataclass
class RegistrationResult:
    accepted: bool
    message: str
    transform: np.ndarray = None
    quality: dict = None


def _transform_corners(width, height, transform):
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, transform).reshape(-1, 2)


def _as_homogeneous(affine_2x3):
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = affine_2x3
    return H


def _affine_quality(matrix_2x3):
    a, b, tx = matrix_2x3[0]
    c, d, ty = matrix_2x3[1]
    scale_x = math.sqrt(a * a + c * c)
    scale_y = math.sqrt(b * b + d * d)
    scale = (scale_x + scale_y) / 2.0
    rotation_deg = math.degrees(math.atan2(c, a))
    return {
        "dx": float(tx),
        "dy": float(ty),
        "scale": float(scale),
        "rotation_deg": float(rotation_deg),
    }


class LightGlueBevRegistrar:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = None
        self.min_matches = 12
        self.min_inliers = 8
        self.min_inlier_ratio = 0.25
        self.min_scale = 0.70
        self.max_scale = 1.30
        self.max_abs_rotation_deg = 35.0
        self.max_translation_factor = 2.5

    def _ensure_model(self):
        if self.model is not None:
            return
        conf = {
            "name": "two_view_pipeline",
            "use_lines": True,
            "extractor": {
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
            "matcher": {
                "name": "lightgluestick",
                "depth_confidence": -1.0,
                "trainable": False,
            },
            "ground_truth": {"from_pose_depth": False},
        }
        self.model = TwoViewPipeline(conf).to(self.device).eval()

    def _match_points(self, fixed_bev, moving_bev):
        self._ensure_model()
        gray0 = cv2.cvtColor(fixed_bev, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(moving_bev, cv2.COLOR_BGR2GRAY)
        t0 = numpy_image_to_torch(gray0).to(self.device)[None]
        t1 = numpy_image_to_torch(gray1).to(self.device)[None]
        with torch.no_grad():
            pred = self.model({"view0": {"image": t0}, "view1": {"image": t1}})
        pred = batch_to_np(pred)
        kp0 = pred["keypoints0"]
        kp1 = pred["keypoints1"]
        matches0 = pred["matches0"]
        valid = matches0 != -1
        return kp0[valid], kp1[matches0[valid]]

    def register(self, fixed_bev, moving_bev):
        if fixed_bev is None or moving_bev is None:
            return RegistrationResult(False, "缺少 BEV 图像。", quality={})

        try:
            pts_fixed, pts_moving = self._match_points(fixed_bev, moving_bev)
        except Exception as exc:
            return RegistrationResult(False, f"LightGlueStick 运行失败：{exc}", quality={})

        match_count = int(len(pts_fixed))
        quality = {"matches": match_count, "inliers": 0, "inlier_ratio": 0.0}
        if match_count < self.min_matches:
            return RegistrationResult(False, f"匹配点不足：{match_count}", quality=quality)

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            pts_moving,
            pts_fixed,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=3000,
            confidence=0.995,
            refineIters=10,
        )
        if affine is None or inlier_mask is None:
            return RegistrationResult(False, "相似变换估计失败。", quality=quality)

        inliers = int(inlier_mask.ravel().sum())
        inlier_ratio = inliers / max(match_count, 1)
        quality.update(_affine_quality(affine))
        quality.update({
            "matches": match_count,
            "inliers": inliers,
            "inlier_ratio": float(inlier_ratio),
        })

        max_dim = max(fixed_bev.shape[1], fixed_bev.shape[0], moving_bev.shape[1], moving_bev.shape[0])
        reasons = []
        if inliers < self.min_inliers:
            reasons.append(f"inliers={inliers} < {self.min_inliers}")
        if inlier_ratio < self.min_inlier_ratio:
            reasons.append(f"inlier_ratio={inlier_ratio:.2f} < {self.min_inlier_ratio:.2f}")
        if not (self.min_scale <= quality["scale"] <= self.max_scale):
            reasons.append(f"scale={quality['scale']:.2f} 超出范围")
        if abs(quality["rotation_deg"]) > self.max_abs_rotation_deg:
            reasons.append(f"rotation={quality['rotation_deg']:.1f}° 过大")
        if max(abs(quality["dx"]), abs(quality["dy"])) > max_dim * self.max_translation_factor:
            reasons.append("平移量异常")

        if reasons:
            return RegistrationResult(False, "自动配准被拒绝：" + "；".join(reasons), quality=quality)

        return RegistrationResult(True, "自动配准成功。", _as_homogeneous(affine), quality)


class MultiCameraFusion:
    def __init__(self, registrar=None, canvas_padding=40):
        self.registrar = registrar or LightGlueBevRegistrar()
        self.canvas_padding = int(canvas_padding)

    def compute_pairwise_registrations(self, cameras, bev_images):
        active = [cam for cam in cameras if cam.enabled and cam.is_calibrated and cam.camera_id in bev_images]
        if not active:
            return "没有可用的已标定摄像头。"

        active[0].bev_to_mosaic_transform = np.eye(3, dtype=np.float64)
        active[0].last_registration_quality = {
            "anchor": True,
            "message": "Anchor camera",
        }
        messages = [f"{active[0].name}: anchor"]

        prev = active[0]
        for cam in active[1:]:
            result = self.registrar.register(bev_images[prev.camera_id], bev_images[cam.camera_id])
            quality = dict(result.quality or {})
            quality["message"] = result.message
            quality["fixed_camera"] = prev.camera_id
            quality["moving_camera"] = cam.camera_id
            if result.accepted:
                cam.bev_to_mosaic_transform = prev.bev_to_mosaic_transform @ result.transform
                cam.last_registration_quality = quality
                prev = cam
                messages.append(
                    f"{cam.name}: OK matches={quality.get('matches', 0)} "
                    f"inliers={quality.get('inliers', 0)} ratio={quality.get('inlier_ratio', 0):.2f}"
                )
            else:
                cam.last_registration_quality = quality
                if cam.bev_to_mosaic_transform is not None:
                    prev = cam
                messages.append(f"{cam.name}: FAIL {result.message}")
        return " | ".join(messages)

    def build_mosaic(self, cameras, frames_by_id):
        entries = []
        has_anchor = False
        for cam in cameras:
            if not cam.enabled or not cam.is_calibrated:
                continue
            frame = frames_by_id.get(cam.camera_id)
            if frame is None:
                continue
            transform = cam.bev_to_mosaic_transform
            if transform is None:
                if has_anchor:
                    continue
                transform = np.eye(3, dtype=np.float64)
            bev, mask = cam.warp_to_bev(frame)
            if bev is None or mask is None:
                continue
            entries.append((cam, bev, mask, transform))
            has_anchor = True

        if not entries:
            return None, {}

        all_corners = []
        for cam, bev, mask, transform in entries:
            all_corners.append(_transform_corners(bev.shape[1], bev.shape[0], transform))
        all_corners = np.vstack(all_corners)
        min_xy = np.floor(all_corners.min(axis=0)).astype(int)
        max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
        width = int(max_xy[0] - min_xy[0] + 2 * self.canvas_padding)
        height = int(max_xy[1] - min_xy[1] + 2 * self.canvas_padding)
        if width <= 0 or height <= 0:
            return None, {}

        shift = np.array([
            [1, 0, -min_xy[0] + self.canvas_padding],
            [0, 1, -min_xy[1] + self.canvas_padding],
            [0, 0, 1],
        ], dtype=np.float64)

        accum = np.zeros((height, width, 3), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        debug_bevs = {}

        for cam, bev, mask, transform in entries:
            T = shift @ transform
            warped = cv2.warpPerspective(bev, T, (width, height), flags=cv2.INTER_LINEAR)
            warped_mask = cv2.warpPerspective(mask, T, (width, height), flags=cv2.INTER_NEAREST)
            binary = (warped_mask > 0).astype(np.uint8)
            weight = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            if weight.max() <= 0:
                weight = binary.astype(np.float32)
            accum += warped.astype(np.float32) * weight[..., None]
            weights += weight
            debug_bevs[cam.camera_id] = bev

        mosaic = np.zeros_like(accum, dtype=np.uint8)
        valid = weights > 0
        mosaic[valid] = np.clip(accum[valid] / weights[valid, None], 0, 255).astype(np.uint8)
        return mosaic, debug_bevs
