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

# 配准前将 BEV 最长边缩到该尺寸以内,匹配点坐标再放大回原分辨率
MAX_MATCH_DIM = 1024


@dataclass
class RegistrationResult:
    accepted: bool
    message: str
    transform: np.ndarray = None
    quality: dict = None


def estimate_rigid_transform(src_pts, dst_pts):
    """Kabsch:估计 src → dst 的纯旋转+平移(scale 恒为 1)。

    两路 BEV 由同一 px/m 比例生成,物理上只可能差刚体变换;
    放开 scale 自由度只会让匹配噪声污染物理尺度。
    """
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 2)
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    cov = (src - centroid_src).T @ (dst - centroid_dst)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, d]) @ u.T
    translation = centroid_dst - rotation @ centroid_src
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rotation
    transform[:2, 2] = translation
    return transform


def solve_layout(node_count, edges, anchor):
    """由通过质检的两两配准边构建最大生成树,从 anchor 出发合成全局变换。

    edges: [(score, i, j, T_ji), ...],T_ji 把节点 j 的坐标映射到节点 i。
    返回 {节点下标: 3x3 变换到 anchor 坐标系},不连通节点不在结果中。
    """
    parent = list(range(node_count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adjacency = {i: [] for i in range(node_count)}
    for score, i, j, transform in sorted(edges, key=lambda e: e[0], reverse=True):
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            continue
        parent[root_i] = root_j
        adjacency[i].append((j, transform, True))    # M_j = M_i @ T
        adjacency[j].append((i, transform, False))   # M_i = M_j @ inv(T)

    transforms = {anchor: np.eye(3, dtype=np.float64)}
    queue = [anchor]
    while queue:
        node = queue.pop(0)
        for neighbor, transform, forward in adjacency[node]:
            if neighbor in transforms:
                continue
            if forward:
                transforms[neighbor] = transforms[node] @ transform
            else:
                transforms[neighbor] = transforms[node] @ np.linalg.inv(transform)
            queue.append(neighbor)
    return transforms


def _resize_for_match(image):
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= MAX_MATCH_DIM:
        return image, 1.0
    scale = MAX_MATCH_DIM / longest
    resized = cv2.resize(
        image,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


class LightGlueBevRegistrar:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = None
        self.min_matches = 12
        self.min_inliers = 8
        self.min_inlier_ratio = 0.25
        # 相似变换估出的 scale 偏离 1 超过该值,说明两路标定的物理比例对不上
        self.scale_tolerance = 0.08
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
        fixed_small, scale_fixed = _resize_for_match(fixed_bev)
        moving_small, scale_moving = _resize_for_match(moving_bev)
        gray0 = cv2.cvtColor(fixed_small, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(moving_small, cv2.COLOR_BGR2GRAY)
        t0 = numpy_image_to_torch(gray0).to(self.device)[None]
        t1 = numpy_image_to_torch(gray1).to(self.device)[None]
        with torch.no_grad():
            pred = self.model({"view0": {"image": t0}, "view1": {"image": t1}})
        pred = batch_to_np(pred)
        kp0 = pred["keypoints0"]
        kp1 = pred["keypoints1"]
        matches0 = pred["matches0"]
        valid = matches0 != -1
        return kp0[valid] / scale_fixed, kp1[matches0[valid]] / scale_moving

    def register(self, fixed_bev, moving_bev):
        if fixed_bev is None or moving_bev is None:
            return RegistrationResult(False, "缺少 BEV 图像。", quality={})

        try:
            pts_fixed, pts_moving = self._match_points(fixed_bev, moving_bev)
        except Exception as exc:
            return RegistrationResult(False, f"LightGlueStick 运行失败:{exc}", quality={})

        match_count = int(len(pts_fixed))
        quality = {"matches": match_count, "inliers": 0, "inlier_ratio": 0.0}
        if match_count < self.min_matches:
            return RegistrationResult(False, f"匹配点不足:{match_count}", quality=quality)

        # 相似变换 RANSAC 仅用于筛内点和检测尺度一致性,最终变换用刚体重拟合
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

        inlier_mask = inlier_mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / max(match_count, 1)
        sim_scale = math.sqrt(abs(affine[0, 0] * affine[1, 1] - affine[0, 1] * affine[1, 0]))

        transform = estimate_rigid_transform(
            pts_moving[inlier_mask], pts_fixed[inlier_mask]
        )
        rotation_deg = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        quality.update({
            "matches": match_count,
            "inliers": inliers,
            "inlier_ratio": float(inlier_ratio),
            "scale": float(sim_scale),
            "rotation_deg": float(rotation_deg),
            "dx": float(transform[0, 2]),
            "dy": float(transform[1, 2]),
        })

        max_dim = max(fixed_bev.shape[1], fixed_bev.shape[0],
                      moving_bev.shape[1], moving_bev.shape[0])
        reasons = []
        if inliers < self.min_inliers:
            reasons.append(f"inliers={inliers} < {self.min_inliers}")
        if inlier_ratio < self.min_inlier_ratio:
            reasons.append(f"inlier_ratio={inlier_ratio:.2f} < {self.min_inlier_ratio:.2f}")
        if abs(sim_scale - 1.0) > self.scale_tolerance:
            reasons.append(
                f"scale={sim_scale:.3f} 偏离 1(两路相机的标定物理比例可能不一致)"
            )
        if abs(rotation_deg) > self.max_abs_rotation_deg:
            reasons.append(f"rotation={rotation_deg:.1f}° 过大")
        if max(abs(quality["dx"]), abs(quality["dy"])) > max_dim * self.max_translation_factor:
            reasons.append("平移量异常")

        if reasons:
            return RegistrationResult(False, "自动配准被拒绝:" + ";".join(reasons), quality=quality)

        return RegistrationResult(True, "自动配准成功。", transform, quality)


class MultiCameraFusion:
    def __init__(self, registrar=None, canvas_padding=40):
        self.registrar = registrar or LightGlueBevRegistrar()
        self.canvas_padding = int(canvas_padding)

    def compute_pairwise_registrations(self, cameras, bev_images):
        active = [
            cam for cam in cameras
            if cam.enabled and cam.is_calibrated and cam.camera_id in bev_images
        ]
        # 先清空全部旧变换:上一轮的结果属于不同的 anchor/布局,
        # 复用会把后续相机挂到不一致的参考系上
        for cam in active:
            cam.bev_to_mosaic_transform = None
            cam.last_registration_quality = {}

        if not active:
            return "没有可用的已标定摄像头。"
        if len(active) == 1:
            active[0].bev_to_mosaic_transform = np.eye(3, dtype=np.float64)
            active[0].last_registration_quality = {"anchor": True, "message": "Anchor camera"}
            return f"{active[0].name}: anchor(仅一路可用)"

        # 两两配准(N≤4 → 最多 6 对),按 inliers 选边
        edges = []
        edge_quality = {}
        messages = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                result = self.registrar.register(
                    bev_images[active[i].camera_id],
                    bev_images[active[j].camera_id],
                )
                pair_name = f"{active[i].name}↔{active[j].name}"
                quality = dict(result.quality or {})
                quality["message"] = result.message
                quality["fixed_camera"] = active[i].camera_id
                quality["moving_camera"] = active[j].camera_id
                if result.accepted:
                    score = quality.get("inliers", 0)
                    edges.append((score, i, j, result.transform))
                    edge_quality[(i, j)] = quality
                    messages.append(
                        f"{pair_name}: OK inliers={quality.get('inliers', 0)} "
                        f"ratio={quality.get('inlier_ratio', 0):.2f}"
                    )
                else:
                    messages.append(f"{pair_name}: FAIL {result.message}")

        if not edges:
            return "所有相机对配准均失败:" + " | ".join(messages)

        # anchor 取连通边 inliers 总和最大的相机
        scores = [0] * len(active)
        for score, i, j, _ in edges:
            scores[i] += score
            scores[j] += score
        anchor = int(np.argmax(scores))

        transforms = solve_layout(len(active), edges, anchor)

        active[anchor].bev_to_mosaic_transform = np.eye(3, dtype=np.float64)
        active[anchor].last_registration_quality = {"anchor": True, "message": "Anchor camera"}
        placed, unplaced = [], []
        for idx, cam in enumerate(active):
            if idx == anchor:
                continue
            if idx in transforms:
                cam.bev_to_mosaic_transform = transforms[idx]
                quality = edge_quality.get((anchor, idx)) or edge_quality.get((idx, anchor)) \
                    or next((q for (a, b), q in edge_quality.items() if idx in (a, b)), {})
                cam.last_registration_quality = quality
                placed.append(cam.name)
            else:
                cam.last_registration_quality = {"message": "未能与其它相机连通,保持未拼接"}
                unplaced.append(cam.name)

        summary = f"anchor={active[anchor].name}"
        if unplaced:
            summary += f" | 未连通:{', '.join(unplaced)}"
        return summary + " | " + " | ".join(messages)
