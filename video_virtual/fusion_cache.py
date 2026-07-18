"""融合缓存:把所有与帧内容无关的计算(画布尺寸、逆映射表、羽化权重)
在标定/配准变化时一次性预计算,运行时每路相机只做一次 cv2.remap + 乘加。"""

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_CANVAS_WIDTH = 6000
MAX_CANVAS_HEIGHT = 6000


@dataclass
class CameraFusionEntry:
    camera_id: str
    map1: np.ndarray          # 定点化逆映射表 (CV_16SC2):画布像素 → 原始图像像素
    map2: np.ndarray          # CV_16UC1 插值表
    weight: np.ndarray        # float32, 画布尺寸,跨相机归一化(有效区内 Σ=1)
    weight3: np.ndarray = None  # weight 的 3 通道版,供 cv2.multiply 使用(比 numpy 广播快)


@dataclass
class FusionCache:
    canvas_size: tuple        # (width, height)
    entries: list = field(default_factory=list)
    signature: tuple = ()


def compute_signature(cameras, frames, padding):
    """缓存签名:标定矩阵、配准变换、帧尺寸任一变化都会触发重建。"""
    parts = []
    for cam in cameras:
        if not (cam.enabled and cam.is_calibrated):
            continue
        frame = frames.get(cam.camera_id)
        if frame is None:
            continue
        transform = cam.bev_to_mosaic_transform
        parts.append((
            cam.camera_id,
            tuple(frame.shape),
            cam.local_bev_homography.tobytes(),
            transform.tobytes() if transform is not None else None,
            tuple(tuple(p) for p in cam.image_points),
        ))
    return (tuple(parts), int(padding))


def _inverse_maps(matrix_canvas_to_src_inv, width, height):
    """由画布→源图像的逆单应生成 remap 用的 map_x/map_y。"""
    m = matrix_canvas_to_src_inv
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    denom = m[2, 0] * xs + m[2, 1] * ys + m[2, 2]
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    map_x = (m[0, 0] * xs + m[0, 1] * ys + m[0, 2]) / denom
    map_y = (m[1, 0] * xs + m[1, 1] * ys + m[1, 2]) / denom
    return map_x.astype(np.float32), map_y.astype(np.float32)


def build_fusion_cache(cameras, frames, padding, signature=()):
    """预计算融合缓存。返回 FusionCache,若无可用相机返回 None。

    每路相机的总变换合成为 原始图像 → 画布 的单一单应,
    避免 原图→BEV→画布 的双重 warp(双重插值 + 双倍计算)。
    """
    placed = []
    for cam in cameras:
        if not (cam.enabled and cam.is_calibrated):
            continue
        if frames.get(cam.camera_id) is None:
            continue
        placed.append(cam)
    if not placed:
        return None

    with_transform = [c for c in placed if c.bev_to_mosaic_transform is not None]
    if with_transform:
        selected = [(c, c.bev_to_mosaic_transform) for c in with_transform]
    else:
        # 尚未配准:单相机以恒等变换显示
        selected = [(placed[0], np.eye(3, dtype=np.float64))]

    items = []
    all_pts = []
    for cam, transform in selected:
        h_total = transform @ cam.local_bev_homography  # 原图 → 画布(未平移)
        hull_pts = np.array(cam.image_points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(hull_pts, h_total).reshape(-1, 2)
        items.append((cam, h_total))
        all_pts.append(projected)

    all_pts = np.vstack(all_pts)
    min_xy = np.floor(all_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int)
    pad = int(padding)
    width = int(max_xy[0] - min_xy[0] + 2 * pad)
    height = int(max_xy[1] - min_xy[1] + 2 * pad)
    width = min(max(width, 32), MAX_CANVAS_WIDTH)
    height = min(max(height, 32), MAX_CANVAS_HEIGHT)

    shift = np.array([
        [1, 0, -min_xy[0] + pad],
        [0, 1, -min_xy[1] + pad],
        [0, 0, 1],
    ], dtype=np.float64)

    entries = []
    raw_weights = []
    for cam, h_total in items:
        m = shift @ h_total
        try:
            m_inv = np.linalg.inv(m)
        except np.linalg.LinAlgError:
            continue
        map_x, map_y = _inverse_maps(m_inv, width, height)
        map1, map2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

        frame = frames[cam.camera_id]
        src_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(
            np.array(cam.image_points, dtype=np.float32)
        ).astype(np.int32)
        cv2.fillPoly(src_mask, [hull], 255)
        warped_mask = cv2.warpPerspective(
            src_mask, m, (width, height), flags=cv2.INTER_NEAREST
        )
        binary = (warped_mask > 0).astype(np.uint8)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            dist = binary.astype(np.float32)

        entries.append(CameraFusionEntry(cam.camera_id, map1, map2, None))
        raw_weights.append(dist.astype(np.float32))

    if not entries:
        return None

    weight_sum = np.sum(raw_weights, axis=0)
    valid = weight_sum > 0
    denom = np.where(valid, weight_sum, 1.0)
    for entry, raw in zip(entries, raw_weights):
        entry.weight = np.where(valid, raw / denom, 0.0).astype(np.float32)
        entry.weight3 = cv2.merge([entry.weight] * 3)

    return FusionCache((width, height), entries, signature)


def fuse(cache, frames):
    """运行时融合:每路相机一次 remap,加权累加。权重已归一化,无需再除。

    乘加走 cv2(SIMD + 多线程),比 numpy 广播快约 1/3。"""
    accum = None
    for entry in cache.entries:
        frame = frames.get(entry.camera_id)
        if frame is None:
            continue
        warped = cv2.remap(frame, entry.map1, entry.map2, cv2.INTER_LINEAR)
        contrib = cv2.multiply(warped.astype(np.float32), entry.weight3)
        accum = contrib if accum is None else cv2.add(accum, contrib)
    if accum is None:
        return None
    return cv2.convertScaleAbs(accum)
