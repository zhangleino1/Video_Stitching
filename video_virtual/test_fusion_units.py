"""纯计算部分的单元测试(不依赖 GUI 和视频流)。

运行:python test_fusion_units.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import CameraCalibration
from fusion import estimate_rigid_transform, solve_layout
from fusion_cache import build_fusion_cache, compute_signature, fuse


def _translation(dx, dy):
    t = np.eye(3)
    t[0, 2] = dx
    t[1, 2] = dy
    return t


def test_rigid_transform():
    rng = np.random.default_rng(0)
    theta = 0.3
    rotation = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])
    translation = np.array([12.0, -7.0])
    src = rng.random((30, 2)) * 200
    dst = (rotation @ src.T).T + translation

    transform = estimate_rigid_transform(src, dst)
    assert np.allclose(transform[:2, :2], rotation, atol=1e-9)
    assert np.allclose(transform[:2, 2], translation, atol=1e-9)
    # scale 恒为 1
    assert abs(np.linalg.det(transform[:2, :2]) - 1.0) < 1e-9
    print("test_rigid_transform ok")


def test_solve_layout_basic():
    t10 = _translation(100, 0)   # 节点 1 → 节点 0
    transforms = solve_layout(3, [(50, 0, 1, t10)], anchor=0)
    assert set(transforms) == {0, 1}          # 节点 2 不连通,不应被放置
    assert np.allclose(transforms[0], np.eye(3))
    assert np.allclose(transforms[1], t10)
    print("test_solve_layout_basic ok")


def test_solve_layout_reverse_direction():
    t10 = _translation(100, 0)
    transforms = solve_layout(2, [(50, 0, 1, t10)], anchor=1)
    assert np.allclose(transforms[0], np.linalg.inv(t10))
    print("test_solve_layout_reverse_direction ok")


def test_solve_layout_prefers_high_score_edges():
    # 三角形:0-1 (强), 1-2 (强), 0-2 (弱且带错误平移)
    # 最大生成树应使用两条强边,弱边被丢弃
    t10 = _translation(100, 0)
    t21 = _translation(0, 50)
    t20_wrong = _translation(999, 999)
    edges = [
        (90, 0, 1, t10),
        (80, 1, 2, t21),
        (5, 0, 2, t20_wrong),
    ]
    transforms = solve_layout(3, edges, anchor=0)
    assert np.allclose(transforms[1], t10)
    assert np.allclose(transforms[2], t10 @ t21)   # 经由节点 1 合成
    print("test_solve_layout_prefers_high_score_edges ok")


def test_calibration_residuals():
    cam = CameraCalibration("c1", "C1")
    cam.image_points = [[100, 100], [500, 100], [500, 400], [100, 400]]
    cam.local_world_points = [[0, 0], [4, 0], [4, 3], [0, 3]]
    ok, msg = cam.compute_local_bev(100.0, 20)
    assert ok, msg
    assert len(cam.last_residuals_cm) == 4
    assert max(cam.last_residuals_cm) < 1.0   # 精确对应点,残差应接近 0
    print("test_calibration_residuals ok")


def test_calibration_rejects_duplicate_world_points():
    cam = CameraCalibration("c1", "C1")
    cam.image_points = [[100, 100], [500, 100], [500, 400], [100, 400], [300, 250]]
    cam.local_world_points = [[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]]  # 第 5 点重复
    ok, msg = cam.compute_local_bev(100.0, 20)
    assert not ok
    assert "重复" in msg
    print("test_calibration_rejects_duplicate_world_points ok")


def _make_calibrated_camera(camera_id, transform=None):
    cam = CameraCalibration(camera_id, camera_id)
    cam.image_points = [[50, 50], [590, 50], [590, 430], [50, 430]]
    cam.local_world_points = [[0, 0], [4, 0], [4, 3], [0, 3]]
    ok, msg = cam.compute_local_bev(50.0, 10)
    assert ok, msg
    cam.bev_to_mosaic_transform = transform
    return cam


def test_cache_and_fuse():
    cam_a = _make_calibrated_camera("camA", np.eye(3))
    cam_b = _make_calibrated_camera("camB", _translation(150, 0))
    frame_a = np.full((480, 640, 3), 200, np.uint8)
    frame_b = np.full((480, 640, 3), 100, np.uint8)
    frames = {"camA": frame_a, "camB": frame_b}

    signature = compute_signature([cam_a, cam_b], frames, 10)
    cache = build_fusion_cache([cam_a, cam_b], frames, 10, signature)
    assert cache is not None
    assert len(cache.entries) == 2
    width, height = cache.canvas_size
    assert width > 0 and height > 0

    # 有效区内权重之和应为 1
    w_sum = sum(e.weight for e in cache.entries)
    valid = w_sum > 1e-6
    assert valid.any()
    assert np.allclose(w_sum[valid], 1.0, atol=1e-4)

    mosaic = fuse(cache, frames)
    assert mosaic is not None
    assert mosaic.shape[:2] == (height, width)
    interior = mosaic[height // 2 - 5:height // 2 + 5, :, 0]
    assert interior.max() > 0   # 画布中部应有内容

    # 仅 camA 区域(远离 camB)应接近 200;重叠区介于 100~200
    assert mosaic.max() <= 200 + 1
    print("test_cache_and_fuse ok")


def test_cache_single_camera_without_transform():
    # 未配准时,第一台已标定相机应以恒等变换显示
    cam = _make_calibrated_camera("camA", None)
    frames = {"camA": np.full((480, 640, 3), 128, np.uint8)}
    cache = build_fusion_cache([cam], frames, 10, ())
    assert cache is not None and len(cache.entries) == 1
    mosaic = fuse(cache, frames)
    assert mosaic is not None and mosaic.max() > 0
    print("test_cache_single_camera_without_transform ok")


def test_signature_changes_on_transform_update():
    cam = _make_calibrated_camera("camA", np.eye(3))
    frames = {"camA": np.zeros((480, 640, 3), np.uint8)}
    sig1 = compute_signature([cam], frames, 10)
    cam.bev_to_mosaic_transform = _translation(5, 5)
    sig2 = compute_signature([cam], frames, 10)
    assert sig1 != sig2
    print("test_signature_changes_on_transform_update ok")


if __name__ == "__main__":
    test_rigid_transform()
    test_solve_layout_basic()
    test_solve_layout_reverse_direction()
    test_solve_layout_prefers_high_score_edges()
    test_calibration_residuals()
    test_calibration_rejects_duplicate_world_points()
    test_cache_and_fuse()
    test_cache_single_camera_without_transform()
    test_signature_changes_on_transform_update()
    print("\n全部测试通过。")
