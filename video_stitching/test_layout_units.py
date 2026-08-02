"""拼接布局与曝光补偿的单元测试（纯计算，不需要相机或模型）。

运行：python video_stitching/test_layout_units.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (GAIN_LIMITS, MAX_COMPOSED_AREA_RATIO, MAX_HEIGHT,
                      MAX_WIDTH, SPAN_SLACK, StitchingPipeline,
                      _is_sane_homography, solve_gains, solve_layout)

SHAPE = (1080, 1920)


class _Frame:
    """只带 shape 的假帧，_prune_diverged 只用得到 shape。"""

    def __init__(self, shape=SHAPE):
        self.shape = shape


def _prune(Hs, root, shape=SHAPE):
    """在不构造完整 pipeline 的前提下调用剪枝逻辑。"""
    frames = [_Frame(shape) for _ in Hs]
    diverged = StitchingPipeline._prune_diverged(
        StitchingPipeline, frames, Hs, root)
    return Hs, diverged


def _scale(s):
    H = np.eye(3)
    H[0, 0] = H[1, 1] = s
    return H


def _translation(dx, dy=0.0):
    H = np.eye(3)
    H[0, 2] = dx
    H[1, 2] = dy
    return H


def test_sane_homography_accepts_identity():
    assert _is_sane_homography(np.eye(3), SHAPE)
    assert _is_sane_homography(_translation(300), SHAPE)
    print("test_sane_homography_accepts_identity ok")


def test_sane_homography_rejects_degenerate():
    assert not _is_sane_homography(None, SHAPE)
    assert not _is_sane_homography(np.full((3, 3), np.nan), SHAPE)
    # 压扁到近乎没有面积
    flat = np.diag([1.0, 0.001, 1.0])
    assert not _is_sane_homography(flat, SHAPE)
    # 左右镜像翻面
    flip = np.diag([-1.0, 1.0, 1.0])
    assert not _is_sane_homography(flip, SHAPE)
    print("test_sane_homography_rejects_degenerate ok")


def test_layout_picks_center_as_root():
    # 链式布局 0—1—2：中心是 1，形变向两侧各摊一半
    edges = [(120, 0, 1, _translation(500)), (110, 1, 2, _translation(500))]
    transforms, root = solve_layout(3, edges)
    assert root == 1, root
    assert len(transforms) == 3
    assert np.allclose(transforms[1], np.eye(3))
    print("test_layout_picks_center_as_root ok")


def test_layout_is_order_independent():
    """相机在 config 中的顺序不同，但真实相邻关系相同 → 布局一致。

    真实链路是 1—0—2（cam0 在中间），无论边以什么顺序给出都应得到同样结果。
    """
    e_a = [(173, 0, 1, _translation(-600)), (113, 0, 2, _translation(600))]
    e_b = [(113, 0, 2, _translation(600)), (173, 0, 1, _translation(-600))]
    ta, ra = solve_layout(3, e_a)
    tb, rb = solve_layout(3, e_b)
    assert ra == rb == 0
    for k in ta:
        assert np.allclose(ta[k], tb[k])
    print("test_layout_is_order_independent ok")


def test_layout_drops_unreachable_camera():
    # cam2 没有任何通过门限的边 → 不应被放置
    edges = [(150, 0, 1, _translation(400))]
    transforms, root = solve_layout(3, edges)
    assert 2 not in transforms
    assert set(transforms) == {0, 1}
    print("test_layout_drops_unreachable_camera ok")


def test_layout_prefers_strong_edges():
    # 三角形中最弱的一条边（且变换错误）应被生成树丢弃
    edges = [
        (200, 0, 1, _translation(500)),
        (180, 1, 2, _translation(500)),
        (5, 0, 2, _translation(9999)),
    ]
    transforms, root = solve_layout(3, edges)
    assert np.allclose(transforms[2], transforms[1] @ _translation(500))
    print("test_layout_prefers_strong_edges ok")


def test_gains_equalise_brightness():
    # cam0 比 cam1 暗一档：增益应把暗的抬高、亮的压低
    stats = {(0, 1): (100000, 80.0, 120.0)}
    g = solve_gains(stats, 2)
    assert g[0] > 1.0 > g[1], g

    # Brown & Lowe 的先验项会把增益拉向 1，因此不会完全拉平，
    # 契约是「亮度台阶显著缩小」——这里要求至少减半。
    before = abs(80.0 - 120.0)
    after = abs(g[0] * 80.0 - g[1] * 120.0)
    assert after < before * 0.5, (g, before, after)
    print(f"test_gains_equalise_brightness ok (台阶 {before:.0f} → {after:.1f})")


def test_gains_identity_when_balanced():
    stats = {(0, 1): (100000, 100.0, 100.0)}
    g = solve_gains(stats, 2)
    assert np.allclose(g, 1.0, atol=1e-3), g
    print("test_gains_identity_when_balanced ok")


def test_gains_are_bounded_and_normalised():
    # 极端亮度差不应产生离谱增益
    stats = {(0, 1): (100000, 5.0, 250.0)}
    g = solve_gains(stats, 2)
    assert np.all(g >= GAIN_LIMITS[0]) and np.all(g <= GAIN_LIMITS[1]), g
    assert abs(float(g.mean()) - 1.0) < 0.5, g
    print("test_gains_are_bounded_and_normalised ok")


def test_gains_no_overlap_returns_ones():
    g = solve_gains({}, 3)
    assert np.allclose(g, 1.0)
    print("test_gains_no_overlap_returns_ones ok")


def test_prune_keeps_healthy_layout():
    """正常布局（各路缩放接近 1、画布装得下）不应剪掉任何相机。

    横向总跨度 = 1920 + 2×1000 = 3920 < MAX_WIDTH，刚好装得下。
    """
    Hs = [np.eye(3), _translation(1000), _translation(-1000)]
    Hs, diverged = _prune(Hs, root=0)
    assert diverged == []
    assert all(H is not None for H in Hs)
    print("test_prune_keeps_healthy_layout ok")


def test_prune_drops_blown_up_camera():
    """某路被链式放大到离谱倍数 → 剪掉它，而不是让它撑爆画布。"""
    Hs = [np.eye(3), _translation(800), _scale(3.0)]   # cam2 面积放大 9 倍
    Hs, diverged = _prune(Hs, root=0)
    assert Hs[2] is None, "发散的相机应被剪掉"
    assert Hs[0] is not None and Hs[1] is not None, "健康的相机应保留"
    assert diverged and diverged[0][0] == 2
    print(f"test_prune_drops_blown_up_camera ok (剪掉 cam2，{diverged[0][1]:.1f}×)")


def test_prune_never_drops_root():
    """即使只剩根相机，也不能把根剪掉。"""
    Hs = [_scale(4.0), _scale(4.0)]
    Hs, _ = _prune(Hs, root=0)
    assert Hs[0] is not None, "根相机必须保留"
    print("test_prune_never_drops_root ok")


def test_prune_tolerates_mild_overflow():
    """轻微超出画布上限只应裁掉边缘，不应剪掉整路相机。

    跨度 1920 + 2×1600 = 5120，超过 MAX_WIDTH(4000) 但在 SPAN_SLACK 容忍范围内，
    且各路缩放都正常 —— 这是旧代码「裁一点边」的行为，必须保留。
    """
    assert MAX_WIDTH < 5120 <= MAX_WIDTH * SPAN_SLACK, "测试前提失效"
    Hs = [np.eye(3), _translation(1600), _translation(-1600)]
    Hs, diverged = _prune(Hs, root=0)
    assert diverged == [], diverged
    assert all(H is not None for H in Hs)
    print("test_prune_tolerates_mild_overflow ok")


def test_prune_shrinks_canvas_when_way_too_big():
    """跨度远超上限时逐路剪枝，直到落回容忍范围内。"""
    Hs = [np.eye(3), _translation(9000), _translation(-9000)]
    Hs, diverged = _prune(Hs, root=0)
    placed = [i for i, H in enumerate(Hs) if H is not None]
    assert diverged, "跨度 ~20000 应触发剪枝"
    h, w = SHAPE
    xs = []
    for i in placed:
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        c = (Hs[i][:2, :2] @ c.T).T + Hs[i][:2, 2]
        xs.append(c)
    xs = np.vstack(xs)
    assert np.ptp(xs[:, 0]) <= MAX_WIDTH * SPAN_SLACK, np.ptp(xs[:, 0])
    print(f"test_prune_shrinks_canvas_when_way_too_big ok (剪掉 {len(diverged)} 路)")


def test_prune_handles_missing_root():
    Hs = [np.eye(3), _scale(5.0)]
    Hs, diverged = _prune(Hs, root=None)
    assert diverged == []
    print("test_prune_handles_missing_root ok")


if __name__ == "__main__":
    test_sane_homography_accepts_identity()
    test_sane_homography_rejects_degenerate()
    test_layout_picks_center_as_root()
    test_layout_is_order_independent()
    test_layout_drops_unreachable_camera()
    test_layout_prefers_strong_edges()
    test_gains_equalise_brightness()
    test_gains_identity_when_balanced()
    test_gains_are_bounded_and_normalised()
    test_gains_no_overlap_returns_ones()
    test_prune_keeps_healthy_layout()
    test_prune_drops_blown_up_camera()
    test_prune_never_drops_root()
    test_prune_tolerates_mild_overflow()
    test_prune_shrinks_canvas_when_way_too_big()
    test_prune_handles_missing_root()
    print("\n全部测试通过。")
