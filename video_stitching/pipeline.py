import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightgluestick.utils import batch_to_np, numpy_image_to_torch
from lightgluestick.two_view_pipeline import TwoViewPipeline

from detector import RFDETRDetector, annotate

MAX_WIDTH = 4000
MAX_HEIGHT = 2000

# 配对质量门限：低于任一条即判定两路不相邻，不建立链路。
# 实测中一条真实重叠的边有 100+ 内点，而不相邻的一对只有约 20 个。
MIN_INLIERS = 40
MIN_INLIER_RATIO = 0.15

# 链式合成后单路允许的最大放大倍数（面积）。逐对检查拦不住多跳累积的发散，
# 实测 FOV 过小时某路会被放大 3 倍以上，把其它相机挤出裁剪后的画布。
MAX_COMPOSED_AREA_RATIO = 2.5
# 跨度超过画布上限多少倍才判定为发散。留出余量是因为轻微超限只会裁掉
# 边缘（可接受），而剪掉整路相机的代价大得多。
SPAN_SLACK = 1.5

# Brown & Lowe 增益补偿的两个方差参数（灰阶 / 增益）
GAIN_SIGMA_N = 10.0
GAIN_SIGMA_G = 0.1
GAIN_LIMITS = (0.5, 2.0)
# 参与增益估计所需的最小重叠面积，按源画面面积取比例而非绝对像素数
MIN_OVERLAP_FRACTION = 5e-4

# RF-DETR 权重由 rfdetr 自行缓存到 ~/.roboflow/models，无需仓库内文件


def _is_sane_homography(H, shape):
    """排除把画面压扁、翻转或拉伸到离谱比例的病态单应。"""
    if H is None or not np.all(np.isfinite(H)):
        return False
    h, w = shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(proj)):
        return False

    # 凸性 + 朝向：源四角在图像坐标系下绕向为正，投影后必须保持同号，
    # 否则说明四边形自交或整体翻面（真实相机间的单应不会改变朝向）
    edges = np.roll(proj, -1, axis=0) - proj
    cross = np.cross(edges, np.roll(edges, -1, axis=0))
    if not np.all(cross > 0):
        return False

    area = 0.5 * abs(np.cross(proj[2] - proj[0], proj[3] - proj[1]))
    ratio = area / float(w * h)
    if not (0.05 < ratio < 20.0):
        return False
    # 单路的跨度不应超过整幅画布上限，否则链路必然已经发散
    if (np.ptp(proj[:, 0]) > MAX_WIDTH * 1.5) or (np.ptp(proj[:, 1]) > MAX_HEIGHT * 1.5):
        return False
    return True


def solve_layout(n, edges):
    """由通过质检的相机对构建最大生成树，返回 (变换字典, 根节点)。

    edges: [(score, i, j, H_ji)]，H_ji 把相机 j 映射到相机 i 的坐标系。
    根节点取生成树的中心（到最远节点跳数最小），这是 `ref = (n-1)//2`
    在任意拓扑下的推广：让形变从中间向两侧各累积一半。
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adjacency = {i: [] for i in range(n)}
    for score, i, j, H in sorted(edges, key=lambda e: e[0], reverse=True):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[ri] = rj
        adjacency[i].append((j, H, True))     # M_j = M_i @ H
        adjacency[j].append((i, H, False))    # M_i = M_j @ inv(H)

    # 取最大连通分量
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if not groups:
        return {}, None
    component = max(groups.values(), key=len)
    if len(component) == 1:
        return {component[0]: np.eye(3)}, component[0]

    def eccentricity(src):
        dist = {src: 0}
        queue = [src]
        while queue:
            node = queue.pop(0)
            for nb, _, _ in adjacency[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return max(dist.values())

    root = min(component, key=eccentricity)

    transforms = {root: np.eye(3)}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nb, H, forward in adjacency[node]:
            if nb in transforms:
                continue
            transforms[nb] = transforms[node] @ (H if forward else np.linalg.inv(H))
            queue.append(nb)
    return transforms, root


def solve_gains(stats, count):
    """Brown & Lowe 增益补偿：解线性方程组得到每路的亮度增益。

    stats[(i, j)] = (重叠像素数, 相机 i 在重叠区的均值, 相机 j 在重叠区的均值)
    """
    A = np.zeros((count, count), np.float64)
    b = np.zeros(count, np.float64)
    inv_n = 1.0 / (GAIN_SIGMA_N ** 2)
    inv_g = 1.0 / (GAIN_SIGMA_G ** 2)

    for (i, j), (npix, mean_i, mean_j) in stats.items():
        A[i, i] += npix * (mean_i * mean_i * inv_n + inv_g)
        A[j, j] += npix * (mean_j * mean_j * inv_n + inv_g)
        A[i, j] -= npix * mean_i * mean_j * inv_n
        A[j, i] -= npix * mean_i * mean_j * inv_n
        b[i] += npix * inv_g
        b[j] += npix * inv_g

    # 没有任何重叠的相机：保持增益 1，避免奇异矩阵
    for i in range(count):
        if A[i, i] == 0:
            A[i, i] = 1.0
            b[i] = 1.0
    try:
        gains = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.ones(count, np.float32)
    if not np.all(np.isfinite(gains)):
        return np.ones(count, np.float32)

    gains = np.clip(gains, *GAIN_LIMITS)
    # 归一到均值 1，避免整幅整体变暗或过曝
    if gains.mean() > 1e-6:
        gains = gains / gains.mean()
    return np.clip(gains, *GAIN_LIMITS).astype(np.float32)


class StitchingPipeline:
    """任意路数（N ≥ 1）相机的水平拼接 + ROI 人数统计。

    相机的相邻关系由全部两两匹配的内点数自动判定，再按最大生成树合成到
    根相机坐标系，因此 config.yaml 中 cameras 的顺序不必是真实空间顺序。
    重叠区另解一组逐相机增益，消除各路曝光不一致造成的亮度台阶。
    """

    def __init__(self, depth_confidence=-1.0, detection=None):
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

        # homographies[i]：把第 i 路映射到参考相机坐标系；无法链接到参考相机时为 None
        self.homographies = None
        # 65 是实测下唯一不触碰画布上限的取值（80 会裁掉顶部）
        self.fov = 65
        self.exposure_compensation = True
        self.layout_info = {}    # 最近一次标定的拓扑：根相机、采用/丢弃的边

        self._cyl_cache = {}     # (源 h, 源 w, fov) -> (map_x, map_y, mask)
        self._cyl_mask_by_out = {}   # (输出 h, 输出 w, fov) -> mask
        self._canvas = None      # 画布尺寸 / 每路 ROI / 融合权重 / 增益，标定后缓存

        # RF-DETR 目标检测；模型在第一帧才真正加载，不拖慢界面启动
        det_conf = dict(detection or {})
        self.count_class = det_conf.pop("count_class", "person")
        self.show_labels = bool(det_conf.pop("show_labels", True))
        self.detector = RFDETRDetector(
            model=det_conf.pop("model", "nano"),
            threshold=det_conf.pop("threshold", 0.5),
            device=det_conf.pop("device", None) or self.device,
            classes=det_conf.pop("classes", None),
            optimize=det_conf.pop("optimize", False),
        )

    # ── 柱面投影 ───────────────────────────────────────────────────────────────

    def _cyl_maps(self, h, w, fov):
        """重映射表随 (尺寸, FOV) 缓存，避免每帧重建 meshgrid。"""
        key = (h, w, fov)
        cached = self._cyl_cache.get(key)
        if cached is not None:
            return cached

        # 输出按柱面**弧长**均匀采样：θ 均匀，输出宽度 = f·fov_rad。
        #
        # 早先的实现直接沿用输入宽度、令 θ = x_c/f，等价于按 tan θ 均匀采样，
        # 于是图像边缘的垂直拉伸是 1/cos(tan(fov/2))——它在
        # fov = 2·atan(π/2) = 115.04° 处发散（100° already 2.7×，110° 已 7.0×），
        # 特征匹配随之全线失败。那不是场景特性而是参数化错误。
        # 正确采样下最大拉伸为 1/cos(fov/2)，只在 180° 才发散。
        fov_rad = np.radians(fov)
        f = w / (2 * np.tan(fov_rad / 2))
        out_w = max(int(round(f * fov_rad)), 8)
        out_h = h

        theta = (np.arange(out_w, dtype=np.float64) - (out_w - 1) / 2.0) / f
        y_c = np.arange(out_h, dtype=np.float64) - (out_h - 1) / 2.0

        map_x = np.broadcast_to(
            (f * np.tan(theta) + w / 2.0).astype(np.float32), (out_h, out_w)
        ).copy()
        map_y = (y_c[:, None] / np.cos(theta)[None, :] + h / 2.0).astype(np.float32)

        # 有效区域掩码：柱面投影后落在原图外的像素为黑边，融合时必须排除
        ones = np.ones((h, w), np.float32)
        mask = cv2.remap(ones, map_x, map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5

        self._cyl_cache[key] = (map_x, map_y, mask)
        # 掩码还需按**输出**尺寸反查（融合权重拿到的是 warp 之后的形状）
        self._cyl_mask_by_out[(out_h, out_w, fov)] = mask
        return self._cyl_cache[key]

    def cylindrical_warp(self, img, fov):
        if fov <= 0:
            return img
        h, w = img.shape[:2]
        map_x, map_y, _ = self._cyl_maps(h, w, fov)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def _valid_mask(self, shape, fov):
        """shape 是**柱面投影之后**的形状，因此按输出尺寸反查掩码。"""
        h, w = shape[:2]
        if fov <= 0:
            return np.ones((h, w), bool)
        mask = self._cyl_mask_by_out.get((h, w, fov))
        return mask if mask is not None else np.ones((h, w), bool)

    # ── 单应估计 ───────────────────────────────────────────────────────────────

    def _match_pair(self, frame_a, frame_b):
        """估计把 frame_b 映射到 frame_a 坐标系的单应。

        返回 (H, 匹配点数, 内点数)；无法估计时 H 为 None。
        内点数是判断两路是否真正相邻的关键依据，必须回传给调用方。
        """
        gray0 = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

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

        n = len(matched_kps0)
        if n < 4:
            return None, n, 0

        H, mask = cv2.findHomography(matched_kps1, matched_kps0, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        return H, n, inliers

    def _accept_pair(self, H, n_matches, inliers, shape):
        """质量门限：拦掉不相邻相机之间的伪匹配。"""
        if H is None or inliers < MIN_INLIERS:
            return False
        if inliers / max(n_matches, 1) < MIN_INLIER_RATIO:
            return False
        return _is_sane_homography(H, shape)

    def calculate_homographies(self, frames):
        """两两匹配全部相机对，按内点数建最大生成树，再合成到根相机。

        不依赖 config 中的相机顺序：真实相邻关系由匹配内点数判定，
        因此顺序填错也能拼对。根相机取生成树中心，形变向两侧各摊一半。
        """
        n = len(frames)
        self._canvas = None

        if n == 1:
            self.homographies = [np.eye(3)]
            self.layout_info = {'root': 0, 'edges': [], 'rejected': [], 'diverged': []}
            return True

        edges, rejected = [], []
        for i in range(n):
            for j in range(i + 1, n):
                H, n_match, inliers = self._match_pair(frames[i], frames[j])
                if self._accept_pair(H, n_match, inliers, frames[j].shape):
                    edges.append((inliers, i, j, H))
                else:
                    rejected.append((i, j, n_match, inliers))

        transforms, root = solve_layout(n, edges)

        Hs = [transforms.get(i) for i in range(n)]
        diverged = self._prune_diverged(frames, Hs, root)

        self.homographies = Hs
        self.layout_info = {
            'root': root,
            # 显式 key：元组末位是 ndarray，无 key 的排序在打平局时会崩在数组比较上
            'edges': [(i, j, s) for s, i, j, _ in
                      sorted(edges, key=lambda e: e[0], reverse=True)],
            'rejected': rejected,
            'diverged': diverged,
        }

        placed = [i for i, H in enumerate(Hs) if H is not None]
        order = ' → '.join(f'cam{i}' for i in placed)
        print(f"[stitch] 参考相机 cam{root}；已定位 {len(placed)}/{n} 路（{order}）")
        for i, j, s in self.layout_info['edges']:
            print(f"[stitch]   采用 cam{i}-cam{j} 内点={s}")
        for i, j, n_match, inliers in rejected:
            print(f"[stitch]   丢弃 cam{i}-cam{j} 匹配={n_match} 内点={inliers}（未过门限）")
        for i, ratio in diverged:
            print(f"[stitch]   丢弃 cam{i}（累积形变 {ratio:.1f}× 过大，"
                  f"保留会撑爆画布并把其它相机挤出画面）")
        failed = [i for i, H in enumerate(Hs) if H is None]
        if failed:
            print(f"[stitch] 无法定位的相机: {failed}")
        return len(failed) == 0

    @classmethod
    def _area_ratio(cls, shape, H):
        """单应把整幅画面放大/缩小的面积倍数。"""
        h, w = shape[:2]
        c = cv2.perspectiveTransform(cls._corners(h, w), H).reshape(-1, 2)
        return float(0.5 * abs(np.cross(c[2] - c[0], c[3] - c[1])) / (w * h))

    def _prune_diverged(self, frames, Hs, root):
        """丢弃链式累积后形变过大的相机。

        `_is_sane_homography` 只逐对检查，而发散是多跳合成才产生的：
        每一跳单独看都正常，合成后却能把某路放大数倍，撑爆画布上限，
        导致裁剪时反而把其它相机整个挤出画面（FOV 设得过小时实测如此）。
        这里以「整体包围盒能否装进画布上限」为准，逐个剔除形变最大的一路。
        """
        diverged = []
        if root is None:
            return diverged

        # 角点与面积倍率只依赖 Hs[i]，剔除相机不会改变幸存者的值，
        # 因此在循环外算一次即可；循环里变化的只有整体跨度。
        corners = {
            i: cv2.perspectiveTransform(
                self._corners(*frames[i].shape[:2]), H).reshape(-1, 2)
            for i, H in enumerate(Hs) if H is not None
        }
        ratios = {i: self._area_ratio(frames[i].shape, Hs[i]) for i in corners}

        # 形变最大的优先考虑剔除；只遍历非根相机，因此永远不会剪到只剩零路
        for worst in sorted((i for i in corners if i != root),
                            key=ratios.get, reverse=True):
            pts = np.concatenate(list(corners.values()), axis=0)
            # 判据分两层：轻微超出画布上限只是裁掉一点边缘，是可接受的旧行为；
            # 只有「单路被放大到失控」或「跨度远超上限」才说明链路真的发散。
            blown_up = ratios[worst] > MAX_COMPOSED_AREA_RATIO
            way_too_big = (np.ptp(pts[:, 0]) > MAX_WIDTH * SPAN_SLACK
                           or np.ptp(pts[:, 1]) > MAX_HEIGHT * SPAN_SLACK)
            if not (blown_up or way_too_big):
                break

            Hs[worst] = None
            del corners[worst]
            diverged.append((worst, ratios[worst]))
        return diverged

    # ── 画布与融合权重 ─────────────────────────────────────────────────────────

    @staticmethod
    def _corners(h, w):
        return np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

    def _feather_weight(self, shape, fov):
        """中心高、边缘低的羽化权重，用于多路重叠区的加权融合。"""
        h, w = shape[:2]
        ramp_x = 1.0 - np.abs(np.linspace(-1.0, 1.0, w, dtype=np.float32))
        ramp_y = 1.0 - np.abs(np.linspace(-1.0, 1.0, h, dtype=np.float32))
        # 保底 0.01，避免只有单路覆盖的边缘像素权重为 0 而变黑
        wt = 0.01 + 0.99 * np.outer(ramp_y, ramp_x).astype(np.float32)
        return wt * self._valid_mask(shape, fov).astype(np.float32)

    def _build_canvas(self, warps):
        """按当前单应计算画布尺寸、每路的绘制区域和归一化融合权重。"""
        valid = [i for i, H in enumerate(self.homographies) if H is not None]

        pts = np.concatenate(
            [cv2.perspectiveTransform(self._corners(*warps[i].shape[:2]),
                                      self.homographies[i]) for i in valid],
            axis=0)
        xmin, ymin = np.int32(pts.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(pts.max(axis=0).ravel() + 0.5)

        # 极端形变时限制画布尺寸，避免爆内存
        if (xmax - xmin) > MAX_WIDTH:
            if xmin < 0: xmin = max(xmin, xmax - MAX_WIDTH)
            else:        xmax = min(xmax, xmin + MAX_WIDTH)
        if (ymax - ymin) > MAX_HEIGHT:
            if ymin < 0: ymin = max(ymin, ymax - MAX_HEIGHT)
            else:        ymax = min(ymax, ymin + MAX_HEIGHT)

        cw, ch = int(xmax - xmin), int(ymax - ymin)
        if cw <= 0 or ch <= 0:
            return None

        Ht = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)

        # 每路只在自己的包围盒内计算，避免整幅画布的无谓运算
        tiles, total = [], np.zeros((ch, cw), np.float32)
        for i in valid:
            h, w = warps[i].shape[:2]
            F = Ht @ self.homographies[i]
            c = cv2.perspectiveTransform(self._corners(h, w), F).reshape(-1, 2)
            x0 = max(int(np.floor(c[:, 0].min())), 0)
            y0 = max(int(np.floor(c[:, 1].min())), 0)
            x1 = min(int(np.ceil(c[:, 0].max())) + 1, cw)
            y1 = min(int(np.ceil(c[:, 1].max())) + 1, ch)
            if x1 <= x0 or y1 <= y0:
                continue

            offset = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
            M = offset @ F
            wt = cv2.warpPerspective(self._feather_weight(warps[i].shape, self.fov),
                                     M, (x1 - x0, y1 - y0),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            total[y0:y1, x0:x1] += wt
            tiles.append({'index': i, 'roi': (x0, y0, x1, y1), 'M': M, 'weight': wt})

        if not tiles:
            return None

        gains = self._solve_exposure_gains(warps, tiles, len(warps))

        # 归一化到 Σw = 1，融合结果自然落在 [0, 255]
        safe = np.maximum(total, 1e-6)
        for t in tiles:
            x0, y0, x1, y1 = t['roi']
            wn = t['weight'] / safe[y0:y1, x0:x1]
            t['weight'] = np.repeat(wn[:, :, None], 3, axis=2)

        return {'size': (cw, ch), 'tiles': tiles, 'gains': gains,
                'shapes': [w.shape[:2] for w in warps]}

    def _solve_exposure_gains(self, warps, tiles, count):
        """在重叠区统计各路亮度，解出逐相机增益，消除接缝处的亮度台阶。

        实测三路场景中亮度台阶占接缝误差的一半以上，补偿后可显著削弱。
        """
        gains = np.ones(count, np.float32)
        if not self.exposure_compensation or len(tiles) < 2:
            return gains

        # 重叠区太小则样本不足以估亮度均值。按源画面面积取相对阈值，
        # 换分辨率时含义不变（绝对像素数会随分辨率漂移）。
        src_h, src_w = warps[tiles[0]['index']].shape[:2]
        min_overlap = max(int(MIN_OVERLAP_FRACTION * src_h * src_w), 64)

        # 先转灰度再 warp：两者都是线性运算，结果等价，但只需搬运 1 个通道而非 3 个
        grays = []
        for t in tiles:
            x0, y0, x1, y1 = t['roi']
            grays.append(cv2.warpPerspective(
                cv2.cvtColor(warps[t['index']], cv2.COLOR_BGR2GRAY), t['M'],
                (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT).astype(np.float32))

        def crop(k, box):
            """把画布坐标的矩形换算成第 k 块瓦片内的切片。"""
            ix0, iy0, ix1, iy1 = box
            x0, y0 = tiles[k]['roi'][0], tiles[k]['roi'][1]
            return (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))

        stats = {}
        for a in range(len(tiles)):
            for b in range(a + 1, len(tiles)):
                ax0, ay0, ax1, ay1 = tiles[a]['roi']
                bx0, by0, bx1, by1 = tiles[b]['roi']
                # 两个包围盒的交集，再在交集内取双方都有效的像素
                box = (max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                sa, sb = crop(a, box), crop(b, box)
                # 此时 weight 还是 2D（归一化成 3 通道发生在本函数返回之后）
                ov = (tiles[a]['weight'][sa] > 1e-6) & (tiles[b]['weight'][sb] > 1e-6)
                npix = int(ov.sum())
                if npix < min_overlap:
                    continue
                ga, gb = grays[a][sa][ov], grays[b][sb][ov]
                stats[(tiles[a]['index'], tiles[b]['index'])] = (
                    npix, float(ga.mean()), float(gb.mean()))

        if not stats:
            return gains

        solved = solve_gains(stats, count)
        for (i, j), (npix, mi, mj) in stats.items():
            print(f"[stitch]   重叠 cam{i}-cam{j}: {npix} px, 亮度 {mi:.1f} vs {mj:.1f}")
        print("[stitch]   曝光增益: " +
              ", ".join(f"cam{i}={solved[i]:.3f}" for i in range(count)))
        return solved

    # ── 拼接 ───────────────────────────────────────────────────────────────────

    def stitch_frames(self, frames):
        """frames: 按相机顺序排列的 BGR 帧列表（也接受单帧）。"""
        if isinstance(frames, np.ndarray):
            frames = [frames]
        frames = [f for f in frames if f is not None]
        if not frames:
            return None

        warps = [self.cylindrical_warp(f, self.fov) for f in frames]
        if len(warps) == 1:
            return warps[0]

        if self.homographies is None or len(self.homographies) != len(warps):
            self.calculate_homographies(warps)
            if sum(H is not None for H in self.homographies) < 2:
                self.homographies = None    # 一路都没接上，下一帧重试
                return warps[(len(warps) - 1) // 2]

        # 拼接失败时回退显示根相机（生成树中心），而非固定的中间序号
        ref = self.layout_info.get('root')
        if ref is None or not (0 <= ref < len(warps)):
            ref = (len(warps) - 1) // 2

        try:
            shapes = [w.shape[:2] for w in warps]
            if self._canvas is None or self._canvas['shapes'] != shapes:
                self._canvas = self._build_canvas(warps)
            if self._canvas is None:
                return warps[ref]

            cw, ch = self._canvas['size']
            gains = self._canvas.get('gains')
            acc = np.zeros((ch, cw, 3), np.float32)
            for t in self._canvas['tiles']:
                x0, y0, x1, y1 = t['roi']
                piece = cv2.warpPerspective(warps[t['index']], t['M'],
                                            (x1 - x0, y1 - y0),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
                contrib = piece.astype(np.float32) * t['weight']
                if gains is not None:
                    contrib *= float(gains[t['index']])
                acc[y0:y1, x0:x1] += contrib

            return np.clip(acc, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Homography error: {e}")
            self._canvas = None
            return warps[ref]

    # ── 检测与计数 ─────────────────────────────────────────────────────────────

    def detect_and_count(self, image, polygon):
        """在拼接图上做目标检测并标注。

        返回 (标注后的图, ROI 内 count_class 的数量)。与旧接口保持一致，
        区别是现在会把所有检出的类别都画出来，而不只是人。
        """
        if image is None:
            return None, 0
        try:
            boxes, names, conf = self.detector.detect(image)
        except Exception as exc:
            print(f"[detect] 推理失败：{exc}")
            return image.copy(), 0

        return annotate(image, boxes, names, conf, polygon=polygon,
                        count_class=self.count_class,
                        show_labels=self.show_labels)
