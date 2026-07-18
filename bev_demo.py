"""
Bird's-Eye View (BEV) fusion demo for two opposite-direction cameras.

Pipeline:
  1. LightGlueStick matches features in the overlapping region
  2. Each camera is projected to a top-down view via IPM on user-supplied
     (or interactively clicked) floor quadrilaterals
  3. Matched keypoints are projected through the IPM into BEV space;
     when that fails (e.g. matches on walls), a homography-composed or
     depth-heuristic fallback estimates both dx and dy
  4. Distance-weighted blending on a proper geometric mask (not pixel>0)

Usage:
    # Interactive (recommended) — click 4 floor corners per camera
    python bev_demo.py -img1 weixin.png -img2 weixin1.png --interactive

    # Explicit floor quads via CLI (TL_x,TL_y TR_x,TR_y BR_x,BR_y BL_x,BL_y)
    python bev_demo.py -img1 weixin.png -img2 weixin1.png \\
        --floor1 "480,620 1100,620 1750,1020 150,1020" \\
        --floor2 "820,620 1440,620 1770,1020 170,1020"

    # Auto heuristic (rough — warns at runtime)
    python bev_demo.py -img1 weixin.png -img2 weixin1.png
"""

import argparse
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from lightgluestick.utils import batch_to_np, numpy_image_to_torch
from lightgluestick.two_view_pipeline import TwoViewPipeline
from lightgluestick.viz2d import (
    plot_images, plot_lines, plot_color_line_matches,
    plot_keypoints, plot_matches,
)


# ======================================================================
# Helpers
# ======================================================================

def _distance_blend(img_a, mask_a, img_b, mask_b):
    """Alpha-blend two images using distance-to-edge weighting on binary masks."""
    m_a = (mask_a > 0).astype(np.uint8)
    m_b = (mask_b > 0).astype(np.uint8)
    d_a = cv2.distanceTransform(m_a, cv2.DIST_L2, 5)
    d_b = cv2.distanceTransform(m_b, cv2.DIST_L2, 5)
    total = d_a + d_b
    total[total == 0] = 1
    w_a = (d_a / total)[..., None]
    w_b = (d_b / total)[..., None]
    out = img_a.astype(np.float32) * w_a + img_b.astype(np.float32) * w_b
    only_a = (m_a > 0) & (m_b == 0)
    only_b = (m_b > 0) & (m_a == 0)
    out[only_a] = img_a[only_a].astype(np.float32)
    out[only_b] = img_b[only_b].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def select_points_interactive(img, window_name, n_points=4):
    """Let user click N points on a resized window; returns original-scale coords."""
    points = []
    scale = min(1.0, 1200.0 / img.shape[1])
    disp = cv2.resize(img, None, fx=scale, fy=scale)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= n_points:
            return
        rx, ry = int(x / scale), int(y / scale)
        points.append([rx, ry])
        cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
        if len(points) > 1:
            px, py = points[-2]
            cv2.line(disp, (int(px * scale), int(py * scale)), (x, y), (0, 255, 0), 2)
        if len(points) == n_points:
            fx_, fy_ = points[0]
            cv2.line(disp, (x, y), (int(fx_ * scale), int(fy_ * scale)), (0, 255, 0), 2)
        cv2.imshow(window_name, disp)

    cv2.imshow(window_name, disp)
    cv2.setMouseCallback(window_name, on_mouse)
    print(f"[Interactive] Click {n_points} floor corners on '{window_name}' "
          f"(TL→TR→BR→BL), then press any key.")
    while len(points) < n_points:
        cv2.waitKey(50)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return np.float32(points)


def draw_quad(img, pts, color=(0, 255, 0), thickness=2):
    vis = img.copy()
    pts_i = np.int32(pts)
    cv2.polylines(vis, [pts_i], True, color, thickness)
    for i, p in enumerate(pts_i):
        cv2.circle(vis, tuple(p), 6, color, -1)
        cv2.putText(vis, f"P{i}", tuple(p + [5, -8]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def parse_floor_arg(s):
    """Parse "x0,y0 x1,y1 x2,y2 x3,y3" into (4,2) float32 array."""
    pts = []
    for tok in s.strip().split():
        x, y = tok.split(',')
        pts.append([float(x), float(y)])
    if len(pts) != 4:
        raise ValueError(f"Need exactly 4 points, got {len(pts)}")
    return np.float32(pts)


def _proj_homogeneous(M, pts):
    """Apply 3×3 matrix M to Nx2 points, return Nx2 de-homogenised."""
    h = np.hstack([pts, np.ones((len(pts), 1))]).T   # 3×N
    out = (M @ h).T                                    # N×3
    return out[:, :2] / out[:, 2:3]


# ======================================================================
# Core class
# ======================================================================

class BEVFusion:
    def __init__(self, depth_confidence=-1.0, max_pts=2048, max_lines=250):
        self.device = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Device: {self.device}")

        conf = {
            'name': 'two_view_pipeline',
            'use_lines': True,
            'extractor': {
                "name": "wireframe",
                "point_extractor": {
                    "name": "superpoint", "trainable": False,
                    "dense_outputs": True,
                    "max_num_keypoints": max_pts, "force_num_keypoints": False,
                },
                "line_extractor": {
                    "name": "lsd", "trainable": False,
                    "max_num_lines": max_lines, "force_num_lines": False,
                    "min_length": 15,
                },
                "wireframe_params": {
                    "merge_points": True, "merge_line_endpoints": True,
                    "nms_radius": 3,
                },
            },
            'matcher': {
                'name': 'lightgluestick',
                'depth_confidence': depth_confidence, 'trainable': False,
            },
            'ground_truth': {'from_pose_depth': False},
        }
        self.pipeline_model = TwoViewPipeline(conf).to(self.device).eval()

    # ------------------------------------------------------------------
    # Feature matching
    # ------------------------------------------------------------------
    def match_features(self, img1, img2):
        gray0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        t0 = numpy_image_to_torch(gray0).to(self.device)[None]
        t1 = numpy_image_to_torch(gray1).to(self.device)[None]

        with torch.no_grad():
            pred = self.pipeline_model({'view0': {"image": t0}, 'view1': {"image": t1}})
        pred = batch_to_np(pred)

        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0, lm0 = pred["matches0"], pred["line_matches0"]
        lines0, lines1 = pred["lines0"], pred["lines1"]

        valid = m0 != -1
        mkp0, mkp1 = kp0[valid], kp1[m0[valid]]
        valid_l = lm0 != -1
        ml0, ml1 = lines0[valid_l], lines1[lm0[valid_l]]

        print(f"Matches: {len(mkp0)} points, {len(ml0)} lines")
        return dict(kp0=kp0, kp1=kp1, mkp0=mkp0, mkp1=mkp1,
                    lines0=lines0, lines1=lines1, ml0=ml0, ml1=ml1)

    # ------------------------------------------------------------------
    # Floor-plane estimation (heuristic — users should prefer explicit)
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_floor_quad(img):
        """
        ROUGH heuristic; will include non-floor content.
        Prefer --interactive or --floor1/--floor2 for real use.
        """
        h, w = img.shape[:2]
        return np.float32([
            [w * 0.30, h * 0.50],
            [w * 0.70, h * 0.50],
            [w * 0.95, h * 0.95],
            [w * 0.05, h * 0.95],
        ])

    # ------------------------------------------------------------------
    # Per-camera IPM
    # ------------------------------------------------------------------
    @staticmethod
    def ipm(img, src_quad, dst_size):
        """
        Inverse Perspective Mapping: warp src_quad → full dst_size rectangle.
        Returns (bev_image, geometric_mask, perspective_matrix).
        The geometric mask is produced by warping a solid-white image through
        the same transform, so it is independent of pixel intensity.
        """
        dst = np.float32([[0, 0], [dst_size[0], 0],
                          [dst_size[0], dst_size[1]], [0, dst_size[1]]])
        M = cv2.getPerspectiveTransform(np.float32(src_quad), dst)
        bev = cv2.warpPerspective(img, M, dst_size,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpPerspective(
            np.ones(img.shape[:2], dtype=np.uint8) * 255,
            M, dst_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT)
        return bev, mask, M

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    def _compute_alignment(self, mkp0, mkp1, M1, M2, R180,
                            bev_w, bev_h, img1_shape, img2_shape,
                            floor1, floor2):
        """
        Compute (dx, dy) offset for placing flipped-BEV2 below BEV1.

        Tries three strategies in order:
          A) Project matched keypoints through IPM into BEV space and
             compute offset from the transformed pairs that land inside.
          B) Compose image-space homography with both IPMs to derive a
             BEV-to-BEV transform, then read off the translation.
          C) Depth-fraction heuristic from image-space Y positions — only
             gives a rough overlap estimate; dx derived from X positions.
        """
        if len(mkp0) < 4:
            print("Alignment: < 4 matches → default 20 % overlap, dx=0")
            return np.array([0.0, -bev_h * 0.20])

        # --- Strategy A: direct BEV-space keypoints ---
        bp0 = _proj_homogeneous(M1, mkp0)                  # Nx2 in bev1
        bp1_raw = _proj_homogeneous(M2, mkp1)              # Nx2 in bev2
        bp1_flip = _proj_homogeneous(R180, bp1_raw)         # after 180° flip

        in0 = ((bp0[:, 0] >= 0) & (bp0[:, 0] < bev_w) &
               (bp0[:, 1] >= 0) & (bp0[:, 1] < bev_h))
        in1 = ((bp1_flip[:, 0] >= 0) & (bp1_flip[:, 0] < bev_w) &
               (bp1_flip[:, 1] >= 0) & (bp1_flip[:, 1] < bev_h))
        both = in0 & in1
        n_valid = int(both.sum())
        print(f"Strategy A (BEV keypoints): {n_valid}/{len(mkp0)} inside both BEVs")

        if n_valid >= 3:
            # bev2-flipped is placed at y_origin = bev_h + dy relative to bev1
            diff = bp0[both] - bp1_flip[both]
            diff[:, 1] -= bev_h          # expected baseline offset
            dx = float(np.median(diff[:, 0]))
            dy = float(np.median(diff[:, 1]))
            dy = np.clip(dy, -bev_h * 0.9, bev_h * 0.3)
            dx = np.clip(dx, -bev_w * 0.5, bev_w * 0.5)
            print(f"  → dx={dx:.0f}, dy={dy:.0f}")
            return np.array([dx, dy])

        # --- Strategy B: homography composed with IPMs ---
        # NOTE: only valid when matched features lie on (or near) the floor
        # plane used by the IPM.  When matches are on walls/ceiling, the
        # composition T = M1 · H · M2⁻¹ mixes two different planes and
        # produces unreliable results — detected by a positive dy (gap).
        H, mask_h = cv2.findHomography(mkp1, mkp0, cv2.RANSAC, 5.0)
        if H is not None and mask_h is not None:
            inliers = int(mask_h.ravel().sum())
            print(f"Strategy B (homography→BEV): {inliers} inliers")
            T_bev = M1 @ H @ np.linalg.inv(M2)
            T_full = T_bev @ np.linalg.inv(R180)

            ref_pts = np.float32([
                [bev_w * 0.25, bev_h * 0.25],
                [bev_w * 0.75, bev_h * 0.25],
                [bev_w * 0.50, bev_h * 0.50],
                [bev_w * 0.25, bev_h * 0.75],
                [bev_w * 0.75, bev_h * 0.75],
            ])
            mapped = _proj_homogeneous(T_full, ref_pts)
            expected = ref_pts + np.array([0, bev_h])
            diffs = mapped - expected
            dx = float(np.median(diffs[:, 0]))
            dy = float(np.median(diffs[:, 1]))

            if dy < 0:
                dx = np.clip(dx, -bev_w * 0.5, bev_w * 0.5)
                dy = np.clip(dy, -bev_h * 0.9, -bev_h * 0.05)
                print(f"  → dx={dx:.0f}, dy={dy:.0f}")
                return np.array([dx, dy])
            else:
                print(f"  → dy={dy:.0f} (positive = gap) — matches likely "
                      f"off floor plane, falling through to strategy C")

        # --- Strategy C: depth-fraction heuristic ---
        print("Strategy C (depth-fraction heuristic from image-space Y)")
        h1 = img1_shape[0]
        h2 = img2_shape[0]
        w1 = img1_shape[1]
        w2 = img2_shape[1]
        floor_bot_y1 = float(floor1[2, 1])
        floor_bot_y2 = float(floor2[2, 1])

        med_y0 = float(np.median(mkp0[:, 1]))
        med_y1 = float(np.median(mkp1[:, 1]))
        depth0 = np.clip(1.0 - med_y0 / floor_bot_y1, 0.05, 0.95)
        depth1 = np.clip(1.0 - med_y1 / floor_bot_y2, 0.05, 0.95)

        bev_y0 = bev_h * (1.0 - depth0)
        bev_y1f = bev_h * depth1
        dy = bev_y0 - bev_y1f - bev_h
        dy = np.clip(dy, -bev_h * 0.9, -bev_h * 0.05)

        # dx from X positions: map median X through floor-quad proportions
        med_x0 = float(np.median(mkp0[:, 0]))
        med_x1 = float(np.median(mkp1[:, 0]))
        frac_x0 = np.clip((med_x0 - floor1[3, 0]) / (floor1[2, 0] - floor1[3, 0]), 0, 1)
        frac_x1 = np.clip((med_x1 - floor2[3, 0]) / (floor2[2, 0] - floor2[3, 0]), 0, 1)
        bev_x0 = frac_x0 * bev_w
        bev_x1_flip = (1.0 - frac_x1) * bev_w
        dx = bev_x0 - bev_x1_flip
        dx = np.clip(dx, -bev_w * 0.3, bev_w * 0.3)

        print(f"  depth: cam1={depth0:.2f}, cam2={depth1:.2f}")
        print(f"  → dx={dx:.0f}, dy={dy:.0f}")
        return np.array([dx, dy])

    # ------------------------------------------------------------------
    # BEV fusion
    # ------------------------------------------------------------------
    def fuse_bev_with_matches(self, img1, img2, floor1, floor2,
                               mkp0, mkp1, bev_w=800, bev_h=600):
        """
        1. IPM each camera to bev_w × bev_h with a proper geometric mask
        2. Flip cam2 BEV 180° (opposite direction)
        3. Compute (dx, dy) alignment via strategies A → B → C
        4. Place both on a canvas, blend using geometric masks
        """
        bev1, gmask1, M1 = self.ipm(img1, floor1, (bev_w, bev_h))
        bev2, gmask2, M2 = self.ipm(img2, floor2, (bev_w, bev_h))

        R180 = np.float64([[-1, 0, bev_w], [0, -1, bev_h], [0, 0, 1]])

        offset = self._compute_alignment(
            mkp0, mkp1, M1, M2, R180, bev_w, bev_h,
            img1_shape=img1.shape, img2_shape=img2.shape,
            floor1=floor1, floor2=floor2)

        bev2_flip = cv2.rotate(bev2, cv2.ROTATE_180)
        gmask2_flip = cv2.rotate(gmask2, cv2.ROTATE_180)

        # --- Place on canvas and blend ---
        pad = 20
        ox, oy = int(round(offset[0])), int(round(offset[1]))
        x1, y1 = pad, pad
        x2, y2 = pad + ox, pad + bev_h + oy

        all_x = [x1, x1 + bev_w, x2, x2 + bev_w]
        all_y = [y1, y1 + bev_h, y2, y2 + bev_h]
        cw = max(all_x) - min(all_x) + 2 * pad
        ch = max(all_y) - min(all_y) + 2 * pad
        sx = -min(all_x) + pad
        sy = -min(all_y) + pad

        canvas1 = np.zeros((ch, cw, 3), np.uint8)
        canvas2 = np.zeros_like(canvas1)
        cmask1 = np.zeros((ch, cw), np.float32)
        cmask2 = np.zeros_like(cmask1)

        r1y, r1x = y1 + sy, x1 + sx
        r2y, r2x = y2 + sy, x2 + sx
        canvas1[r1y:r1y+bev_h, r1x:r1x+bev_w] = bev1
        cmask1[r1y:r1y+bev_h, r1x:r1x+bev_w] = gmask1.astype(np.float32) / 255.0
        canvas2[r2y:r2y+bev_h, r2x:r2x+bev_w] = bev2_flip
        cmask2[r2y:r2y+bev_h, r2x:r2x+bev_w] = gmask2_flip.astype(np.float32) / 255.0

        fused = _distance_blend(canvas1, cmask1, canvas2, cmask2)

        # Crop to the valid mask region
        valid = (cmask1 > 0) | (cmask2 > 0)
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            fused = fused[rmin:rmax+1, cmin:cmax+1]

        return bev1, bev2_flip, fused

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def visualize_matches(self, img1, img2, mr, prefix="bev"):
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        plot_images([rgb1, rgb2], ['Cam1 - keypoints', 'Cam2 - keypoints'],
                    dpi=150, pad=2.0)
        plot_keypoints([mr['kp0'], mr['kp1']], colors='c')
        plt.savefig(f'{prefix}_detected_points.png', bbox_inches='tight')
        plt.close()

        plot_images([rgb1, rgb2], ['Cam1 - lines', 'Cam2 - lines'],
                    dpi=150, pad=2.0)
        plot_lines([mr['lines0'], mr['lines1']], ps=4, lw=2)
        plt.savefig(f'{prefix}_detected_lines.png', bbox_inches='tight')
        plt.close()

        if len(mr['mkp0']) > 0:
            plot_images([rgb1, rgb2],
                        ['Cam1 - point matches', 'Cam2 - point matches'],
                        dpi=150, pad=2.0)
            plot_matches(mr['mkp0'], mr['mkp1'], 'green', lw=1, ps=2)
            plt.savefig(f'{prefix}_point_matches.png', bbox_inches='tight')
            plt.close()

        if len(mr['ml0']) > 0:
            plot_images([rgb1, rgb2],
                        ['Cam1 - line matches', 'Cam2 - line matches'],
                        dpi=150, pad=2.0)
            plot_color_line_matches([mr['ml0'], mr['ml1']], lw=2)
            plt.savefig(f'{prefix}_line_matches.png', bbox_inches='tight')
            plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Bird's-Eye View fusion of two opposite-direction cameras "
                    "using LightGlueStick feature matching")
    ap.add_argument('-img1', required=True, help='Camera 1 image (forward)')
    ap.add_argument('-img2', required=True, help='Camera 2 image (backward)')
    ap.add_argument('--depth_confidence', type=float, default=-1.0)
    ap.add_argument('--bev-width', type=int, default=800)
    ap.add_argument('--bev-height', type=int, default=600)
    ap.add_argument('--interactive', action='store_true',
                    help='Click 4 floor corners per camera for IPM')
    ap.add_argument('--floor1', type=str, default=None,
                    help='Floor quad for cam1: "x0,y0 x1,y1 x2,y2 x3,y3" (TL TR BR BL)')
    ap.add_argument('--floor2', type=str, default=None,
                    help='Floor quad for cam2: same format')
    ap.add_argument('--output-prefix', type=str, default='bev')
    args = ap.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Cannot read: {args.img1} / {args.img2}")

    print(f"Cam1: {img1.shape[1]}x{img1.shape[0]}  "
          f"Cam2: {img2.shape[1]}x{img2.shape[0]}")

    fuser = BEVFusion(depth_confidence=args.depth_confidence)
    pfx = args.output_prefix

    # ---- Step 1: Feature matching ----
    print("\n--- Step 1: LightGlueStick feature matching ---")
    mr = fuser.match_features(img1, img2)
    fuser.visualize_matches(img1, img2, mr, prefix=pfx)
    print(f"Saved match visualizations ({pfx}_*.png)")

    # ---- Step 2: Floor estimation ----
    print("\n--- Step 2: Floor-plane definition ---")
    if args.floor1 and args.floor2:
        floor1 = parse_floor_arg(args.floor1)
        floor2 = parse_floor_arg(args.floor2)
        print("Using explicit floor quads from CLI")
    elif args.interactive:
        floor1 = select_points_interactive(img1, "Cam1 - click 4 floor corners")
        floor2 = select_points_interactive(img2, "Cam2 - click 4 floor corners")
    else:
        floor1 = fuser.estimate_floor_quad(img1)
        floor2 = fuser.estimate_floor_quad(img2)
        print("WARNING: using rough heuristic floor quad — result will include "
              "non-floor content.  Use --interactive or --floor1/--floor2 for "
              "accurate BEV.")
    print(f"Floor quad cam1: {floor1.tolist()}")
    print(f"Floor quad cam2: {floor2.tolist()}")

    ann1 = draw_quad(img1, floor1, (0, 255, 0), 3)
    ann2 = draw_quad(img2, floor2, (0, 255, 0), 3)
    cv2.imwrite(f'{pfx}_floor_cam1.png', ann1)
    cv2.imwrite(f'{pfx}_floor_cam2.png', ann2)

    # ---- Step 3: BEV fusion ----
    print("\n--- Step 3: IPM → BEV + alignment + fusion ---")
    bev1, bev2_flip, fused = fuser.fuse_bev_with_matches(
        img1, img2, floor1, floor2,
        mr['mkp0'], mr['mkp1'],
        bev_w=args.bev_width, bev_h=args.bev_height)

    cv2.imwrite(f'{pfx}_bev_cam1.png', bev1)
    cv2.imwrite(f'{pfx}_bev_cam2.png', bev2_flip)
    cv2.imwrite(f'{pfx}_bird_eye_view.png', fused)
    print(f"Saved: {pfx}_bev_cam1.png, {pfx}_bev_cam2.png")
    print(f"Saved: {pfx}_bird_eye_view.png  ({fused.shape[1]}x{fused.shape[0]})")

    # ---- Overview figure ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes[0, 0].imshow(cv2.cvtColor(ann1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Cam1 + floor region'); axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(ann2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Cam2 + floor region'); axes[0, 1].axis('off')

    if len(mr['mkp0']) > 0:
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        h1, w1 = rgb1.shape[:2]
        h2, w2 = rgb2.shape[:2]
        max_h = max(h1, h2)
        pair = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        pair[:h1, :w1] = rgb1
        pair[:h2, w1:] = rgb2
        for p0, p1 in zip(mr['mkp0'][:50], mr['mkp1'][:50]):
            pt0 = (int(p0[0]), int(p0[1]))
            pt1 = (int(p1[0]) + w1, int(p1[1]))
            cv2.line(pair, pt0, pt1, (0, 255, 0), 1)
        axes[0, 2].imshow(pair)
        axes[0, 2].set_title(f'Matches ({len(mr["mkp0"])} pts)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(bev1, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('BEV Cam1'); axes[1, 0].axis('off')
    axes[1, 1].imshow(cv2.cvtColor(bev2_flip, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('BEV Cam2 (flipped)'); axes[1, 1].axis('off')
    axes[1, 2].imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Fused Bird's-Eye View"); axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{pfx}_result_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pfx}_result_overview.png")
    print("\nDone!")


if __name__ == '__main__':
    main()
