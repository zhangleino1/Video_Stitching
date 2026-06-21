import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAMERA_COUNT = 4
DEFAULT_SCALE = 100.0
DEFAULT_PADDING = 20


def _matrix_to_list(matrix):
    return matrix.tolist() if matrix is not None else None


def _matrix_from_list(value):
    return np.array(value, dtype=np.float64) if value is not None else None


@dataclass
class CameraCalibration:
    camera_id: str
    name: str
    url: str = ""
    enabled: bool = True
    image_points: list = field(default_factory=list)
    local_world_points: list = field(default_factory=list)
    local_bev_homography: np.ndarray = None
    bev_to_mosaic_transform: np.ndarray = None
    last_registration_quality: dict = field(default_factory=dict)
    bev_width: int = 800
    bev_height: int = 800
    world_offset_px: tuple = (0.0, 0.0)

    @property
    def is_calibrated(self):
        return self.local_bev_homography is not None

    def clear_points(self):
        self.image_points = []
        self.local_world_points = []
        self.local_bev_homography = None
        self.bev_to_mosaic_transform = None
        self.last_registration_quality = {}

    def compute_local_bev(self, scale, padding=DEFAULT_PADDING):
        if len(self.image_points) < 4 or len(self.local_world_points) < 4:
            return False, "至少需要 4 个标定点。"
        if len(self.image_points) != len(self.local_world_points):
            return False, "图像点和物理坐标点数量必须一致。"

        img_pts = np.array(self.image_points, dtype=np.float32)
        world_pts = np.array(self.local_world_points, dtype=np.float32)
        if np.linalg.matrix_rank(world_pts - world_pts.mean(axis=0)) < 2:
            return False, "物理坐标点不能共线。"
        if np.linalg.matrix_rank(img_pts - img_pts.mean(axis=0)) < 2:
            return False, "图像标定点不能共线。"

        bev_pts = world_pts * float(scale)
        min_xy = bev_pts.min(axis=0)
        max_xy = bev_pts.max(axis=0)
        offset = np.array([padding, padding], dtype=np.float32) - min_xy
        bev_pts = bev_pts + offset

        width = int(np.ceil((max_xy[0] - min_xy[0]) + 2 * padding))
        height = int(np.ceil((max_xy[1] - min_xy[1]) + 2 * padding))
        width = max(width, 100)
        height = max(height, 100)

        H, status = cv2.findHomography(img_pts, bev_pts, cv2.RANSAC, 4.0)
        if H is None:
            return False, "Homography 计算失败，请检查标定点顺序和坐标。"

        self.local_bev_homography = H
        self.bev_width = width
        self.bev_height = height
        self.world_offset_px = (float(offset[0]), float(offset[1]))
        self.bev_to_mosaic_transform = None
        self.last_registration_quality = {}
        return True, f"BEV 标定完成：{width}x{height}"

    def warp_to_bev(self, frame):
        if frame is None or self.local_bev_homography is None:
            return None, None

        src_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(np.array(self.image_points, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(src_mask, [hull], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=src_mask)

        size = (int(self.bev_width), int(self.bev_height))
        bev = cv2.warpPerspective(
            masked_frame,
            self.local_bev_homography,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpPerspective(
            src_mask,
            self.local_bev_homography,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        bev = cv2.bitwise_and(bev, bev, mask=mask)
        return bev, mask

    def to_config(self):
        return {
            "id": self.camera_id,
            "name": self.name,
            "url": self.url,
            "enabled": self.enabled,
            "image_points": self.image_points,
            "local_world_points": self.local_world_points,
            "local_bev_homography": _matrix_to_list(self.local_bev_homography),
            "bev_to_mosaic_transform": _matrix_to_list(self.bev_to_mosaic_transform),
            "last_registration_quality": self.last_registration_quality,
            "bev_width": self.bev_width,
            "bev_height": self.bev_height,
            "world_offset_px": list(self.world_offset_px),
        }

    @classmethod
    def from_config(cls, data, index):
        cam = cls(
            camera_id=data.get("id", f"cam{index + 1}"),
            name=data.get("name", f"Camera {index + 1}"),
            url=data.get("url", ""),
            enabled=bool(data.get("enabled", True)),
            image_points=[list(p) for p in data.get("image_points", [])],
            local_world_points=[list(p) for p in data.get("local_world_points", [])],
            local_bev_homography=_matrix_from_list(data.get("local_bev_homography")),
            bev_to_mosaic_transform=_matrix_from_list(data.get("bev_to_mosaic_transform")),
            last_registration_quality=dict(data.get("last_registration_quality", {})),
            bev_width=int(data.get("bev_width", 800)),
            bev_height=int(data.get("bev_height", 800)),
            world_offset_px=tuple(data.get("world_offset_px", [0.0, 0.0])),
        )
        return cam


class CalibrationProject:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else Path(__file__).with_name("config.json")
        self.scale = DEFAULT_SCALE
        self.padding = DEFAULT_PADDING
        self.cameras = []
        self.load()

    def load(self):
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.scale = float(data.get("scale", DEFAULT_SCALE))
            self.padding = int(data.get("canvas_padding_px", DEFAULT_PADDING))
            self.cameras = [
                CameraCalibration.from_config(cam, idx)
                for idx, cam in enumerate(data.get("cameras", []))
            ]
        if not self.cameras:
            self.cameras = [
                CameraCalibration(f"cam{i + 1}", f"Camera {i + 1}")
                for i in range(DEFAULT_CAMERA_COUNT)
            ]
        self.cameras = self.cameras[:DEFAULT_CAMERA_COUNT]
        while len(self.cameras) < DEFAULT_CAMERA_COUNT:
            idx = len(self.cameras)
            self.cameras.append(CameraCalibration(f"cam{idx + 1}", f"Camera {idx + 1}"))

    def save(self):
        data = {
            "scale": self.scale,
            "canvas_padding_px": self.padding,
            "cameras": [cam.to_config() for cam in self.cameras],
        }
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def enabled_calibrated_cameras(self):
        return [cam for cam in self.cameras if cam.enabled and cam.is_calibrated]
