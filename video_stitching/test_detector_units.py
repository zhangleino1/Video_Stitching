"""检测器纯逻辑的单元测试（不加载 RF-DETR 权重）。

运行：python video_stitching/test_detector_units.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import (PERSON_CLASS_ID, RFDETRDetector, annotate,
                      class_names_of)


class FakeDetections:
    """模拟 supervision.Detections。"""

    def __init__(self, xyxy, class_id, confidence, class_name=None):
        self.xyxy = np.asarray(xyxy, np.float32)
        self.class_id = np.asarray(class_id, np.int32)
        self.confidence = np.asarray(confidence, np.float32)
        self.data = {"class_name": np.array(class_name)} if class_name else {}


def test_person_class_id_is_one_based():
    """rfdetr 的 class_id 是 COCO 1 基编号，person == 1（不是 YOLO 的 0）。"""
    assert PERSON_CLASS_ID == 1
    from rfdetr.assets.coco_classes import COCO_CLASSES
    assert COCO_CLASSES[PERSON_CLASS_ID] == "person"
    print("test_person_class_id_is_one_based ok")


def test_class_names_prefers_data_field():
    det = FakeDetections([[0, 0, 10, 10]], [62], [0.9], class_name=["chair"])
    assert class_names_of(det) == ["chair"]
    print("test_class_names_prefers_data_field ok")


def test_class_names_falls_back_to_one_based_table():
    """没有 class_name 字段时，用 1 基表查，不能错位。"""
    det = FakeDetections([[0, 0, 10, 10], [1, 1, 5, 5]], [1, 62], [0.9, 0.8])
    table = {1: "person", 62: "chair"}
    assert class_names_of(det, fallback=table) == ["person", "chair"]
    print("test_class_names_falls_back_to_one_based_table ok")


def test_class_names_survives_unknown_id():
    det = FakeDetections([[0, 0, 10, 10]], [999], [0.9])
    assert class_names_of(det, fallback={1: "person"}) == ["999"]
    print("test_class_names_survives_unknown_id ok")


def test_annotate_counts_by_foot_point():
    """用底边中心判定是否在 ROI 内，而不是框中心。

    构造一个框：中心在多边形外，脚点在多边形内 —— 应当被计入。
    """
    img = np.zeros((200, 200, 3), np.uint8)
    poly = [[0, 120], [200, 120], [200, 200], [0, 200]]   # 下半部分
    boxes = np.array([[80, 60, 120, 140]], np.float32)     # 中心 y=100(外)，脚点 y=140(内)
    out, count = annotate(img, boxes, ["person"], [0.9], polygon=poly)
    assert count == 1, count
    assert out.shape == img.shape
    print("test_annotate_counts_by_foot_point ok")


def test_annotate_excludes_outside_roi():
    img = np.zeros((200, 200, 3), np.uint8)
    poly = [[0, 150], [200, 150], [200, 200], [0, 200]]
    boxes = np.array([[10, 10, 40, 60]], np.float32)       # 脚点 y=60，在 ROI 外
    _, count = annotate(img, boxes, ["person"], [0.9], polygon=poly)
    assert count == 0, count
    print("test_annotate_excludes_outside_roi ok")


def test_annotate_only_counts_target_class():
    img = np.zeros((200, 200, 3), np.uint8)
    boxes = np.array([[10, 10, 40, 60], [50, 50, 90, 100]], np.float32)
    _, count = annotate(img, boxes, ["person", "chair"], [0.9, 0.9], polygon=None)
    assert count == 1, count
    print("test_annotate_only_counts_target_class ok")


def test_annotate_without_polygon_counts_all():
    img = np.zeros((200, 200, 3), np.uint8)
    boxes = np.array([[10, 10, 40, 60], [50, 50, 90, 100]], np.float32)
    _, count = annotate(img, boxes, ["person", "person"], [0.9, 0.8], polygon=None)
    assert count == 2, count
    print("test_annotate_without_polygon_counts_all ok")


def test_annotate_does_not_mutate_input():
    img = np.zeros((120, 120, 3), np.uint8)
    before = img.copy()
    annotate(img, np.array([[5, 5, 60, 90]], np.float32), ["person"], [0.9],
             polygon=[[0, 0], [120, 0], [120, 120], [0, 120]])
    assert np.array_equal(img, before), "annotate 不应修改传入的图像"
    print("test_annotate_does_not_mutate_input ok")


def test_annotate_handles_no_detections():
    img = np.zeros((100, 100, 3), np.uint8)
    out, count = annotate(img, np.empty((0, 4), np.float32), [], [],
                          polygon=[[0, 0], [50, 0], [50, 50]])
    assert count == 0 and out.shape == img.shape
    print("test_annotate_handles_no_detections ok")


def test_detector_rejects_unknown_model():
    d = RFDETRDetector(model="does-not-exist")
    try:
        d._ensure_model()
    except ValueError as exc:
        assert "未知的 RF-DETR 型号" in str(exc)
        print("test_detector_rejects_unknown_model ok")
        return
    raise AssertionError("应当抛出 ValueError")


def test_detector_empty_image_returns_empty():
    d = RFDETRDetector()
    boxes, names, conf = d.detect(None)
    assert len(boxes) == 0 and names == [] and len(conf) == 0
    print("test_detector_empty_image_returns_empty ok")


if __name__ == "__main__":
    test_person_class_id_is_one_based()
    test_class_names_prefers_data_field()
    test_class_names_falls_back_to_one_based_table()
    test_class_names_survives_unknown_id()
    test_annotate_counts_by_foot_point()
    test_annotate_excludes_outside_roi()
    test_annotate_only_counts_target_class()
    test_annotate_without_polygon_counts_all()
    test_annotate_does_not_mutate_input()
    test_annotate_handles_no_detections()
    test_detector_rejects_unknown_model()
    test_detector_empty_image_returns_empty()
    print("\n全部测试通过。")
