"""RF-DETR 目标检测封装。

只负责「推理 + 画到图上」，不含拼接逻辑，便于单独替换或测试。

关于类别索引的坑：rfdetr 返回的 ``class_id`` 是 COCO 的 **1 基**编号
（person == 1），与 ``model.class_names`` 那个 0 基列表差 1。用错会把
「人」认成别的类。这里统一走 ``detections.data['class_name']``，
没有该字段时再退回 ``COCO_CLASSES``（同为 1 基字典）。
"""

import cv2
import numpy as np

PERSON_CLASS_ID = 1          # COCO 1 基编号中的 person

# 画框配色（BGR）
COLOR_ROI = (255, 128, 0)
COLOR_IN_ROI = (80, 230, 120)
COLOR_OUT_ROI = (70, 130, 255)
COLOR_OTHER = (200, 190, 120)

_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def _coco_table():
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
        return COCO_CLASSES
    except Exception:
        return {}


def class_names_of(detections, fallback=None):
    """稳妥地取出每个检测框的类别名。"""
    data = getattr(detections, "data", None) or {}
    if "class_name" in data:
        return [str(n) for n in data["class_name"]]

    ids = getattr(detections, "class_id", None)
    if ids is None:
        return []
    table = fallback if fallback is not None else _coco_table()
    return [str(table.get(int(i), int(i))) for i in ids]


class RFDETRDetector:
    """RF-DETR 检测器；模型在首次使用时才加载，避免拖慢启动。"""

    def __init__(self, model="nano", threshold=0.5, device=None,
                 classes=None, optimize=False):
        self.model_name = str(model).lower()
        self.threshold = float(threshold)
        self.device = device
        self.optimize = bool(optimize)
        # classes 为 None 表示不过滤；否则是允许的类别名集合（小写）
        self.classes = {str(c).lower() for c in classes} if classes else None
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        import rfdetr
        cls_name = _MODEL_CLASSES.get(self.model_name)
        if cls_name is None or not hasattr(rfdetr, cls_name):
            raise ValueError(
                f"未知的 RF-DETR 型号 {self.model_name!r}，"
                f"可选：{sorted(_MODEL_CLASSES)}")
        cls = getattr(rfdetr, cls_name)

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        try:
            self._model = cls(**kwargs)
        except TypeError:
            # 老版本不接受 device 参数
            self._model = cls()

        if self.optimize:
            try:
                self._model.optimize_for_inference()
            except Exception as exc:
                print(f"[detect] optimize_for_inference 跳过：{exc}")
        return self._model

    def detect(self, image_bgr):
        """对 BGR 图像推理，返回 (boxes_xyxy, names, confidences)。"""
        if image_bgr is None or image_bgr.size == 0:
            return np.empty((0, 4), np.float32), [], np.empty(0, np.float32)

        model = self._ensure_model()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        det = model.predict(rgb, threshold=self.threshold)

        boxes = np.asarray(getattr(det, "xyxy", np.empty((0, 4))), np.float32)
        names = class_names_of(det)
        conf = np.asarray(getattr(det, "confidence", np.empty(0)), np.float32)
        assert len(names) == len(conf) == len(boxes), "检测结果三者长度应一致"

        if self.classes is not None:
            keep = [i for i, n in enumerate(names) if n.lower() in self.classes]
            boxes = boxes[keep]
            names = [names[i] for i in keep]
            conf = conf[keep]
        return boxes, names, conf


def annotate(image, boxes, names, confidences, polygon=None,
             count_class="person", show_labels=True):
    """把检测结果画到图上，并统计 ROI 内指定类别的数量。

    返回 (标注后的图, ROI 内计数)。polygon 为空时不做 ROI 判定，
    计数退化为该类别的总数。
    """
    out = image.copy()
    pts = None
    if polygon:
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, COLOR_ROI, 2)

    count = 0
    target = str(count_class).lower() if count_class else None

    for box, name, conf in zip(boxes, names, confidences):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        is_target = target is not None and name.lower() == target

        if is_target:
            # 用底边中心（脚点）判定：它比框中心更贴近人在地面的实际位置
            fx, fy = (x1 + x2) // 2, y2
            inside = pts is None or cv2.pointPolygonTest(pts, (fx, fy), False) >= 0
            if inside:
                count += 1
            color = COLOR_IN_ROI if inside else COLOR_OUT_ROI
            cv2.circle(out, (fx, fy), 4, color, -1)
            thickness = 2
        else:
            color = COLOR_OTHER
            thickness = 1

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if show_labels:
            label = f"{name} {conf:.2f}"
            scale, lt = 0.5, 1
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, lt)
            ty = max(y1, th + 4)
            cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 4, ty), color, -1)
            cv2.putText(out, label, (x1 + 2, ty - base + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), lt, cv2.LINE_AA)

    return out, count
