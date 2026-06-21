import os
import sys

import cv2
from PyQt5.QtCore import QPointF, QRectF, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration import CalibrationProject
from fusion import MultiCameraFusion
from video_thread import VideoThread


class RegistrationThread(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(self, fusion, cameras, bev_images, missing_names):
        super().__init__()
        self.fusion = fusion
        self.cameras = list(cameras)
        self.bev_images = dict(bev_images)
        self.missing_names = list(missing_names)

    def run(self):
        msg = self.fusion.compute_pairwise_registrations(self.cameras, self.bev_images)
        if self.missing_names:
            msg += " | 缺少帧：" + ", ".join(self.missing_names)
        self.finished_signal.emit(msg)


class ZoomableImageWidget(QWidget):
    DRAG_THRESHOLD = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap = None
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._drag_start = QPointF()
        self._press_pos = None
        self._is_panning = False
        self.setMinimumSize(360, 260)

    def setImage(self, pixmap):
        self._source_pixmap = pixmap
        self.update()

    def resetView(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def _get_transform(self):
        if self._source_pixmap is None or self._source_pixmap.width() == 0:
            return 0, 0, 1.0
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        ww, wh = self.width(), self.height()
        base_scale = min(ww / pw, wh / ph)
        scale = base_scale * self._zoom
        ox = (ww - pw * scale) / 2 + self._pan.x()
        oy = (wh - ph * scale) / 2 + self._pan.y()
        return ox, oy, scale

    def _widget_to_image(self, pos):
        ox, oy, scale = self._get_transform()
        if scale <= 0:
            return None
        return (pos.x() - ox) / scale, (pos.y() - oy) / scale

    def _image_to_widget(self, ix, iy):
        ox, oy, scale = self._get_transform()
        return ox + ix * scale, oy + iy * scale

    def wheelEvent(self, event):
        if self._source_pixmap is None:
            return
        img_pos = self._widget_to_image(event.pos())
        if img_pos is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._zoom = max(0.1, min(20.0, self._zoom * factor))
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        ww, wh = self.width(), self.height()
        base_scale = min(ww / pw, wh / ph)
        scale = base_scale * self._zoom
        self._pan = QPointF(
            event.pos().x() - img_pos[0] * scale - (ww - pw * scale) / 2,
            event.pos().y() - img_pos[1] * scale - (wh - ph * scale) / 2,
        )
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._press_pos = QPointF(event.pos().x(), event.pos().y())
            self._drag_start = QPointF(event.pos().x(), event.pos().y())
            self._is_panning = False

    def mouseMoveEvent(self, event):
        if self._press_pos is None:
            return
        if not self._is_panning:
            dx = event.pos().x() - self._press_pos.x()
            dy = event.pos().y() - self._press_pos.y()
            if (dx * dx + dy * dy) ** 0.5 > self.DRAG_THRESHOLD:
                self._is_panning = True
                self.setCursor(Qt.ClosedHandCursor)
        if self._is_panning:
            dx = event.pos().x() - self._drag_start.x()
            dy = event.pos().y() - self._drag_start.y()
            self._pan = QPointF(self._pan.x() + dx, self._pan.y() + dy)
            self._drag_start = QPointF(event.pos().x(), event.pos().y())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            was_panning = self._is_panning
            self._is_panning = False
            self._press_pos = None
            self.setCursor(Qt.ArrowCursor)
            if not was_panning:
                self._on_click(event)

    def _on_click(self, event):
        pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#f0f6fa"))
        if self._source_pixmap is None:
            painter.setPen(QColor("#999999"))
            painter.drawText(self.rect(), Qt.AlignCenter, "暂无画面")
            painter.end()
            return

        ox, oy, scale = self._get_transform()
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        painter.drawPixmap(
            QRectF(ox, oy, pw * scale, ph * scale),
            self._source_pixmap,
            QRectF(0, 0, pw, ph),
        )
        painter.end()


class ClickableVideoWidget(ZoomableImageWidget):
    point_added = pyqtSignal(float, float)
    point_removed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []

    def set_points_normalized(self, points):
        self.points = [tuple(p) for p in points]
        self.update()

    def _on_click(self, event):
        if self._source_pixmap is None or event.button() != Qt.LeftButton:
            return
        img_pos = self._widget_to_image(event.pos())
        if img_pos is None:
            return
        ix, iy = img_pos
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        if 0 <= ix <= pw and 0 <= iy <= ph:
            nx, ny = ix / pw, iy / ph
            self.points.append((nx, ny))
            self.point_added.emit(nx, ny)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self._source_pixmap is not None:
            idx = self._nearest_point(event.pos())
            if idx is not None:
                self.points.pop(idx)
                self.point_removed.emit(idx)
                self.update()
                return
        super().mousePressEvent(event)

    def _nearest_point(self, widget_pos, threshold=20):
        if not self.points or self._source_pixmap is None:
            return None
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        best_idx, best_dist = None, float("inf")
        for i, (nx, ny) in enumerate(self.points):
            wx, wy = self._image_to_widget(nx * pw, ny * ph)
            dist = ((widget_pos.x() - wx) ** 2 + (widget_pos.y() - wy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx if best_dist <= threshold else None

    def remove_point(self, index):
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self.update()

    def clear_points(self):
        self.points = []
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._source_pixmap is None or not self.points:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()

        if len(self.points) > 1:
            painter.setPen(QPen(QColor(126, 200, 227, 180), 2))
            for i in range(len(self.points) - 1):
                x1, y1 = self._image_to_widget(self.points[i][0] * pw, self.points[i][1] * ph)
                x2, y2 = self._image_to_widget(self.points[i + 1][0] * pw, self.points[i + 1][1] * ph)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            if len(self.points) >= 3:
                x1, y1 = self._image_to_widget(self.points[-1][0] * pw, self.points[-1][1] * ph)
                x2, y2 = self._image_to_widget(self.points[0][0] * pw, self.points[0][1] * ph)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        for i, (nx, ny) in enumerate(self.points):
            wx, wy = self._image_to_widget(nx * pw, ny * ph)
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(QBrush(QColor(58, 152, 195)))
            painter.drawEllipse(int(wx - 8), int(wy - 8), 16, 16)
            font = painter.font()
            font.setBold(True)
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(Qt.white)
            painter.drawText(QRectF(wx - 8, wy - 8, 16, 16), Qt.AlignCenter, str(i + 1))
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多路视频 BEV 拼接工具")
        self.setGeometry(80, 80, 1680, 980)
        self.project = CalibrationProject()
        self.fusion = MultiCameraFusion(canvas_padding=self.project.padding)
        self.threads = {}
        self.frames = {}
        self.frozen_frames = {}
        self.registration_thread = None
        self.mosaic_dirty = False
        self.current_camera_index = 0
        self._loading_camera = False
        self.init_ui()
        self.load_styles()
        self.load_camera_to_ui(0)
        self.mosaic_timer = QTimer(self)
        self.mosaic_timer.setInterval(200)
        self.mosaic_timer.timeout.connect(self.flush_mosaic_refresh)
        self.mosaic_timer.start()

    def load_styles(self):
        try:
            style_path = os.path.join(os.path.dirname(__file__), "styles.qss")
            with open(style_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        except Exception as exc:
            print(f"Could not load styles.qss: {exc}")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self.camera_combo = QComboBox()
        for cam in self.project.cameras:
            self.camera_combo.addItem(cam.name, cam.camera_id)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)

        self.enabled_check = QCheckBox("启用")
        self.enabled_check.stateChanged.connect(self.on_enabled_changed)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("RTSP 地址或本地视频路径")
        self.url_input.editingFinished.connect(self.on_url_changed)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(10, 1000)
        self.scale_spin.setValue(self.project.scale)
        self.scale_spin.setSuffix(" px/m")
        self.scale_spin.valueChanged.connect(self.on_scale_changed)

        self.start_btn = QPushButton("开始视频")
        self.stop_btn = QPushButton("停止视频")
        self.stop_btn.setEnabled(False)
        self.save_btn = QPushButton("保存配置")
        self.start_btn.clicked.connect(self.start_streams)
        self.stop_btn.clicked.connect(self.stop_streams)
        self.save_btn.clicked.connect(self.save_project)

        top_bar.addWidget(QLabel("摄像头："))
        top_bar.addWidget(self.camera_combo)
        top_bar.addWidget(self.enabled_check)
        top_bar.addWidget(QLabel("视频源："))
        top_bar.addWidget(self.url_input, 1)
        top_bar.addWidget(QLabel("统一比例："))
        top_bar.addWidget(self.scale_spin)
        top_bar.addWidget(self.start_btn)
        top_bar.addWidget(self.stop_btn)
        top_bar.addWidget(self.save_btn)
        main_layout.addLayout(top_bar)

        display_layout = QHBoxLayout()
        original_group = QGroupBox("当前相机原始画面（左键选点 | 右键删点 | 滚轮缩放 | 拖拽平移）")
        original_layout = QVBoxLayout()
        self.video_widget = ClickableVideoWidget()
        self.video_widget.point_added.connect(self.on_point_added)
        self.video_widget.point_removed.connect(self.on_point_removed)
        original_layout.addWidget(self.video_widget)
        original_group.setLayout(original_layout)

        bev_group = QGroupBox("当前相机本地 BEV")
        bev_layout = QVBoxLayout()
        self.bev_widget = ZoomableImageWidget()
        bev_layout.addWidget(self.bev_widget)
        bev_group.setLayout(bev_layout)

        mosaic_group = QGroupBox("融合 Mosaic")
        mosaic_layout = QVBoxLayout()
        self.mosaic_widget = ZoomableImageWidget()
        mosaic_layout.addWidget(self.mosaic_widget)
        mosaic_group.setLayout(mosaic_layout)

        display_layout.addWidget(original_group, 34)
        display_layout.addWidget(bev_group, 33)
        display_layout.addWidget(mosaic_group, 33)
        main_layout.addLayout(display_layout, 1)

        bottom_layout = QHBoxLayout()
        table_group = QGroupBox("当前相机标定点")
        table_layout = QVBoxLayout()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["图像 X", "图像 Y", "本地 X（米）", "本地 Y（米）", ""])
        header = self.table.horizontalHeader()
        for col in range(4):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(4, 46)
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        bottom_layout.addWidget(table_group, 70)

        controls_group = QGroupBox("操作")
        controls_layout = QVBoxLayout()
        self.compute_btn = QPushButton("计算当前相机 BEV")
        self.freeze_btn = QPushButton("冻结当前帧")
        self.clear_btn = QPushButton("清除当前相机点")
        self.register_btn = QPushButton("自动拼接 / 重算拼接")
        self.reset_view_btn = QPushButton("重置视图")
        self.compute_btn.clicked.connect(self.compute_current_bev)
        self.freeze_btn.clicked.connect(self.toggle_freeze_current_frame)
        self.clear_btn.clicked.connect(self.clear_current_points)
        self.register_btn.clicked.connect(self.recompute_registration)
        self.reset_view_btn.clicked.connect(self.reset_views)
        controls_layout.addWidget(self.compute_btn)
        controls_layout.addWidget(self.freeze_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addWidget(self.register_btn)
        controls_layout.addWidget(self.reset_view_btn)
        self.quality_label = QLabel("配准状态：尚未自动拼接")
        self.quality_label.setWordWrap(True)
        controls_layout.addWidget(self.quality_label)
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        bottom_layout.addWidget(controls_group, 30)
        main_layout.addLayout(bottom_layout)

    @property
    def current_camera(self):
        return self.project.cameras[self.current_camera_index]

    def on_camera_changed(self, index):
        if index < 0 or self._loading_camera:
            return
        if not self.sync_ui_to_camera():
            QMessageBox.warning(self, "数据无效", "当前相机标定表存在无效坐标，请修正后再切换。")
            self._loading_camera = True
            self.camera_combo.setCurrentIndex(self.current_camera_index)
            self._loading_camera = False
            return
        self.current_camera_index = index
        self.load_camera_to_ui(index)

    def load_camera_to_ui(self, index):
        self._loading_camera = True
        cam = self.project.cameras[index]
        self.enabled_check.setChecked(cam.enabled)
        self.url_input.setText(cam.url)
        self.table.setRowCount(0)
        for img_pt, world_pt in zip(cam.image_points, cam.local_world_points):
            self.add_table_row(img_pt[0], img_pt[1], world_pt[0], world_pt[1])
        self.video_widget.setImage(None)
        self.bev_widget.setImage(None)
        self.reload_video_points()
        self.refresh_current_views()
        self.refresh_quality_label()
        self.refresh_freeze_button()
        self._loading_camera = False

    def read_table_points(self, strict=False):
        image_points = []
        world_points = []
        for row in range(self.table.rowCount()):
            try:
                image_points.append([
                    float(self.table.item(row, 0).text()),
                    float(self.table.item(row, 1).text()),
                ])
                world_points.append([
                    float(self.table.item(row, 2).text()),
                    float(self.table.item(row, 3).text()),
                ])
            except Exception as exc:
                if strict:
                    raise ValueError(f"第 {row + 1} 行标定点坐标无效。") from exc
                return None, None
        return image_points, world_points

    def sync_ui_to_camera(self, strict=False):
        if self._loading_camera:
            return True
        cam = self.current_camera
        cam.enabled = self.enabled_check.isChecked()
        cam.url = self.url_input.text().strip()
        image_points, world_points = self.read_table_points(strict=strict)
        if image_points is None:
            return False
        cam.image_points = image_points
        cam.local_world_points = world_points
        return True

    def on_enabled_changed(self):
        if not self._loading_camera:
            self.current_camera.enabled = self.enabled_check.isChecked()

    def on_url_changed(self):
        if not self._loading_camera:
            self.current_camera.url = self.url_input.text().strip()

    def on_scale_changed(self, value):
        self.project.scale = float(value)
        for cam in self.project.cameras:
            if len(cam.image_points) >= 4 and len(cam.local_world_points) == len(cam.image_points):
                cam.compute_local_bev(self.project.scale, self.project.padding)
        self.refresh_current_views()
        self.schedule_mosaic_refresh()

    def add_table_row(self, img_x, img_y, world_x, world_y):
        row = self.table.rowCount()
        self.table.insertRow(row)
        item_x = QTableWidgetItem(str(int(round(float(img_x)))))
        item_y = QTableWidgetItem(str(int(round(float(img_y)))))
        item_x.setFlags(item_x.flags() & ~Qt.ItemIsEditable)
        item_y.setFlags(item_y.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 0, item_x)
        self.table.setItem(row, 1, item_y)
        self.table.setItem(row, 2, QTableWidgetItem(str(world_x)))
        self.table.setItem(row, 3, QTableWidgetItem(str(world_y)))
        del_btn = QPushButton("X")
        del_btn.setObjectName("deleteBtn")
        del_btn.clicked.connect(self._on_table_delete)
        self.table.setCellWidget(row, 4, del_btn)

    def on_point_added(self, nx, ny):
        frame = self.calibration_frame(self.current_camera.camera_id)
        if frame is None:
            return
        h, w = frame.shape[:2]
        orig_x = int(nx * w)
        orig_y = int(ny * h)
        default_world = [(0, 0), (5, 0), (5, 5), (0, 5)]
        row = self.table.rowCount()
        wx, wy = default_world[row % 4] if row < 4 else (0.0, 0.0)
        self.add_table_row(orig_x, orig_y, wx, wy)
        self.sync_ui_to_camera()

    def _on_table_delete(self):
        btn = self.sender()
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row, 4) is btn:
                self.table.removeRow(row)
                self.video_widget.remove_point(row)
                self.sync_ui_to_camera()
                break

    def on_point_removed(self, index):
        if 0 <= index < self.table.rowCount():
            self.table.removeRow(index)
            self.sync_ui_to_camera()

    def clear_current_points(self):
        self.table.setRowCount(0)
        self.video_widget.clear_points()
        self.current_camera.clear_points()
        self.bev_widget.setImage(None)
        self.refresh_quality_label()
        self.schedule_mosaic_refresh()

    def reload_video_points(self):
        cam = self.current_camera
        frame = self.calibration_frame(cam.camera_id)
        if frame is None:
            self.video_widget.set_points_normalized([])
            return
        h, w = frame.shape[:2]
        points = []
        for x, y in cam.image_points:
            points.append((float(x) / max(w, 1), float(y) / max(h, 1)))
        self.video_widget.set_points_normalized(points)

    def compute_current_bev(self):
        try:
            if not self.sync_ui_to_camera(strict=True):
                return
        except ValueError as exc:
            QMessageBox.critical(self, "数据无效", str(exc))
            return
        cam = self.current_camera
        ok, msg = cam.compute_local_bev(self.project.scale, self.project.padding)
        if ok:
            self.refresh_current_views()
            self.schedule_mosaic_refresh()
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "计算失败", msg)

    def recompute_registration(self):
        try:
            if not self.sync_ui_to_camera(strict=True):
                return
        except ValueError as exc:
            QMessageBox.critical(self, "数据无效", str(exc))
            return
        active = self.project.enabled_calibrated_cameras()
        if len(active) < 2:
            QMessageBox.warning(self, "提示", "至少需要 2 路已启用且已标定的摄像头。")
            return

        bev_images = {}
        missing = []
        for cam in active:
            frame = self.calibration_frame(cam.camera_id)
            if frame is None:
                missing.append(cam.name)
                continue
            bev, mask = cam.warp_to_bev(frame)
            if bev is not None:
                bev_images[cam.camera_id] = bev
        if len(bev_images) < 2:
            QMessageBox.warning(self, "提示", "当前可用帧不足，无法自动拼接。")
            return

        self.quality_label.setText("配准状态：LightGlueStick 正在后台计算，请稍候...")
        self.register_btn.setEnabled(False)
        self.registration_thread = RegistrationThread(self.fusion, active, bev_images, missing)
        self.registration_thread.finished_signal.connect(self.on_registration_finished)
        self.registration_thread.finished.connect(self.registration_thread.deleteLater)
        self.registration_thread.start()

    def on_registration_finished(self, msg):
        self.quality_label.setText("配准状态：" + msg)
        self.register_btn.setEnabled(True)
        self.registration_thread = None
        self.schedule_mosaic_refresh()

    def toggle_freeze_current_frame(self):
        cam_id = self.current_camera.camera_id
        if cam_id in self.frozen_frames:
            self.frozen_frames.pop(cam_id, None)
        else:
            frame = self.frames.get(cam_id)
            if frame is None:
                QMessageBox.warning(self, "提示", "当前相机还没有可冻结的画面。")
                return
            self.frozen_frames[cam_id] = frame.copy()
        self.refresh_freeze_button()
        self.refresh_current_views()

    def refresh_freeze_button(self):
        if self.current_camera.camera_id in self.frozen_frames:
            self.freeze_btn.setText("恢复实时画面")
        else:
            self.freeze_btn.setText("冻结当前帧")

    def calibration_frame(self, camera_id):
        if camera_id in self.frozen_frames:
            return self.frozen_frames[camera_id]
        return self.frames.get(camera_id)

    def start_streams(self):
        if not self.sync_ui_to_camera():
            QMessageBox.warning(self, "数据无效", "当前相机标定表存在无效坐标，请修正后再启动视频。")
            return
        self.stop_streams()
        started = 0
        for cam in self.project.cameras:
            if not cam.enabled:
                continue
            if not cam.url:
                continue
            thread = VideoThread(cam.camera_id, cam.url)
            thread.frame_signal.connect(self.on_frame)
            thread.error_signal.connect(self.on_stream_error)
            thread.start()
            self.threads[cam.camera_id] = thread
            started += 1
        if started == 0:
            QMessageBox.warning(self, "提示", "没有启用且填写视频源的摄像头。")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_streams(self):
        for thread in list(self.threads.values()):
            thread.stop()
        self.threads.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_frame(self, camera_id, frame):
        self.frames[camera_id] = frame
        if camera_id == self.current_camera.camera_id and camera_id not in self.frozen_frames:
            self.video_widget.setImage(self.cv_to_pixmap(frame))
            self.reload_video_points()
            self.refresh_current_views()
        self.schedule_mosaic_refresh()

    def on_stream_error(self, camera_id, msg):
        cam = next((c for c in self.project.cameras if c.camera_id == camera_id), None)
        name = cam.name if cam else camera_id
        self.quality_label.setText(f"视频状态：{name} {msg}")

    def refresh_current_views(self):
        cam = self.current_camera
        frame = self.calibration_frame(cam.camera_id)
        if frame is None:
            return
        self.video_widget.setImage(self.cv_to_pixmap(frame))
        if cam.is_calibrated:
            bev, mask = cam.warp_to_bev(frame)
            if bev is not None:
                self.bev_widget.setImage(self.cv_to_pixmap(bev))

    def refresh_mosaic(self):
        mosaic, debug_bevs = self.fusion.build_mosaic(self.project.cameras, self.frames)
        if mosaic is not None:
            self.mosaic_widget.setImage(self.cv_to_pixmap(mosaic))
        else:
            self.mosaic_widget.setImage(None)

    def schedule_mosaic_refresh(self):
        self.mosaic_dirty = True

    def flush_mosaic_refresh(self):
        if not self.mosaic_dirty:
            return
        self.mosaic_dirty = False
        self.refresh_mosaic()

    def refresh_quality_label(self):
        cam = self.current_camera
        q = cam.last_registration_quality or {}
        if q.get("anchor"):
            text = f"配准状态：{cam.name} 是 anchor"
        elif q:
            text = (
                f"配准状态：{cam.name} {q.get('message', '')} | "
                f"matches={q.get('matches', 0)} inliers={q.get('inliers', 0)} "
                f"ratio={q.get('inlier_ratio', 0):.2f}"
            )
        else:
            text = "配准状态：尚未自动拼接"
        self.quality_label.setText(text)

    def reset_views(self):
        self.video_widget.resetView()
        self.bev_widget.resetView()
        self.mosaic_widget.resetView()

    def save_project(self):
        try:
            if not self.sync_ui_to_camera(strict=True):
                return
        except ValueError as exc:
            QMessageBox.critical(self, "数据无效", str(exc))
            return
        self.project.save()
        QMessageBox.information(self, "成功", f"配置已保存：{self.project.config_path}")

    @staticmethod
    def cv_to_pixmap(cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def closeEvent(self, event):
        self.sync_ui_to_camera()
        self.stop_streams()
        if self.registration_thread is not None:
            self.registration_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
