import sys

import cv2
import numpy as np
import yaml
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow

from pipeline import StitchingPipeline
from widgets import StitchedView  # uic 需要能找到自定义控件

STYLE = """
QMainWindow, QWidget {
    background-color: #14161e;
    color: #d0d8f0;
    font-family: "Segoe UI", "SF Pro Display", Arial, sans-serif;
}
QLabel { color: #d0d8f0; }

QPushButton {
    background-color: #252a3a;
    color: #c0c8e0;
    border: 1px solid #353b55;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 13px;
    min-width: 88px;
}
QPushButton:hover   { background-color: #32384f; border-color: #4a5270; }
QPushButton:pressed { background-color: #3a4060; }

QPushButton#btn_draw {
    background-color: #143828; color: #60e890; border-color: #1f6040;
}
QPushButton#btn_draw:hover { background-color: #1a4e34; }
QPushButton#btn_draw[active="true"] {
    background-color: #602010; color: #ff9070; border-color: #903020;
}
QPushButton#btn_clear {
    background-color: #3a1818; color: #ff8888; border-color: #602828;
}
QPushButton#btn_clear:hover { background-color: #4a2020; }
QPushButton#btn_recal {
    background-color: #101e3a; color: #70b8ff; border-color: #203060;
}
QPushButton#btn_recal:hover { background-color: #182840; }

QStatusBar {
    background-color: #0e1018;
    color: #4a5880;
    font-size: 11px;
    border-top: 1px solid #1e2238;
    padding: 2px 6px;
}
"""


# ── 线程 ───────────────────────────────────────────────────────────────────────

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, url, index):
        super().__init__()
        self.url = url
        self.index = index
        self._run_flag = True

    def run(self):
        while self._run_flag:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                self.msleep(3000)   # 3 秒后重试
                continue
            while self._run_flag:
                ret, cv_img = cap.read()
                if not ret:
                    break           # 流中断，跳出内循环重新连接
                self.change_pixmap_signal.emit(cv_img, self.index)
            cap.release()
            if self._run_flag:
                self.msleep(2000)   # 断线后等 2 秒再重连

    def stop(self):
        self._run_flag = False
        self.wait()


class StitchingThread(QThread):
    stitched_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, roi_polygon):
        super().__init__()
        self.pipeline = StitchingPipeline()
        self.roi_polygon = list(roi_polygon)
        self.frame1 = None
        self.frame2 = None
        self._run_flag = True

    def run(self):
        while self._run_flag:
            if self.frame1 is not None and self.frame2 is not None:
                stitched = self.pipeline.stitch_frames(self.frame1, self.frame2)
                if self.roi_polygon:
                    result, count = self.pipeline.detect_and_count(stitched, self.roi_polygon)
                else:
                    result, count = stitched, 0
                self.stitched_signal.emit(result, count)
                self.frame1 = None
                self.frame2 = None
            else:
                self.msleep(30)

    @pyqtSlot(list)
    def set_roi(self, roi):
        self.roi_polygon = roi

    def update_frame(self, frame, index):
        if index == 0:
            self.frame1 = frame.copy()
        elif index == 1:
            self.frame2 = frame.copy()

    def stop(self):
        self._run_flag = False
        self.wait()


# ── 主窗口 ─────────────────────────────────────────────────────────────────────

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main_window.ui", self)

        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)

        self.display_width  = self.config["display"]["width"]
        self.display_height = self.config["display"]["height"]
        self.cam_w = self.display_width  // 2
        self.cam_h = self.display_height // 2

        # 根据配置设置各控件尺寸
        self.cam1_label.setFixedSize(self.cam_w, self.cam_h)
        self.cam2_label.setFixedSize(self.cam_w, self.cam_h)
        self.stitched_view.setFixedSize(self.display_width, self.display_height)

        # 连接按钮信号
        self.btn_draw.clicked.connect(self._toggle_draw)
        self.btn_clear.clicked.connect(self._clear_roi)
        self.btn_recal.clicked.connect(self._recalibrate)
        self.fovSlider.valueChanged.connect(
            lambda v: self.fovLabel.setText(f"FOV: {v}°"))
        self.fovSlider.sliderReleased.connect(
            lambda: self._on_fov_changed(self.fovSlider.value()))
        self.stitched_view.roi_changed.connect(self._on_roi_changed)

        self._start_threads()
        self.statusbar.showMessage(
            '就绪  ·  点击【绘制区域】后在拼接图上依次点击顶点定义监控区域')

    # ── 线程启动 ───────────────────────────────────────────────────────────────

    def _start_threads(self):
        roi = self.config.get("region_of_interest", [])
        if roi:
            self.stitched_view._roi_img = [list(p) for p in roi]

        self.stitching_thread = StitchingThread(roi)
        self.stitching_thread.stitched_signal.connect(self.update_stitched_image)
        self.stitching_thread.start()

        self.threads = []
        for idx, cam in enumerate(self.config["cameras"][:2]):
            t = VideoThread(cam["url"], idx)
            t.change_pixmap_signal.connect(self.update_image)
            t.change_pixmap_signal.connect(self.stitching_thread.update_frame)
            t.start()
            self.threads.append(t)

    # ── 按钮操作 ───────────────────────────────────────────────────────────────

    def _toggle_draw(self):
        if self.stitched_view.is_drawing:
            self.stitched_view._drawing = False
            self.stitched_view._draw_pts = []
            self.stitched_view.setCursor(Qt.ArrowCursor)
            self.stitched_view.update()
            self._set_draw_btn(active=False)
            self.statusbar.showMessage("已取消绘制。")
        else:
            self.stitched_view.start_drawing()
            self._set_draw_btn(active=True)
            self.statusbar.showMessage(
                "绘制模式：点击添加顶点，双击或点击起点完成，右键撤销上一点")

    def _clear_roi(self):
        self.stitched_view.clear_roi()
        self.count_num.setText("—")
        self._set_draw_btn(active=False)
        self.statusbar.showMessage("监控区域已清除。")

    def _recalibrate(self):
        self.stitching_thread.pipeline.homography = None
        self.statusbar.showMessage("正在重新校准拼接矩阵…", 4000)

    def _on_fov_changed(self, value):
        self.fovLabel.setText(f"FOV: {value}°")
        self.stitching_thread.pipeline.fov = value
        self._recalibrate()

    def _on_roi_changed(self, roi):
        self.stitching_thread.set_roi(roi)
        self._set_draw_btn(active=False)
        if roi:
            self._save_roi(roi)
            self.statusbar.showMessage(
                f"监控区域已设置（{len(roi)} 个顶点），已保存至 config.yaml")
        else:
            self.statusbar.showMessage("监控区域已清除。")

    def _save_roi(self, roi):
        self.config["region_of_interest"] = roi
        with open("config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=None, allow_unicode=True)

    def _set_draw_btn(self, active):
        self.btn_draw.setText("取消绘制" if active else "绘制区域")
        self.btn_draw.setProperty("active", "true" if active else "false")
        self.btn_draw.style().unpolish(self.btn_draw)
        self.btn_draw.style().polish(self.btn_draw)

    # ── 画面更新 ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pixmap(cv_img, w, h):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ih, iw, ch = rgb.shape
        qimg = QImage(rgb.data, iw, ih, ch * iw, QImage.Format_RGB888)
        return QPixmap.fromImage(
            qimg.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(np.ndarray, int)
    def update_image(self, cv_img, index):
        pix = self._to_pixmap(cv_img, self.cam_w, self.cam_h)
        (self.cam1_label if index == 0 else self.cam2_label).setPixmap(pix)

    @pyqtSlot(np.ndarray, int)
    def update_stitched_image(self, cv_img, count):
        self.stitched_view.update_image(cv_img)
        if count >= 0:
            self.count_num.setText(str(count))

    def closeEvent(self, event):
        for t in self.threads:
            t.stop()
        self.stitching_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = App()
    window.show()
    sys.exit(app.exec_())
