import numpy as np
import sys
import yaml
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from pipeline import StitchingPipeline

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, url, index):
        super().__init__()
        self.url = url
        self.index = index
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.url)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img, self.index)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class StitchingThread(QThread):
    stitched_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, roi_polygon):
        super().__init__()
        self.pipeline = StitchingPipeline()
        self.roi_polygon = roi_polygon
        self.frame1 = None
        self.frame2 = None
        self._run_flag = True

    def run(self):
        while self._run_flag:
            if self.frame1 is not None and self.frame2 is not None:
                # Need to resize frames for faster homography in real-time, but LightGlue works well
                # We do stitching
                stitched_img = self.pipeline.stitch_frames(self.frame1, self.frame2)
                # Then we do detection
                result_img, count = self.pipeline.detect_and_count(stitched_img, self.roi_polygon)
                self.stitched_signal.emit(result_img, count)
                # clear frames after processing to wait for new ones
                self.frame1 = None
                self.frame2 = None
            else:
                self.msleep(30) # small sleep

    def update_frame(self, frame, index):
        if index == 0:
            self.frame1 = frame.copy()
        elif index == 1:
            self.frame2 = frame.copy()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Stitching & Detection")

        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        self.display_width = self.config['display']['width']
        self.display_height = self.config['display']['height']

        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.cams_layout = QHBoxLayout()
        self.main_layout.addLayout(self.cams_layout)

        self.cam1_label = QLabel(self)
        self.cam1_label.resize(self.display_width//2, self.display_height//2)
        self.cams_layout.addWidget(self.cam1_label)

        self.cam2_label = QLabel(self)
        self.cam2_label.resize(self.display_width//2, self.display_height//2)
        self.cams_layout.addWidget(self.cam2_label)

        self.stitched_label = QLabel(self)
        self.stitched_label.resize(self.display_width, self.display_height)
        self.main_layout.addWidget(self.stitched_label)

        self.count_label = QLabel("Person Count: 0", self)
        self.count_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.main_layout.addWidget(self.count_label)

        self.threads = []

        # Start Video Threads
        urls = [cam['url'] for cam in self.config['cameras']]

        # Start Stitching Thread
        roi = self.config['region_of_interest']
        self.stitching_thread = StitchingThread(roi)
        self.stitching_thread.stitched_signal.connect(self.update_stitched_image)
        self.stitching_thread.start()

        for idx, url in enumerate(urls[:2]):
            thread = VideoThread(url, idx)
            thread.change_pixmap_signal.connect(self.update_image)
            thread.change_pixmap_signal.connect(self.stitching_thread.update_frame)
            thread.start()
            self.threads.append(thread)

    def convert_cv_qt(self, cv_img, width, height):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(np.ndarray, int)
    def update_image(self, cv_img, index):
        qt_img = self.convert_cv_qt(cv_img, self.display_width//2, self.display_height//2)
        if index == 0:
            self.cam1_label.setPixmap(qt_img)
        elif index == 1:
            self.cam2_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray, int)
    def update_stitched_image(self, cv_img, count):
        qt_img = self.convert_cv_qt(cv_img, self.display_width, self.display_height)
        self.stitched_label.setPixmap(qt_img)
        self.count_label.setText(f"Person Count in Region: {count}")

    def closeEvent(self, event):
        for thread in self.threads:
            thread.stop()
        self.stitching_thread.stop()
        event.accept()

if __name__ == "__main__":
    import numpy as np # Needed in global scope for pyqtSignal
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
