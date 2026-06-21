import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class VideoThread(QThread):
    frame_signal = pyqtSignal(str, np.ndarray)
    error_signal = pyqtSignal(str, str)

    TARGET_FPS = 8

    def __init__(self, camera_id, url):
        super().__init__()
        self.camera_id = camera_id
        self.url = url
        self._run_flag = True

    def run(self):
        self._run_flag = True
        while self._run_flag:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                self.error_signal.emit(self.camera_id, f"无法打开视频流：{self.url}")
                self.msleep(2000)
                continue

            src_fps = cap.get(cv2.CAP_PROP_FPS)
            if src_fps <= 0:
                src_fps = 30.0
            skip = max(1, int(round(src_fps / self.TARGET_FPS)))
            frame_interval = 1.0 / self.TARGET_FPS
            frame_count = 0

            while self._run_flag:
                t0 = time.monotonic()
                ret = cap.grab()
                if not ret:
                    self.error_signal.emit(self.camera_id, "视频流中断，正在重连。")
                    break

                frame_count += 1
                if frame_count % skip != 0:
                    continue

                ret, frame = cap.retrieve()
                if ret and frame is not None and frame.size > 0:
                    self.frame_signal.emit(self.camera_id, frame.copy())

                elapsed = time.monotonic() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cap.release()
            if self._run_flag:
                self.msleep(1500)

    def stop(self):
        self._run_flag = False
        self.wait()
