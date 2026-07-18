"""帧仓库与后台融合线程:mosaic 计算完全移出 GUI 主线程。"""

import threading
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from fusion_cache import build_fusion_cache, compute_signature, fuse


class FrameStore:
    """互斥锁保护的最新帧字典。put 由 VideoThread 线程直接调用。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._frames = {}
        self._seq = 0

    def put(self, camera_id, frame):
        with self._lock:
            self._frames[camera_id] = frame
            self._seq += 1

    def snapshot(self):
        with self._lock:
            return dict(self._frames), self._seq

    def clear(self):
        with self._lock:
            self._frames.clear()
            self._seq += 1


class FusionWorker(QThread):
    """固定节奏(默认 8Hz)拉取最新帧快照、融合并发出 mosaic。

    通过缓存签名自动感知标定/配准/帧尺寸变化并重建 FusionCache,
    上层无需手动通知刷新。
    """

    mosaic_ready = pyqtSignal(object)  # np.ndarray 或 None

    INTERVAL_MS = 125

    def __init__(self, frame_store, project):
        super().__init__()
        self.frame_store = frame_store
        self.project = project
        self._run_flag = True
        self._cache = None
        self._signature = None
        self._last_seq = -1

    def run(self):
        while self._run_flag:
            try:
                self._tick()
            except Exception:
                traceback.print_exc()
            self.msleep(self.INTERVAL_MS)

    def _tick(self):
        frames, seq = self.frame_store.snapshot()
        signature = compute_signature(
            self.project.cameras, frames, self.project.padding
        )
        rebuilt = False
        if signature != self._signature:
            self._signature = signature
            self._cache = build_fusion_cache(
                self.project.cameras, frames, self.project.padding, signature
            )
            rebuilt = True
            if self._cache is None:
                self.mosaic_ready.emit(None)

        if self._cache is None:
            self._last_seq = seq
            return
        if seq == self._last_seq and not rebuilt:
            return
        self._last_seq = seq
        self.mosaic_ready.emit(fuse(self._cache, frames))

    def stop(self):
        self._run_flag = False
        self.wait()
