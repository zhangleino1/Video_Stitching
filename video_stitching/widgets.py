import math
import cv2
from PyQt5.QtCore import QPoint, Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QImage
from PyQt5.QtWidgets import QLabel

class StitchedView(QLabel):
    """拼接画面控件，支持滚轮缩放、拖动和点击绘制多边形监控区域。"""
    roi_changed = pyqtSignal(list)  # [[x, y], ...] 原始图像坐标

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        
        self._drawing = False
        self._draw_pts_img = []   # 绘制中的顶点（图像坐标）
        self._mouse_pos = None    # 当前鼠标位置（标签坐标）
        self._roi_img = []        # 已提交的 ROI（图像坐标）
        
        self._qimage = None
        
        # 缩放与平移状态
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._is_panning = False
        self._last_pan_pos = None

    def update_image(self, cv_img):
        # 将 cv_img 转换为 QImage 缓存起来自己画，不再依赖 QLabel.setPixmap
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ih, iw, ch = rgb.shape
        bytes_per_line = ch * iw
        self._qimage = QImage(rgb.data, iw, ih, bytes_per_line, QImage.Format_RGB888).copy()
        self.update()

    # (为了兼容 app.py 旧代码保留空实现)
    def set_image_size(self, w, h):
        pass

    @property
    def is_drawing(self):
        return self._drawing

    def start_drawing(self):
        self._drawing = True
        self._draw_pts_img = []
        self._mouse_pos = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def clear_roi(self):
        self._drawing = False
        self._draw_pts_img = []
        self._roi_img = []
        self.setCursor(Qt.ArrowCursor)
        self.roi_changed.emit([])
        self.update()

    # ── 坐标转换 ──────────────────────────────────────────────────────────────

    def _lbl_to_img(self, qpt):
        if not self._qimage:
            return [qpt.x(), qpt.y()]
        iw, ih = self._qimage.width(), self._qimage.height()
        vw, vh = self.width(), self.height()
        
        scale_fit = min(vw / iw, vh / ih) if iw > 0 and ih > 0 else 1.0
        scale = scale_fit * self._zoom
        
        img_draw_w = iw * scale
        img_draw_h = ih * scale
        
        draw_x = (vw - img_draw_w) / 2 + self._pan_x
        draw_y = (vh - img_draw_h) / 2 + self._pan_y
        
        img_x = (qpt.x() - draw_x) / scale
        img_y = (qpt.y() - draw_y) / scale
        return [int(img_x), int(img_y)]

    def _img_to_lbl(self, xy):
        if not self._qimage:
            return QPoint(int(xy[0]), int(xy[1]))
        iw, ih = self._qimage.width(), self._qimage.height()
        vw, vh = self.width(), self.height()
        
        scale_fit = min(vw / iw, vh / ih) if iw > 0 and ih > 0 else 1.0
        scale = scale_fit * self._zoom
        
        img_draw_w = iw * scale
        img_draw_h = ih * scale
        
        draw_x = (vw - img_draw_w) / 2 + self._pan_x
        draw_y = (vh - img_draw_h) / 2 + self._pan_y
        
        lbl_x = xy[0] * scale + draw_x
        lbl_y = xy[1] * scale + draw_y
        return QPoint(int(lbl_x), int(lbl_y))

    # ── 鼠标与滚轮事件 ─────────────────────────────────────────────────────────

    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self._zoom *= zoom_factor
        else:
            self._zoom /= zoom_factor
            
        self._zoom = max(0.1, min(self._zoom, 20.0))
        self.update()

    def mousePressEvent(self, event):
        if self._drawing:
            if event.button() == Qt.LeftButton:
                pos = event.pos()
                if len(self._draw_pts_img) >= 3:
                    fp_lbl = self._img_to_lbl(self._draw_pts_img[0])
                    if math.hypot(pos.x() - fp_lbl.x(), pos.y() - fp_lbl.y()) < 15:
                        self._commit()
                        return
                img_pos = self._lbl_to_img(pos)
                self._draw_pts_img.append(img_pos)
                self.update()
            elif event.button() == Qt.RightButton and self._draw_pts_img:
                self._draw_pts_img.pop()
                self.update()
        else:
            if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
                self._is_panning = True
                self._last_pan_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        if self._drawing and len(self._draw_pts_img) >= 3:
            self._commit()
        elif not self._drawing:
            # 双击复原缩放和平移
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self.update()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._mouse_pos = event.pos()
            self.update()
        elif self._is_panning and self._last_pan_pos:
            delta = event.pos() - self._last_pan_pos
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._last_pan_pos = event.pos()
            self.update()

    def _commit(self):
        self._roi_img = list(self._draw_pts_img)
        self._drawing = False
        self._draw_pts_img = []
        self.setCursor(Qt.ArrowCursor)
        self.roi_changed.emit(self._roi_img)
        self.update()

    # ── 绘制 ──────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        # 不再调用 super().paintEvent(event)，我们完全自己接管绘制
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 绘制图像
        if self._qimage:
            iw, ih = self._qimage.width(), self._qimage.height()
            vw, vh = self.width(), self.height()
            
            scale_fit = min(vw / iw, vh / ih) if iw > 0 and ih > 0 else 1.0
            scale = scale_fit * self._zoom
            
            img_draw_w = iw * scale
            img_draw_h = ih * scale
            
            draw_x = (vw - img_draw_w) / 2 + self._pan_x
            draw_y = (vh - img_draw_h) / 2 + self._pan_y
            
            target_rect = QRectF(draw_x, draw_y, img_draw_w, img_draw_h)
            source_rect = QRectF(0, 0, iw, ih)
            
            # 使用平滑缩放绘制图片
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(target_rect, self._qimage, source_rect)

        # 2. 绘制已完成的 ROI
        if self._roi_img:
            self._paint_poly(painter,
                             [self._img_to_lbl(p) for p in self._roi_img],
                             committed=True)

        # 3. 绘制正在绘制中的 ROI
        if self._drawing and self._draw_pts_img:
            pts_lbl = [self._img_to_lbl(p) for p in self._draw_pts_img]
            self._paint_poly(painter, pts_lbl, committed=False)
            if self._mouse_pos and pts_lbl:
                painter.setPen(QPen(QColor(100, 200, 255, 160), 1, Qt.DashLine))
                painter.drawLine(pts_lbl[-1], self._mouse_pos)
                if len(pts_lbl) >= 3:
                    fp = pts_lbl[0]
                    if math.hypot(self._mouse_pos.x() - fp.x(),
                                  self._mouse_pos.y() - fp.y()) < 15:
                        painter.setPen(QPen(QColor(255, 240, 60), 2))
                        painter.setBrush(QBrush(QColor(255, 240, 60, 60)))
                        painter.drawEllipse(fp, 12, 12)
        painter.end()

    def _paint_poly(self, painter, pts, committed):
        if not pts:
            return
        if committed:
            fill, stroke, dot, style = (
                QColor(0, 180, 255, 35), QColor(0, 210, 255, 230),
                QColor(40, 220, 255), Qt.SolidLine)
        else:
            fill, stroke, dot, style = (
                QColor(100, 255, 150, 20), QColor(120, 255, 160, 200),
                QColor(140, 255, 170), Qt.DashLine)

        if committed and len(pts) >= 3:
            path = QPainterPath()
            path.moveTo(pts[0])
            for p in pts[1:]:
                path.lineTo(p)
            path.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(fill))
            painter.drawPath(path)

        pen = QPen(stroke, 2, style)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        n = len(pts)
        if committed:
            for i in range(n):
                painter.drawLine(pts[i], pts[(i + 1) % n])
        else:
            for i in range(n - 1):
                painter.drawLine(pts[i], pts[i + 1])

        painter.setBrush(QBrush(dot))
        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        for p in pts:
            painter.drawEllipse(p, 5, 5)