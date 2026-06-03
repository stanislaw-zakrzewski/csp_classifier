import math
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPixmap, QPen, QColor
from PySide6.QtCore import Qt, Signal, QPoint

class ElectrodeCanvas(QWidget):
    """
    Unified Electrode Placement Canvas component.
    Supports interactive single electrode selection and multi-electrode highlights.
    """
    electrodeClicked = Signal(str)

    def __init__(self, coordinates, parent=None, interactive=True, limit_to_electrodes=None):
        super().__init__(parent)
        self.pixmap = QPixmap("gui/electrode_placement_filled.png")
        self.coordinates = coordinates
        self.interactive = interactive
        self.limit_to_electrodes = limit_to_electrodes # Optional set/list to restrict interaction/drawing
        
        self.active_electrode = None
        self.highlighted_electrodes = None
        
        if not self.pixmap.isNull():
            self.setFixedSize(self.pixmap.size())

    def set_active_electrode(self, electrode_code):
        self.active_electrode = electrode_code
        self.highlighted_electrodes = None
        self.update()

    def update_electrode_colors(self, highlighted_list):
        self.highlighted_electrodes = set(highlighted_list)
        self.active_electrode = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap.isNull():
            painter.drawPixmap(0, 0, self.pixmap)
            
        r = 35
        for name in self.coordinates:
            if self.limit_to_electrodes is not None and name not in self.limit_to_electrodes:
                continue
                
            x, y = self.coordinates[name]
            
            # Determine circle stroke width based on active state or highlighting
            if self.active_electrode is not None:
                width = 10 if name == self.active_electrode else 5
            elif self.highlighted_electrodes is not None:
                width = 10 if name in self.highlighted_electrodes else 0
            else:
                width = 0
                
            if width > 0:
                pen = QPen(QColor("red"))
                pen.setWidth(width)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPoint(x, y), r, r)

    def mousePressEvent(self, event):
        if not self.interactive:
            return
        pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        r = 35
        for name in self.coordinates:
            if self.limit_to_electrodes is not None and name not in self.limit_to_electrodes:
                continue
                
            x, y = self.coordinates[name]
            dx = pos.x() - x
            dy = pos.y() - y
            if math.sqrt(dx*dx + dy*dy) <= r:
                self.electrodeClicked.emit(name)
                break
