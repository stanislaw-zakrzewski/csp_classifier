from PySide6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView
from PySide6.QtCore import Qt

def open_annotation_viewer(parent, edf_path, annotations):
    viewer = AnnotationViewer(parent, edf_path, annotations)
    viewer.exec()

class AnnotationViewer(QDialog):
    def __init__(self, parent, edf_path, annotations):
        super().__init__(parent)
        self.setWindowTitle(f"Browse annotations for {edf_path}")
        self.resize(500, 500)
        
        layout = QVBoxLayout(self)
        
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['Start (s)', 'Length (s)', 'Label'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        table.setRowCount(len(annotations))
        for row_idx, annotation in enumerate(annotations):
            table.setItem(row_idx, 0, QTableWidgetItem(str(annotation[0])))
            table.setItem(row_idx, 1, QTableWidgetItem(str(annotation[1])))
            table.setItem(row_idx, 2, QTableWidgetItem(str(annotation[2])))
            
        layout.addWidget(table)
