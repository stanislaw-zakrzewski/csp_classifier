from PySide6.QtWidgets import QLabel
from gui.fonts import fonts

class TitleLabel(QLabel):
    """
    Reusable app Title Label with preconfigured fonts and margin styling.
    """
    def __init__(self, text="Kombajn EEG", parent=None):
        super().__init__(text, parent)
        self.setFont(fonts['large_bold_font'])
        self.setStyleSheet("margin: 10px; border: none; color: #ffffff;")
