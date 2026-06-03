from PySide6.QtWidgets import QPushButton
from gui.fonts import fonts

class BackButton(QPushButton):
    """
    Reusable Back button widget with dark-theme styling, 
    navigating the page controller back to StartPage.
    """
    def __init__(self, controller, parent=None):
        super().__init__("Back to Start Page", parent)
        self.setFont(fonts['medium_font'])
        
        # Connect to controller navigation (imported locally to avoid circular dependencies)
        def on_click():
            from gui.pages.start_page import StartPage
            controller.show_frame(StartPage)
            
        self.clicked.connect(on_click)
        self.setStyleSheet("""
            QPushButton {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
                border-color: #3e3e3e;
            }
            QPushButton:pressed {
                background-color: #3d3d3d;
            }
        """)
