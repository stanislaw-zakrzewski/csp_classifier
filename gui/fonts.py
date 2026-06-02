from PySide6.QtGui import QFont

def create_font(family, size, bold=False):
    f = QFont(family, size)
    if bold:
        f.setBold(True)
    return f

fonts = {
    'large_bold_font': create_font("Segoe UI", 35, True),
    'large_font': create_font("Segoe UI", 35, False),
    'medium_font': create_font("Segoe UI", 20, False),
    'medium_bold': create_font("Segoe UI", 9, True)
}
