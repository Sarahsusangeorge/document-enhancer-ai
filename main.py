"""Entry point for the AI Document Enhancement System."""

import sys
import logging
from pathlib import Path

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    from PySide6.QtCore import Qt
except ImportError:
    print(
        "PySide6 is required but not installed.\n"
        "Install it with:  pip install PySide6>=6.6.0"
    )
    sys.exit(1)

from app.ui.main_window import MainWindow
from app.ui.styles import get_theme


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting AI Document Enhancement System")

    app = QApplication(sys.argv)
    app.setApplicationName("AI Document Enhancement System")
    app.setOrganizationName("SE_Project")

    icon_path = Path(__file__).resolve().parent / "resources" / "icons" / "app_icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    app.setStyleSheet(get_theme("dark"))

    window = MainWindow()
    window.show()

    logger.info("Application window displayed")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
