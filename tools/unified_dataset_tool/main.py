#!/usr/bin/env python3
"""
unified dataset tool - a comprehensive dataset management application

this tool provides:
- dataset analysis with detailed statistics
- dataset splitting for train/validation
- sample visualization with bounding boxes
- support for yolo format annotations
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_window import MainWindow

def main():
    # create application
    app = QApplication(sys.argv)
    
    # set application properties
    app.setApplicationName("unified dataset tool")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("azusaing")
    
    # enable high dpi support
    # app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    # app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # create and show main window
    window = MainWindow()
    window.show()
    
    # run application
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
