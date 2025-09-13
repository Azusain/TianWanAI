# PyInstaller spec file for Unified Dataset Tool
# This file defines how to build the executable on Windows.
# Usage:
#   pyinstaller unified_dataset_tool.spec

# Never use emoji or Chinese characters in source codes per project rules.

import os

block_cipher = None

a = Analysis(
    ['tools/unified_dataset_tool/main_window.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # add any non-Python data files here, e.g. icons, ui files, etc.
        # ('tools/unified_dataset_tool/assets', 'assets')
    ],
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtGui',
        'PyQt6.QtCore',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='unified-dataset-tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='tools/unified_dataset_tool/app.ico' if os.path.exists('tools/unified_dataset_tool/app.ico') else None,
)

