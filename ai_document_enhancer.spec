# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec file for AI Document Enhancement System.
#
# Usage:
#   pyinstaller ai_document_enhancer.spec            (one-directory build)
#   pyinstaller ai_document_enhancer.spec --onefile   (single .exe build)
#
# The spec bundles all SVG icons and JSON templates from resources/ into the
# executable so the application works without the source tree.

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

PROJECT_ROOT = Path(SPECPATH)
RESOURCES_DIR = PROJECT_ROOT / "resources"

# ── Data files to bundle ────────────────────────────────────────────
# Each tuple is (source_path, dest_directory_in_bundle).
datas = [
    (str(RESOURCES_DIR / "icons"), os.path.join("resources", "icons")),
    (str(RESOURCES_DIR / "templates"), os.path.join("resources", "templates")),
]

# Collect transformer model config files that PyInstaller misses
datas += collect_data_files("transformers", include_py_files=False)

# ── Hidden imports ──────────────────────────────────────────────────
# Packages whose submodules are imported dynamically or lazily and are
# not detected by PyInstaller's static analysis.
hiddenimports = [
    # PySide6 SVG plugin (needed for sidebar icons)
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",

    # scikit-learn internals used by TF-IDF / cosine similarity
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._typedefs",
    "sklearn.neighbors._partition_nodes",
    "sklearn.tree._utils",
    "sklearn.feature_extraction.text",

    # sumy summariser backends
    "sumy.parsers.plaintext",
    "sumy.nlp.tokenizers",
    "sumy.summarizers.lsa",

    # textstat readability
    "textstat",

    # Torch / transformers (DistilBART, T5)
    "torch",
    "transformers",

    # python-docx
    "docx",

    # fpdf2 PDF export
    "fpdf",

    # pytesseract
    "pytesseract",

    # pdf2image
    "pdf2image",

    # language_tool_python (grammar)
    "language_tool_python",

    # numpy / Pillow / OpenCV
    "numpy",
    "PIL",
    "cv2",
]

# Collect all submodules for packages that heavily use lazy imports
hiddenimports += collect_submodules("sklearn")

# ── Excluded modules (reduce bundle size) ───────────────────────────
excludes = [
    "tkinter",
    "matplotlib",
    "notebook",
    "IPython",
    "scipy.spatial.cKDTree",
]

# ── Analysis ────────────────────────────────────────────────────────
a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── EXE (one-directory mode by default) ─────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIDocumentEnhancer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed application — no terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(RESOURCES_DIR / "icons" / "app_icon.ico"),
)

# ── COLLECT (gather into dist/AIDocumentEnhancer/) ──────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AIDocumentEnhancer",
)
