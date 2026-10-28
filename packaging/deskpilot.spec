# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
import os

from PyInstaller.utils.hooks import collect_submodules


project = Path(SPECPATH).resolve().parent
if not (project / "run_demo.py").is_file():
    project = project.parent
conda_dll_dir = Path(sys.prefix) / "Library" / "bin"
runtime_dlls = [
    "libcrypto-3-x64.dll", "libssl-3-x64.dll", "sqlite3.dll",
    "liblzma.dll", "LIBBZ2.dll", "ffi.dll", "libexpat.dll",
]
extra_binaries = [
    (str(conda_dll_dir / name), ".") for name in runtime_dlls
    if (conda_dll_dir / name).exists()
]
base_excludes = [
    "pandas", "pyarrow", "sqlalchemy", "pytest", "scipy", "chromadb",
]
if os.getenv("DESKPILOT_BUNDLE_LOCAL_MODELS") != "1":
    base_excludes += ["torch", "transformers", "paddle", "paddleocr", "huggingface_hub"]
analysis = Analysis(
    [str(project / "run_demo.py")],
    pathex=[str(project)],
    binaries=extra_binaries,
    datas=[(str(project / "deskpilot" / "qml"), "deskpilot/qml")],
    hiddenimports=collect_submodules("deskpilot"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=base_excludes,
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [], exclude_binaries=True,
    name="DeskPilot", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
)
coll = COLLECT(
    exe, analysis.binaries, analysis.datas,
    strip=False, upx=False, name="DeskPilot",
)
