# -*- mode: python ; coding: utf-8 -*-

import os
import importlib.metadata
from PyInstaller.utils.hooks import collect_submodules

# ----------------- 基础路径配置 -----------------
base_path = os.path.abspath(".")
streamlit_dist_info = importlib.metadata.distribution("streamlit")._path

# ----------------- 自动查找 xgboost 的 DLL -----------------
import xgboost
xgb_root = os.path.dirname(xgboost.__file__)
xgb_dll_path = os.path.join(xgb_root, 'lib', 'xgboost.dll')
xgb_version_file = os.path.join(xgb_root, 'VERSION')
if not os.path.exists(xgb_dll_path):
    raise FileNotFoundError(f"noFound xgboost.dll: {xgb_dll_path}\n请使用 conda 安装 py-xgboost")

block_cipher = None

# ----------------- 分析打包配置 -----------------
a = Analysis(
    ['main.py'],
    pathex=[base_path],
    binaries=[
        (xgb_dll_path, 'xgboost/lib'),  # 自动包含 xgboost.dll
    ],
    datas=[
        (xgb_version_file, 'xgboost'),
        (os.path.join(base_path, '../result/*.pkl'), 'result'),  # 模型文件夹
        (str(streamlit_dist_info), os.path.basename(str(streamlit_dist_info))),  # streamlit 元数据
    ],
    hiddenimports=collect_submodules("sklearn") + collect_submodules("data_preprocessing"),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ----------------- 构建 exe -----------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='data_science_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 可改为 True 查看控制台调试
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='data_science_gui'
)
