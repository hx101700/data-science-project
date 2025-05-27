# -- coding: utf-8 --
block_cipher = None
import os
a = Analysis(
    ['cli_predict.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('main.py', '.'),
        ('data_preprocessing', 'data_preprocessing'),
        ('modeling', 'modeling'),
        ('visualization', 'visualization'),
        ('modes_result', 'modes_result'),
    ],
    hiddenimports=[
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'sklearn', 'sklearn.ensemble', 'sklearn.model_selection', 'sklearn.preprocessing', 
        'sklearn.metrics', 'sklearn.svm', 'sklearn.decomposition', 'sklearn.utils',
        'joblib',
        'xgboost',
        'tensorflow', 'tensorflow.keras', 'tensorflow.keras.models',
        'tensorflow.keras.layers', 'tensorflow.keras.utils', 'tensorflow.keras.optimizers',
        'tensorflow.keras.callbacks', 'keras_tuner', 'keras_tuner.engine', 'keras_tuner.tuners',
        'shap',
        'streamlit',
    ],
    hookspath=['./hooks'],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cli_predict',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  #False（隐藏控制台）
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cli_predict'
)
