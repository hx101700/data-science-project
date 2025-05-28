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
        # .dist-info部分
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/streamlit-1.45.1.dist-info', 'streamlit-1.45.1.dist-info'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/shap-0.47.2.dist-info', 'shap-0.47.2.dist-info'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/keras-3.10.0.dist-info', 'keras-3.10.0.dist-info'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost-3.0.1.dist-info', 'xgboost-3.0.1.dist-info'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/tensorflow-2.19.0.dist-info', 'tensorflow-2.19.0.dist-info'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/streamlit/static', 'streamlit/static'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/lib/xgboost.dll', 'xgboost/lib'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/VERSION', 'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/__init__.py', 'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/core.py', 'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/tracker.py', 'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/libpath.py', 'xgboost'),
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
    hookspath=[],
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
    console=True,
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
