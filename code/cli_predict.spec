# -*- mode: python ; coding: utf-8 -*-
block_cipher = None
import os


# ==============================================
# 分析阶段配置
# ==============================================
a = Analysis(
    ['cli_predict.py'],
    pathex   = [os.getcwd()],
    binaries = [],
    datas=[
        ('data_preprocessing', 'data_preprocessing'),
        ('visualization'     , 'visualization'     ),
        ('modes_result'      , 'modes_result'      ),
         # .dist-info部分 不同PC环境重新打包，请修改下面的目录
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/VERSION',         'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/__init__.py',     'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/core.py',         'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/tracker.py',      'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/libpath.py',      'xgboost'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost/lib/xgboost.dll', 'xgboost/lib'),
        ('C:/Users/mobei/.conda/envs/py311/Lib/site-packages/xgboost-3.0.1.dist-info', 'xgboost-3.0.1.dist-info'),

    ],
    hiddenimports=[
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'sklearn', 'sklearn.ensemble', 'sklearn.model_selection',
        'xgboost', 'tensorflow', 'shap', 'tkinter',
        'matplotlib.backends.backend_tkagg',
    ],
    hookspath               = [],
    runtime_hooks           = [],
    excludes                = [],
    win_no_prefer_redirects = False,
    win_private_assemblies  = False,
    cipher                  = block_cipher,
)


# ==============================================
# 构建配置
# ==============================================
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name                      = 'cli_predict',
    debug                     = False,
    bootloader_ignore_signals = True,#允许Ctlr+c中断程序
    strip                     = False,
    upx                       = True,
    console                   = True,
    runtime_tmpdir            = None,#避免临时文件重复解压
    icon                      = None,
    onefile                   = True
)