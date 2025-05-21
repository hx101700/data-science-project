import numpy as np
import pandas as pd
from scipy.stats import gmean

def clr_transform(df: pd.DataFrame, cols: list[str] = None, safe: bool = True, eps: float = 1e-9) -> tuple[pd.DataFrame, str]:
    """
    CLR 变换: ln(x / gmean(x))z

    Args:
        df: 原始 DataFrame
        cols: 要变换列，为 None 时自动识别所有数值列
        safe: True 时对 x<=0 的值加 eps
        eps: 小常数
    Returns:
        tuple:
            df: CLR 变换结果 DataFrame
            info: 执行信息或错误描述
    """
    if cols is None:
        cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # 创建结果 DataFrame，并首先复制 Label 和 Deposit 列
    res = pd.DataFrame(index=df.index)
    if 'Label' in df.columns:
        res['Label'] = df['Label']
    if 'Deposit' in df.columns:
        res['Deposit'] = df['Deposit']
        
    # 进行 CLR 变换
    for col in cols:
        arr = df[col].astype(float).values
        if safe:
            arr = np.where(arr <= 0, arr + eps, arr)
        if np.any(arr <= 0):
            return pd.DataFrame(), f"CLR 变换失败: {col} 存在非正值"
        gm = gmean(arr)
        res[f"clr_{col}"] = np.log(arr / gm)
    
    return res, "CLR 变换完成"

def log_transform(df: pd.DataFrame, cols: list[str] = None, base: float = np.e, safe: bool = True, eps: float = 1e-9) -> tuple[pd.DataFrame, str]:
    """
    对数变换

    Args:
        df: 原始 DataFrame
        cols: 要变换列，为 None 时自动识别所有数值列
        base: 对数底
        safe: True 时对 x<=0 的值加 eps
        eps: 小常数
    Returns:
        tuple:
            df: 对数变换结果 DataFrame
            info: 执行信息或错误描述
    """
    if cols is None:
        cols = df.select_dtypes(include=['number']).columns.tolist()
    res = pd.DataFrame(index=df.index)
    for col in cols:
        arr = df[col].astype(float).values
        if safe:
            arr = np.where(arr <= 0, arr + eps, arr)
        if np.any(arr <= 0):
            return pd.DataFrame(), f"对数变换失败: {col} 存在非正值"
        res[f"log_{col}"] = np.log(arr) / np.log(base)
    return res, "对数变换完成"
