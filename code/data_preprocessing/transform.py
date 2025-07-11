import numpy as np
import pandas as pd
from scipy.stats import gmean

def harmonise_units(df: pd.DataFrame, target_unit: str = 'ppm') -> tuple[pd.DataFrame, str]:
    """
    单位统一：将 wt% 与 ppm 互换

    Args:
        df: 原始 DataFrame，每列需在 attrs 中设置 'unit' 为 'wt%' 或 'ppm'
        target_unit: 目标单位，'ppm' 或 'wt%'
    Returns:
        tuple:
            df_res: 单位转换后的 DataFrame
            code: 状态码，0 正常，1 警告，2 错误
            info: 执行信息
    """
    df_res = df.copy()
    # 构建单位映射：仅包含在 attrs 中已标注单位的列
    unit_map = {col: df_res[col].attrs.get('unit')
                for col in df_res.columns if df_res[col].attrs.get('unit') in ['wt%', 'ppm']}
    for col, unit in unit_map.items():
        # 转换过程使用 to_numeric，遇到非数值自动置 NaN
        series = pd.to_numeric(df_res[col], errors='coerce')
        if unit == 'wt%' and target_unit == 'ppm':
            df_res[col] = series * 10000
            df_res[col].attrs['unit'] = 'ppm'
        elif unit == 'ppm' and target_unit == 'wt%':
            df_res[col] = series / 10000
            df_res[col].attrs['unit'] = 'wt%'
    return df_res, 0, f"所有单位均同一至 {target_unit}"


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
            code: 状态码，0 正常，1 警告，2 错误
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
            return pd.DataFrame(), 2, f"CLR 变换失败: {col} 存在非正值"
        gm = gmean(arr)
        res[f"clr_{col}"] = np.log(arr / gm)
    
    return res, 0, "CLR 变换完成"

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
        df: 对数变换结果 DataFrame
        code: 状态码，0 正常，1 警告，2 错误
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
            return pd.DataFrame(), 2, f"对数变换失败: {col} 存在非正值"
        res[f"log_{col}"] = np.log(arr) / np.log(base)
    return res, 0, "对数变换完成"
