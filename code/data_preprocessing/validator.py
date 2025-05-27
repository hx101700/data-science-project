import pandas as pd
from pandas.api.types import is_dtype_equal, pandas_dtype
import numpy as np
from scipy import stats

def verify_dtype(
    df: pd.DataFrame,
    expected_schema: dict[str, any]
) -> tuple[bool, dict[str, dict[str, str]], str]:
    """
    按给定的 schema 校验 DataFrame 中各列的 dtype 是否与预期一致。

    Args:
        df: 原始 DataFrame
        expected_schema: 列名到预期 dtype 的映射，例如
            {
                "SiO2": "float64",
                "TiO2": "float64",
                "Count": "int64",
                "Date": "datetime64[ns]",
                "Category": "category"
            }

    Returns:
        valid: 如果所有列 dtype 都符合预期返回 True，否则 False
        mismatch: {列名: {"actual": 实际 dtype, "expected": 预期 dtype}}
        code: 状态码，0 正常，1 警告，2 错误
        info: 执行结果描述
    """
    mismatch: dict[str, dict[str, str]] = {}

    for col, exp in expected_schema.items():
        if col not in df.columns:
            # 列不存在也算不匹配
            mismatch[col] = {"actual": "缺失", "expected": str(exp)}
            continue

        actual_dtype = df[col].dtype
        # 统一把预期 dtype 转成 pandas dtype
        exp_dtype = pandas_dtype(exp)

        # 精确比较两个 dtype
        if not is_dtype_equal(actual_dtype, exp_dtype):
            mismatch[col] = {
                "actual": str(actual_dtype),
                "expected": str(exp_dtype)
            }

    if mismatch:
        return False, mismatch, 1, f"共发现 {len(mismatch)} 列 dtype 与预期不符"
    else:
        return True, {}, 0, "所有列 dtype 与预期一致"

def verify_units(
    df: pd.DataFrame,
    expected_units: dict[str, str]
) -> tuple[bool, dict[str, dict[str, str]], str]:
    """
    按给定的 units 映射校验 DataFrame 中各列的单位是否与预期一致。
    
    Args:
        df: 原始 DataFrame
        expected_units: 列名到预期单位的映射，例如:
            {"SiO2": "wt%", "Rb": "ppm"}

    Returns:
        valid: 如果所有列单位都符合预期返回 True，否则 False
        mismatch: {列名: {"actual": 实际单位, "expected": 预期单位}}
        code: 状态码，0 正常，1 警告，2 错误
        info: 执行结果描述
    """
    mismatch: dict[str, dict[str, str]] = {}
    for col, exp_unit in expected_units.items():
        if col not in df.columns:
            mismatch[col] = {"actual": "<missing>", "expected": exp_unit}
            continue
        actual_unit = df[col].attrs.get('unit', None)
        if actual_unit != exp_unit:
            mismatch[col] = {
                "actual": str(actual_unit),
                "expected": exp_unit
            }
    if mismatch:
        return False, mismatch, 1, f"共发现 {len(mismatch)} 列 unit 与预期不符"
    return True, {}, 0, "所有列 unit 与预期一致"

def handle_missing_values(df: pd.DataFrame, cols: list[str] = None, method: str = 'zero') -> tuple[pd.DataFrame, str]:
    """
    处理缺失值：针对 CLR 预处理，默认使用零替换

    Args:
        df: 原始 DataFrame
        cols: 要处理的列名列表，为 None 时自动识别所有数值列
        method: 'zero' 用 0 替换, 'median' 用中位数替换
    Returns:
        df: 处理后的 DataFrame
        code: 状态码，0 正常，1 警告，2 错误
        info: 执行信息
    """
    df_proc = df.copy()
    # 自动识别数值列
    num_cols = df_proc.select_dtypes(include=['number']).columns.tolist()
    # 强制尝试将所有列转换为数值，非数值置 NaN
    for col in num_cols:
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
    if cols is None:
        cols = num_cols
    if method not in ['zero', 'median']:
        return df_proc, 1, f"不支持的缺失值处理方法: {method}"
    for col in cols:
        if method == 'zero':
            df_proc[col] = df_proc[col].fillna(0)
        else:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
    return df_proc, 0, f"缺失值处理完成，使用方法: {method}"

def flag_outliers(df: pd.DataFrame, cols: list[str] = None, z_thresh: float = 3.0) -> tuple[pd.DataFrame, str]:
    """
    标记异常值：使用 Z 分数法，对超过阈值的样本进行标记

    Args:
        df: 原始 DataFrame
        cols: 要检测的列名列表，为 None 时自动识别所有数值列
        z_thresh: Z 分数阈值
    Returns:
        tuple:
            df: 带 'outlier_<col>' 标志列的 DataFrame
            code: 状态码，0 正常，1 警告，2 错误
            info: 执行信息
    """
    df_proc = df.copy()
    if cols is None:
        cols = df_proc.select_dtypes(include=['number']).columns.tolist()
    for col in cols:
        vals = df_proc[col].astype(float)
        z = np.abs(stats.zscore(vals, nan_policy='omit'))
        df_proc[f"outlier_{col}"] = z > z_thresh
    return df_proc, 0, f"异常值标记完成 (阈值: {z_thresh})"