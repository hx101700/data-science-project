import pandas as pd

def compute_ratios(df: pd.DataFrame, ratios: list[tuple[str, str]]) -> tuple[pd.DataFrame, str]:
    """
    批量计算地球化学比值

    Args:
        df: 原始 DataFrame
        ratios: 列表，元素为 (numerator, denominator) 元组
    Returns:
        df: 包含 ratio_<num>_<den> 列的 DataFrame
        code: 状态码，0 正常，1 警告，2 错误
        info: 执行信息或错误描述
    """
    out = pd.DataFrame(index=df.index)
    for num, den in ratios:
        if num not in df.columns or den not in df.columns:
            return pd.DataFrame(), 2, f"比值计算失败: 缺少列 {num} 或 {den}"
        denom = df[den].astype(float)
        if any(denom == 0):
            return pd.DataFrame(), 2, f"比值计算失败: {den} 存在零值"
        out[f"ratio_{num}_{den}"] = df[num].astype(float) / denom
    return out, 0, "比值计算完成"