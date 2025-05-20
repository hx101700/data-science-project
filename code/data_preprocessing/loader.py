import os
import pandas as pd

def load_data(filepath: str) -> tuple[pd.DataFrame, str]:
    """
    根据文件后缀加载数据

    Args:
        filepath: Excel 或 CSV 文件路径
    Returns:
        tuple:
            df: pandas.DataFrame
            info: 执行信息，成功或错误描述
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in ['.xls', '.xlsx']:
            return pd.read_excel(filepath), f"Excel 文件 {filepath} 读取成功"
        elif ext == '.csv':
            return pd.read_csv(filepath), f"CSV 文件 {filepath} 读取成功"
        else:
            return pd.DataFrame(), f"不支持的文件格式: {ext}"
    except FileNotFoundError:
        return pd.DataFrame(), f"文件未找到: {filepath}"
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), f"文件 {filepath} 为空"
    except Exception as e:
        return pd.DataFrame(), f"读取文件 {filepath} 时发生错误: {e}"