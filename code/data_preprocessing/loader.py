import os
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    根据文件后缀加载数据

    Args:
        filepath: Excel 或 CSV 文件路径
    Returns:
        pandas.DataFrame
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
            info = f"Excel文件 {filepath} 读取成功"
        elif ext == '.csv':
            df = pd.read_csv(filepath)
            info = f"CSV文件 {filepath} 读取成功"
        else:
            df = pd.DataFrame()
            info = f'不支持的文件格式: {ext}'
    except FileNotFoundError:
        df = pd.DataFrame()
        info = f"文件未找到: {filepath}"
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
        info = f"文件 {filepath} 为空"
    except Exception as e:
        df = pd.DataFrame()
        info = f"读取文件 {filepath} 时发生错误: {e}"
    return df, info