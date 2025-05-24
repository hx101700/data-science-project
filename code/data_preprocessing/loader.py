import os
import pandas as pd

def load_data(source) -> tuple[pd.DataFrame, int, str]:
    """
    根据文件后缀加载数据

    Args:
        source: Excel 或 CSV 文件流或者路径
    Returns:
        df: pandas.DataFrame
        code: 状态码，0 正常，1 警告，2 错误
        info: 执行信息，成功或错误描述
    """
    if hasattr(source, "read"):
        # Streamlit 上传的 file-like
        buffer = source
        filename = getattr(source, "name", "")
    else:
        # 本地路径
        buffer = source
        filename = str(source)

    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext in [".xls", ".xlsx"]:
            df = pd.read_excel(buffer)
        elif ext == ".csv":
            df = pd.read_csv(buffer)
        else:
            return pd.DataFrame(), 2, f"不支持的文件格式：{ext}"

        if df.empty:
            return df, 1, f"文件“{filename}”读取成功，但内容为空"
        return df, 0, f"文件“{filename}”读取成功，共 {df.shape[0]} 行，{df.shape[1]} 列"

    except FileNotFoundError:
        return pd.DataFrame(), 2, f"文件未找到：{filename}"
    except Exception as e:
        return pd.DataFrame(), 2, f"读取文件“{filename}”时出错：{e}"