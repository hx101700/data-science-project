import io
import streamlit as st
import pandas as pd
from data_preprocessing.loader import load_data

# 页面配置
st.set_page_config(
    page_title="data-science-project",
    layout="wide",
)
st.title("data-science-project")

# 初始化会话状态
if "step" not in st.session_state:
    st.session_state.step = 1

# 缓存读取函数
@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda buf: hash(buf.getvalue())})
def load_file(buffer: io.BytesIO):
    return load_data(buffer)

# Step 1：上传并验证数据
with st.expander("Step 1：上传测试文件", expanded=(st.session_state.step == 1)):
    uploaded = st.file_uploader(
        label="请选择 CSV / XLS / XLSX 文件",
        type=["csv", "xls", "xlsx"],
    )
    if uploaded:
        with st.spinner("加载中…"):
            buf = io.BytesIO(uploaded.getvalue()); buf.name = uploaded.name
            df, code, msg = load_file(buf)
        if code == 0:
            st.success(msg)
        elif code == 1:
            st.warning(msg)
        else:
            st.error(msg)
        if code == 0 and not df.empty:
            st.dataframe(df, use_container_width=True)
            st.session_state.step = 2

with st.expander("Step 2：选择模型并预测", expanded=(st.session_state.step == 2)):
    if st.session_state.step < 2:
        st.info("请先完成 Step 1")
    else:
        st.write("TODO")
