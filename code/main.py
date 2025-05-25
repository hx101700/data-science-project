import io
import streamlit as st
import pandas as pd
from data_preprocessing.loader import load_data

# 加载所有模型
import joblib
rf_model = joblib.load("./result/rf_model.pkl")
dl_model = joblib.load("./result/dl_model.pkl")
svm_model = joblib.load("./result/svm_model.pkl")
xgb_model = joblib.load("./result/xgb_model.pkl")

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
            # 1. 定义单位映射
            expected_units = {
                # Major oxides (wt%)
                'SiO2':  'wt%',
                'TiO2':  'wt%',
                'Al2O3':'wt%',
                'TFe2O3':'wt%',
                'MnO':   'wt%',
                'MgO':   'wt%',
                'CaO':   'wt%',
                'Na2O':  'wt%',
                'K2O':   'wt%',
                'P2O5':  'wt%',
                # Trace elements (ppm)
                'Rb':  'ppm',  'Sr':  'ppm',  'Y':   'ppm',  'Zr':  'ppm',
                'Nb':  'ppm',  'Ba':  'ppm',  'La':  'ppm',  'Ce':  'ppm',
                'Pr':  'ppm',  'Nd':  'ppm',  'Sm':  'ppm',  'Eu':  'ppm',
                'Gd':  'ppm',  'Tb':  'ppm',  'Dy':  'ppm',  'Ho':  'ppm',
                'Er':  'ppm',  'Tm':  'ppm',  'Yb':  'ppm',  'Lu':  'ppm',
                'Hf':  'ppm',  'Ta':  'ppm',  'Th':  'ppm',  'U':   'ppm'
            }

            # 2. 为每一列设置 attrs['unit']
            for col, unit in expected_units.items():
                if col in df.columns:
                    df[col].attrs['unit'] = unit

            # 3. 验证设置是否成功
            for col in expected_units:
                if col in df.columns:
                    print(f"{col}: unit = {df[col].attrs.get('unit')}")
                    
            from data_preprocessing.validator import verify_dtype

            valid, error_location, code, info = verify_dtype(df, {"SiO2": "float64",})
            
            from data_preprocessing.validator import verify_units

            valid, error_location, code, info = verify_units(df, expected_units)
            
            from data_preprocessing.validator import handle_missing_values

            df_miss, code, info = handle_missing_values(df, method='zero')
            
            from data_preprocessing.transform import clr_transform
            df_clr, code, info = clr_transform(df, ["SiO2", "TiO2", "Al2O3", "TFe2O3", "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5"])
            
            from data_preprocessing.validator import flag_outliers

            #
            # 少可视化！
            #
            
            df_proc, code, info = flag_outliers(df_clr, z_thresh=3.0)
            # 找出所有 outlier 列名
            outlier_cols = [c for c in df_proc.columns if c.startswith('outlier_')]
            # 如果任意一列为 True，就保留该行
            mask = df_proc[outlier_cols].any(axis=1)
            # 应用掩码
            outliers = df_proc[mask]
            
            from data_preprocessing.transform import harmonise_units
            # 把所有 unit='wt%' 的列统一转换为 ppm
            df_ppm, _, info = harmonise_units(df, target_unit='ppm')
            
            from data_preprocessing.transform import log_transform
            df_log, _, linfo = log_transform(df_ppm, cols=['SiO2', 'Rb'], base=10)

            # 去除异常值
            df_finial = df_proc[~mask]
            df_finial = df_finial.drop(columns=outlier_cols)
            
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
        # 选择模型
        model_choice = st.selectbox(
            label="请选择模型",
            options=["随机森林", "深度学习", "支持向量机", "XGBoost"],
            index=0,
        )
        # 预测按钮
        if st.button("开始预测"):
            if model_choice == "随机森林":
                model = rf_model
            elif model_choice == "深度学习":
                model = dl_model
            elif model_choice == "支持向量机":
                model = svm_model
            elif model_choice == "XGBoost":
                model = xgb_model
            else:
                st.error("未知模型")
                model = None
            
            if model:
                with st.spinner("预测中…"):
                    try:
                        X = df_finial.drop(['Label', 'Deposit'], axis=1)
                        y = df_finial['Label']
                        predictions = model.predict(X)
                        df_finial["预测结果"] = predictions
                        st.success("预测完成！")
                        st.dataframe(df_finial, use_container_width=True)
                    except Exception as e:
                        st.error(f"预测失败：{e}")
