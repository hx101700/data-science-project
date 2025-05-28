import io
import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay

# 导入预处理模块
from data_preprocessing.loader import load_data
from data_preprocessing.transform import clr_transform, harmonise_units, log_transform
from data_preprocessing.validator import (
    flag_outliers,
    verify_dtype,
    verify_units,
    handle_missing_values,
)
# 添加中文以及数学符号支持（支持负号等）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
# 设置 matplotlib 图形在 Tkinter 中显示
plt.switch_backend('TkAgg')
# 设置 matplotlib 图形的 DPI
plt.rcParams['figure.dpi'] = 100  # 设置图形分辨率

# 配置日志
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()

class DataScienceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("data-science-project (Tkinter)")
        self.geometry('1440x900')

        # 初始化状态
        self.step = 1
        self.df = None
        self.df_final = None
        self.models = {}

        # 创建界面
        self._create_widgets()
        # 加载模型
        self._load_models()

    def _create_widgets(self):
        # 分页组件
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)

        # Step1 页面
        self.page1 = ttk.Frame(notebook)
        notebook.add(self.page1, text='Step 1：上传与预处理')
        self._build_step1(self.page1)

        # Step2 页面
        self.page2 = ttk.Frame(notebook)
        notebook.add(self.page2, text='Step 2：模型预测与展示')
        self._build_step2(self.page2)

        # 日志输出区
        log_frame = ttk.LabelFrame(self, text='日志')
        log_frame.pack(fill='x')
        self.log_widget = scrolledtext.ScrolledText(log_frame, height=8, state='disabled')
        self.log_widget.pack(fill='x')

    def _log(self, level, msg):
        # 日志写入widget
        logger_method = getattr(logger, level)
        logger_method(msg)
        self.log_widget.config(state='normal')
        self.log_widget.insert(tk.END, f"{datetime.now():%Y-%m-%d %H:%M:%S} - {level.upper()} - {msg}\n")
        self.log_widget.yview(tk.END)
        self.log_widget.config(state='disabled')

    def _load_models(self):
        # 加载4个模型
        try:
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.abspath(os.path.join(base_dir, './modes_result'))
            for name in ['rf_model', 'dl_model', 'svm_model', 'xgb_model']:
                path = os.path.join(model_dir, f"{name}.pkl")
                self.models[name] = joblib.load(path)
                self._log('info', f"模型 {name} 加载成功")
        except Exception as e:
            self._log('error', f"加载模型失败: {e}")
            messagebox.showerror('错误', f"模型加载失败: {e}")

    def _build_step1(self, parent):
        # 按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)
        upload_btn = ttk.Button(btn_frame, text='选择 CSV/XLS/XLSX 文件', command=self._upload_file)
        upload_btn.pack()
        
        # 创建多标签 Notebook
        self.pp_notebook = ttk.Notebook(parent)
        self.pp_notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # 定义每个预处理步骤的名称
        step_names = [
            '原始数据',
            '缺失值处理',
            'CLR变换',
            '异常值标记',
            '单位统一',
            '对数变换',
            '最终结果'
        ]
        # 为每个步骤创建一个 Frame 并添加到 Notebook，同时保存引用
        self.step_frames = {}
        for name in step_names:
            frame = ttk.Frame(self.pp_notebook)
            self.pp_notebook.add(frame, text=name)
            self.step_frames[name] = frame
            
    def _upload_file(self):
        filetypes = [('CSV 文件', '*.csv'), ('Excel 文件', '*.xls;*.xlsx')]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return
        self._log('info', f"选择文件: {path}")

        try:
            # 读取原始数据
            buffer = io.BytesIO(open(path, 'rb').read())
            buffer.name = os.path.basename(path)
            df_raw, code, msg = load_data(buffer)
            self._log('info', f"load_data: {msg}")
            # 显示“原始数据”
            self._populate_table(self.step_frames['原始数据'], df_raw)

            expected_units = {
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
                'Rb':  'ppm',  'Sr':  'ppm',  'Y':   'ppm',  'Zr':  'ppm',
                'Nb':  'ppm',  'Ba':  'ppm',  'La':  'ppm',  'Ce':  'ppm',
                'Pr':  'ppm',  'Nd':  'ppm',  'Sm':  'ppm',  'Eu':  'ppm',
                'Gd':  'ppm',  'Tb':  'ppm',  'Dy':  'ppm',  'Ho':  'ppm',
                'Er':  'ppm',  'Tm':  'ppm',  'Yb':  'ppm',  'Lu':  'ppm',
                'Hf':  'ppm',  'Ta':  'ppm',  'Th':  'ppm',  'U':   'ppm'
            }
            
            for col, unit in expected_units.items():
                if col in df_raw.columns:
                    df_raw[col].attrs['unit'] = unit
            valid, loc, code, info = verify_dtype(df_raw, {'SiO2':'float64'})
            self._log('info', info)
            valid, loc, code, info = verify_units(df_raw, expected_units)
            self._log('info', info)

            # 缺失值处理
            df_miss, code, info = handle_missing_values(df_raw, method='zero')
            self._log('info', info)
            self._populate_table(self.step_frames['缺失值处理'], df_miss)

            # CLR 变换
            df_clr, code, info = clr_transform(df_raw, list(expected_units.keys())[:10])
            self._log('info', info)
            self._populate_table(self.step_frames['CLR变换'], df_clr)

            # 异常值标记
            df_flag, code, info = flag_outliers(df_clr, z_thresh=3.0)
            self._log('info', info)
            self._populate_table(self.step_frames['异常值标记'], df_flag)

            # 单位统一
            df_ppm, code, info = harmonise_units(df_raw, target_unit='ppm')
            self._log('info', info)
            self._populate_table(self.step_frames['单位统一'], df_ppm)

            # 对数变换(全部列)
            df_log, code, info = log_transform(df_ppm, cols=list(expected_units.keys()),base=10)
            self._log('info', info)
            self._populate_table(self.step_frames['对数变换'], df_log)

            # 最终筛选并展示
            mask = df_flag[[c for c in df_flag.columns if c.startswith('outlier_')]].any(axis=1)
            self.df = df_flag[~mask].drop(columns=[c for c in df_flag.columns if c.startswith('outlier_')])
            self._populate_table(self.step_frames['最终结果'], self.df)
            self.step = 2

        except Exception as e:
            self._log('error', f"预处理失败: {e}")
            messagebox.showerror('错误', f"数据预处理失败: {e}")

    def _populate_table(self, parent, df: pd.DataFrame):
        # 清空 parent 下所有子控件
        for w in parent.winfo_children():
            w.destroy()

        # 设置表格样式, 加粗
        style = ttk.Style()
        style.configure(
            "Treeview.Heading",
            font=('Arial', 10, 'bold'),
        )
        
        # 建立一个 container，用于 grid 管理
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True)

        cols = list(df.columns)
        # Treeview 主表格
        tree = ttk.Treeview(
            container,
            columns=cols,
            show='headings'
        )
        tree.grid(row=0, column=0, sticky='nsew')

        # 设置列头和列宽
        for c in cols:
            tree.heading(c, text=c)  # 这里文本就会自动使用上面配置的加粗字体
            tree.column(c, minwidth=200, anchor='center')

        # 插入数据
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=list(row))

        # 垂直滚动条
        vsb = ttk.Scrollbar(container, orient='vertical', command=tree.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        tree.configure(yscrollcommand=vsb.set)

        # 水平滚动条
        hsb = ttk.Scrollbar(container, orient='horizontal', command=tree.xview)
        hsb.grid(row=1, column=0, sticky='ew')
        tree.configure(xscrollcommand=hsb.set)

        # 让表格区可以随父窗体拉伸
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

    def _build_step2(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(pady=10)
        ttk.Label(control_frame, text='选择模型:').pack(side='left')
        self.model_var = tk.StringVar(value='随机森林')
        ttk.Combobox(control_frame, textvariable=self.model_var,
                     values=['随机森林','深度学习','支持向量机','XGBoost']).pack(side='left', padx=5)
        ttk.Button(control_frame, text='开始预测', command=self._predict).pack(side='left')

        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill='both', expand=True)
        self.fig_canvases = []

        result_frame = ttk.LabelFrame(parent, text='预测结果')
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.result_table = ttk.Treeview(result_frame, columns=[], show='headings')
        self.result_table.pack(fill='both', expand=True)

    def _predict(self):
        if self.step < 2 or self.df is None or self.df.empty:
            messagebox.showinfo('提示', '请先完成 Step 1 上传以及数据预处理')
            return
        self._log('info', '开始预测')

        # 局部拷贝，避免反复在 self.df 上添加“预测结果”列
        df_work = self.df.copy()

        # 真值 y_true_num
        y_true_num = np.where(df_work['Label'] == 'Au-rich PCDs', 0, 1)

        # 删除非特征列：Label、Deposit，以及任何 object 类型列
        df_work = df_work.drop(columns=['Label', 'Deposit'], errors='ignore')
        # 保留纯数值列
        X = df_work.select_dtypes(include=[np.number])

        # 选择模型
        name_map = {
            '随机森林':'rf_model',
            '深度学习':'dl_model',
            '支持向量机':'svm_model',
            'XGBoost':'xgb_model'
        }
        model_key = name_map[self.model_var.get()]
        model = self.models.get(model_key)

        try:
            # 预测概率和原始输出
            if model_key == 'dl_model':
                probas = model.predict(X).ravel()
                preds_raw = (probas > 0.5).astype(int)
            else:
                preds_raw = model.predict(X)
                probas = model.predict_proba(X)[:, 1]

            # 确保 preds_num 是数值型
            if preds_raw.dtype.kind in {'U','S','O'}:
                preds_num = np.where(preds_raw == 'Au-rich PCDs', 0, 1)
            else:
                preds_num = preds_raw.astype(int)

            # 清除旧图并绘制新图
            self._draw_plots(y_true_num, preds_num, probas)

            # 把预测结果映射回字符串，仅用于展示
            df_display = self.df.copy()
            df_display['预测结果'] = np.where(preds_num == 0,
                                              'Au-rich PCDs',
                                              'Cu-rich PCDs')

            # 更新结果表格并保存
            self._populate_result_table(df_display)
            self._save_results(df_display)
            self._log('info','预测完成')

        except Exception as e:
            self._log('error', f"预测失败: {e}")
            messagebox.showerror('错误', f"预测失败: {e}")

    def _draw_plots(self, y, preds, probas):
        for canv in getattr(self, 'fig_canvases', []):
            try:
                canv.get_tk_widget().destroy()
            except:
                pass
        self.fig_canvases = []

        cm = confusion_matrix(y, preds)
        fig1, ax1 = plt.subplots(figsize=(4,4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Au-rich PCDs','Cu-rich PCDs']
        )
        disp.plot(ax=ax1, colorbar=False)
        ax1.set_title('混淆矩阵')
        canvas1 = FigureCanvasTkAgg(fig1, master=self.page2)
        canvas1.get_tk_widget().pack(side='left', padx=5, pady=5)
        canvas1.draw()
        self.fig_canvases.append(canvas1)

        fpr, tpr, _ = roc_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
        ax2.plot([0,1], [0,1], '--')
        ax2.set_title('ROC 曲线')
        ax2.legend()
        canvas2 = FigureCanvasTkAgg(fig2, master=self.page2)
        canvas2.get_tk_widget().pack(side='left', padx=5, pady=5)
        canvas2.draw()
        self.fig_canvases.append(canvas2)

        precision, recall, _ = precision_recall_curve(y, probas)
        fig3, ax3 = plt.subplots(figsize=(4,4))
        ax3.plot(recall, precision, marker='.')
        ax3.set_title('Precision-Recall 曲线')
        canvas3 = FigureCanvasTkAgg(fig3, master=self.page2)
        canvas3.get_tk_widget().pack(side='left', padx=5, pady=5)
        canvas3.draw()
        self.fig_canvases.append(canvas3)

    def _populate_result_table(self, df: pd.DataFrame):
        parent = self.result_table.master

        # —— 1. 清除 parent 下所有子控件 —— #
        for w in parent.winfo_children():
            w.destroy()

        # —— 2. 新建 container —— #
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True)

        # —— 3. 构建新的 Treeview —— #
        cols = list(df.columns)
        tree = ttk.Treeview(container, columns=cols, show='headings')
        tree.grid(row=0, column=0, sticky='nsew')

        # 配置列头与列宽
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, minwidth=200, anchor='center')

        # 插入所有行
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=list(row))

        # —— 垂直滚动条 —— #
        vsb = ttk.Scrollbar(container, orient='vertical', command=tree.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        tree.configure(yscrollcommand=vsb.set)

        # —— 水平滚动条 —— #
        hsb = ttk.Scrollbar(container, orient='horizontal', command=tree.xview)
        hsb.grid(row=1, column=0, sticky='ew')
        tree.configure(xscrollcommand=hsb.set)

        # 让 grid 区域随窗口伸缩
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # 保存引用，以便下一次重建或清除
        self.result_table = tree

    def _save_results(self, df: pd.DataFrame):
        file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                 filetypes=[('CSV文件','*.csv')],
                                                 initialfile=f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv")
        if file_path:
            df.to_csv(file_path, index=False)
            self._log('info', f"预测结果已保存: {file_path}")
        result_dir = os.path.join(os.path.dirname(__file__), 'result_csv')
        os.makedirs(result_dir, exist_ok=True)
        auto_path = os.path.join(result_dir, os.path.basename(file_path) if file_path else f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv")
        df.to_csv(auto_path, index=False)
        self._log('info', f"自动保存至: {auto_path}")

if __name__ == '__main__':
    app = DataScienceApp()
    app.mainloop()