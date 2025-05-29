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
        menubar = tk.Menu(self)

        # “文件”菜单：打开/上传、保存预测结果、退出
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="打开/上传数据…",
            accelerator="Ctrl+O",
            command=self._upload_file
        )
        file_menu.add_command(
            label="保存预测结果",
            accelerator="Ctrl+S",
            command=self._menu_save_results
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="退出",
            accelerator="Ctrl+Q",
            command=self.destroy
        )
        menubar.add_cascade(label="文件", menu=file_menu)

        # “帮助”菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="使用说明",
            accelerator="F1",
            command=self._show_help
        )
        help_menu.add_command(
            label="关于",
            command=self._show_about
        )
        menubar.add_cascade(label="帮助", menu=help_menu)

        # 绑定到窗口并设置快捷键
        self.config(menu=menubar)
        self.bind_all("<Control-o>", lambda e: self._upload_file())
        self.bind_all("<Control-s>", lambda e: self._menu_save_results())
        self.bind_all("<Control-q>", lambda e: self.destroy())
        self.bind_all("<F1>", lambda e: self._show_help())

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

    def _menu_save_results(self):
        if not hasattr(self, 'df_display') or self.df_display is None or self.df_display.empty or self.df is None or self.df.empty:
            messagebox.showwarning('提示', '没有可保存的预测结果')
        else:
            self._save_results(self.df_display)
            
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
        self.pp_notebook = ttk.Notebook(parent)
        self.pp_notebook.pack(fill='both', expand=True, padx=10, pady=5)
        step_names = [
            '原始数据',
            '缺失值处理',
            'CLR变换',
            '异常值标记',
            '单位统一',
            '对数变换',
            '最终结果'
        ]
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

        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.pack(fill='both', expand=True)

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

        # 局部拷贝，避免修改原始 self.df
        df_work = self.df.copy()

        # 真实标签数值化：Au-rich PCDs→0, Cu-rich PCDs→1
        y_true_num = np.where(df_work['Label'] == 'Au-rich PCDs', 0, 1)

        # 删除非特征列，并仅保留数值型特征
        df_work = df_work.drop(columns=['Label', 'Deposit'], errors='ignore')
        X = df_work.select_dtypes(include=[np.number])

        # 选择模型
        name_map = {
            '随机森林': 'rf_model',
            '深度学习': 'dl_model',
            '支持向量机': 'svm_model',
            'XGBoost': 'xgb_model'
        }
        model_key = name_map[self.model_var.get()]
        model = self.models.get(model_key)

        try:
            # 预测概率与原始输出
            if model_key == 'dl_model':
                probas = model.predict(X).ravel()
                preds_raw = (probas > 0.5).astype(int)
            else:
                preds_raw = model.predict(X)
                probas = model.predict_proba(X)[:, 1]

            # 确保 preds_num 数值化
            if preds_raw.dtype.kind in {'U', 'S', 'O'}:
                preds_num = np.where(preds_raw == 'Au-rich PCDs', 0, 1)
            else:
                preds_num = preds_raw.astype(int)

            # 清除旧图并绘制新图
            self._draw_plots(y_true_num, preds_num, probas)

            # 构造展示用 DataFrame，并存入实例属性
            df_display = self.df.copy()
            df_display['预测结果'] = np.where(
                preds_num == 0, 'Au-rich PCDs', 'Cu-rich PCDs'
            )
            self.df_display = df_display

            # 更新结果表格（不自动保存）
            self._populate_result_table(df_display)
            self._log('info', '预测完成，结果可通过“文件→保存预测结果”导出')

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

        for w in self.plot_frame.winfo_children():
            w.destroy()
        inner = ttk.Frame(self.plot_frame)
        inner.pack(expand=True)

        cm = confusion_matrix(y, preds)
        fig1, ax1 = plt.subplots(figsize=(4,4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Au-rich PCDs','Cu-rich PCDs']
        )
        disp.plot(ax=ax1, colorbar=False)
        ax1.set_title('混淆矩阵')
        canvas1 = FigureCanvasTkAgg(fig1, master=inner)
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
        canvas2 = FigureCanvasTkAgg(fig2, master=inner)
        canvas2.get_tk_widget().pack(side='left', padx=5, pady=5)
        canvas2.draw()
        self.fig_canvases.append(canvas2)
        
        precision, recall, _ = precision_recall_curve(y, probas)
        fig3, ax3 = plt.subplots(figsize=(4,4))
        ax3.plot(recall, precision, marker='.')
        ax3.set_title('Precision-Recall 曲线')
        canvas3 = FigureCanvasTkAgg(fig3, master=inner)
        canvas3.get_tk_widget().pack(side='left', padx=5, pady=5)
        canvas3.draw()
        self.fig_canvases.append(canvas3)

    def _populate_result_table(self, df: pd.DataFrame):
        parent = self.result_table.master

        for w in parent.winfo_children():
            w.destroy()

        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True)

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

        vsb = ttk.Scrollbar(container, orient='vertical', command=tree.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(container, orient='horizontal', command=tree.xview)
        hsb.grid(row=1, column=0, sticky='ew')
        tree.configure(xscrollcommand=hsb.set)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.result_table = tree

    def _save_results(self, df: pd.DataFrame):
        if df.empty:
            messagebox.showwarning('警告', '没有可保存的预测结果')
            return
        # 获取并格式化模型名称，用于文件名
        model_name = self.model_var.get().replace(' ', '_')  # e.g. "随机森林" -> "随机森林"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"predictions_{model_name}_{timestamp}.csv"

        # 弹出“另存为”对话框，默认文件名包含模型名
        file_path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV 文件', '*.csv')],
            initialfile=default_name
        )
        if file_path:
            df.to_csv(file_path, index=False)
            self._log('info', f"预测结果已保存: {file_path}")

        # 自动保存到本地 result_csv 文件夹
        result_dir = os.path.join(os.path.dirname(__file__), 'result_csv')
        os.makedirs(result_dir, exist_ok=True)
        # 如果用户取消了另存，则使用 default_name
        auto_filename = os.path.basename(file_path) if file_path else default_name
        auto_path = os.path.join(result_dir, auto_filename)
        df.to_csv(auto_path, index=False)
        self._log('info', f"自动保存至: {auto_path}")
        
    def _show_help(self):
        help_text = (
            "使用说明：\n"
            "1. 数据上传：\n"
            "   在主菜单 “文件” → “打开/上传数据…”（支持 .xlsx/.xls/.csv，快捷键 Ctrl+O），加载地球化学数据。\n"
            "2. Step 1：预处理流程\n"
            "   • 标签页依次展示：\n"
            "     – 原始数据\n"
            "     – 缺失值处理\n"
            "     – CLR 变换\n"
            "     – 异常值标记\n"
            "     – 单位统一\n"
            "     – 对数变换\n"
            "     – 最终结果\n"
            "   • 可在各标签页中查看对应步骤的中间 DataFrame。\n"
            "3. Step 2：模型预测与展示\n"
            "   • 在“选择模型”下拉框中选定模型（随机森林/深度学习/支持向量机/XGBoost），点击“开始预测”按钮。\n"
            "   • 窗口会分别展示：混淆矩阵、ROC 曲线、Precision-Recall 曲线。\n"
            "   • 下方“预测结果”表格显示所有样本的预测标签。\n"
            "4. 结果保存：\n"
            "   • 使用主菜单 “文件” → “保存预测结果”（快捷键 Ctrl+S）导出 CSV。\n"
            "   • 自动保存至程序目录下的 result_csv 文件夹。\n"
            "5. 日志查看：\n"
            "   • 底部日志区域实时记录处理与预测过程，便于调试和追踪。\n"
            "6. 快捷键：\n"
            "   • Ctrl+O：打开/上传数据\n"
            "   • Ctrl+S：保存预测结果\n"
            "   • Ctrl+Q：退出程序\n"
            "   • F1：查看使用说明\n"
        )
        messagebox.showinfo("使用帮助", help_text)

    def _show_about(self):
        """
        弹出一个对话框，展示本应用的依赖库及用途说明
        """
        about_text = (
            "data-science-project (Tkinter) v1.0\n"
            "作者：黄熙 李杰相 彭嘉男 吴百恒\n"
            "\n"
            "主要使用库：\n"
            "• Python 3.11\n"
            "• Tkinter（GUI 界面）\n"
            "• pandas, numpy（数据处理）\n"
            "• matplotlib（可视化）\n"
            "• scikit-learn（机器学习模型）\n"
            "• joblib（模型序列化/反序列化）\n"
            "\n"
            "本工具用于地球化学数据的批量预处理及矿床类型预测：\n"
            "• 对上传的 CSV/XLS/XLSX 文件执行缺失值填充、CLR 变换、异常值检测、单位换算、对数变换等步骤\n"
            "• 可视化每一步的数据中间结果，方便用户检查和调试\n"
            "• 提供随机森林、深度学习、支持向量机、XGBoost 四种模型，可分别完成 Au-rich/Cu-rich PCDs 分类预测\n"
            "• 支持混淆矩阵、ROC 曲线、Precision-Recall 曲线可视化，并导出带有模型名称和时间戳的预测结果\n"
        )
        messagebox.showinfo("关于", about_text)
        
if __name__ == '__main__':
    # 显示启动提示（仅打包后生效）
    if getattr(sys, 'frozen', False):
        print("="*50)
        print("程序正在初始化，请耐心等待...")
        print("="*50)
        sys.stdout.flush()  # 确保立即输出
    app = DataScienceApp()
    app.mainloop()