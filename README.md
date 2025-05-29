# 项目说明文档

## 一、Linux (WSL/Ubuntu) 环境

> 使用 Python 3.11 创建并激活 Conda 环境

1. **创建环境**  
   ```bash
   conda create -n py311 python=3.11
   ```

2. **激活环境**  
   - 临时激活：  
     ```bash
     conda activate py311
     ```
   - 添加快捷命令（写入 `~/.bashrc`）：  
     ```bash
     alias py311="conda activate py311"
     ```
   - 生效配置并使用：  
     ```bash
     source ~/.bashrc
     py311
     ```

3. **安装依赖**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 二、Windows 环境

> 使用 Python 3.10.16 创建并激活 Conda 环境

1. **创建环境**  
   ```powershell
   conda create -n py310 python=3.10
   ```

2. **激活环境**  
   ```powershell
   conda activate py310
   ```

3. **安装依赖**  
   ```powershell
   pip install -r requirements_py3.10.txt
   ```

4. **运行 Streamlit 应用**  
   ```powershell
   cd <项目根目录>
   streamlit run main.py
   ```

## 三、项目目录结构及基本说明
```text
📦 data-science-project
├── 📂 code
│   ├── 📂 data_preprocessing
│   │   ├── 📜 calculator.py                                  # 地质元素比例计算
│   │   ├── 📜 loader.py                                      # 地质数据加载(CSV/XLSX)
│   │   ├── 📜 transform.py                                   # 包含CLR/对数变换
│   │   └── 📜 validator.py                                   # 数据验证(单位/异常值)
│   ├── 📂 modeling
│   │   └── 📜 svm_classifier.py                              # SVM分类器实现
│   ├── 📂 modes_result
│   │   ├── 📂 keras_tuner_demo                               # 训练结果模型：超参数调优实验
│   │   ├── 📜 dl_model.pkl                                   # 训练结果模型：深度学习模型
│   │   ├── 📜 rf_model.pkl                                   # 训练结果模型：随机森林模型  
│   │   ├── 📜 svm_model.pkl                                  # 训练结果模型：SVM模型
│   │   └── 📜 xgb_model.pkl                                  # 训练结果模型：XGBoost模型
│   ├── 📂 requirements
│   │   ├── 📜 requirements_for_linux_conda_py311.txt         # linux下python环境依赖文件
│   │   └── 📜 requirements_for_win_conda_py310.txt           # win下python环境依赖文件
│   ├── 📂 result_csv
│   │   ├── 📜 predictions_XGBoost_20250529_013953.csv
│   │   └── 📜 predictions_随机森林_20250529_013946.csv
│   ├── 📂 result_png
│   │   ├── 📜 fig1_scatter_matrix.png                        # 散点矩阵图
│   │   ├── 📜 fig2_heatmap_pearson.png                       # Pearson热力图
│   │   ├── 📜 fig3_heatmap_spearman.png                      # Spearman热力图
│   │   ├── 📜 fig4_pca_biplot.png                            # PCA双标图
│   │   ├── 📜 fig5_ratio_diagrams.png                        # 元素比例图
│   │   ├── 📜 shap_beeswarm.png                              # SHAP蜂群图
│   │   ├── 📜 shap_dependence_clr_SiO2.png                   # SiO2依赖图
│   │   └── 📜 shap_dependence_clr_TFe2O3.png                 # TFe2O3依赖图
│   ├── 📂 script
│   │   ├── 📜 ReadME.md                                      # 脚本说明文件
│   │   ├── 📜 cli_predict_exe_create.bat                     # Windows打包脚本
│   │   ├── 📜 cli_predict_linux_create.sh                    # Linux打包脚本
│   │   ├── 📜 run_for_linux.sh                               # 在linux下python环境下运行脚本
│   │   └── 📜 run_for_win.bat                                # 在win下python环境下运行脚本
│   ├── 📂 visualization
│   │   └── 📜 plot.py                                        # 地质数据可视化
│   ├── 📜 cli_predict.py                                     # 命令行预测接口
│   ├── 📜 cli_predict.spec                                   # PyInstaller配置
│   ├── 📜 cli_predict_win_launch.py                          # Windows启动器
│   ├── 📜 main.py                                            # Streamlit主界面
│   └── 📜 process.ipynb                                      # 数据分析及模型训练笔记
├── 📂 docker
│   ├── 📜 Dockerfile                                         # 容器化配置
│   ├── 📜 package_Dockerfile.sh                              # 镜像构建脚本
├── 📂 executable
│   ├── 📂 cli_predict_exe
│   │   └── 📜 cli_predict.exe                                # Windows可执行文件
│   └── 📜 2025-Project-Data(ESM Table 1).csv                 # 原始地质数据
├── 📂 期末报告
│   └── 📜 24MBD数据科学常用工具期末报告.docx                   # 课程报告
├── 📜 .gitignore
└── 📜 README.md                                              # 项目说明文档

```
## 四、工程运行方式
### 1. 需要环境依赖（Linux\Win Python）
#### run_for_linux.sh
1. 运行环境Linux Conda Python3.11 -直接运行工程
2. 依赖安装:pip install -r '../requirements/requirements_for_linux_conda_py311.txt'
3. 运行指令:在./code/script目录下执行 ./run_for_linux.sh(如果权限有问题：执行chmod +x run_for_linux.sh)

#### run_for_win.bat
1. 运行环境win Conda Python3.10 -直接运行工程
2. 依赖安装:pip install -r '../requirements/requirements_for_win_conda_py310.txt'
3. 运行指令:在./code/script目录下执行 ./run_for_win.bat(如果权限有问题：执行chmod +x run_for_win.bat)

#### 无脚本参与
1. 运行环境win/linux Conda Python
2. 运行目录：在./code 文件夹下执行
3. 方式1：streamlit run main.py启动steamlit服务窗口
4. 方式2：python cli_predict.py启动GUI服务窗口

### 2.无环境依赖
1. 双击提供的EXE可执行程序
2. 注意：启动加载程序根据机器不同需要一定时间，请耐心等待控制终端开始log输出，并弹出GUI界面

## 注意事项

- **Conda 与系统 Python**  
  - 确保命令行已切换到对应环境，避免依赖冲突。  
  - 使用 `which python`（Linux）或 `where python`（Windows）检查当前 Python 路径。

- **依赖版本管理**  
  ```bash
  pip install --upgrade pip
  ```
  若新增或升级依赖，请更新对应的 `requirements*.txt` 文件：
  ```bash
  pip freeze > requirements.txt
  ```
