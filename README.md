# 执行环境与运行流程

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

---

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