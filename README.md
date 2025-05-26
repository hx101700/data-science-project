执行环境及过程
1. 创建环境：conda create -n py311 python==3.11
2. 激活环境：conda activate py311 ()
   便捷方式：在环境变量中(~/.bashrc)添加别名:alias py311="conda activate py311"
   更新环境变量:source ~/.bashrc;在wls或者ubuntu中执行py311激活环境；
3. 安装依赖库：pip install -r requirements.txt
------------------------------------------------------------------------------------------------------
Windows环境下 Python3.10.16版本
1. 创建环境：conda create -n py310 python==3.10
2. 激活环境：conda activate py310
3. 安装依赖库：pip install -r requirements_py3.10.txt
4. 进入相应目录，使用 streamlit run main.py 运行