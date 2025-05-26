import os
import sys
import platform

# 项目入口和需要打包的目录/文件
main_script = "run_app.py"
add_data = [
    "code:code",
    "result:result"
]

# 路径分隔符
sep = ";" if platform.system() == "Windows" else ":"


# 构建 pyinstaller 命令
add_data_args = []
for item in add_data:
    add_data_args.extend(["--add-data", item.replace(":", sep)])

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    *add_data_args,
    main_script
]

print("打包命令：")
print(" ".join(cmd))
os.system(" ".join(cmd))