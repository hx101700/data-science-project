import subprocess
import os
import sys
import webbrowser
import time
import socket
import traceback

def resource_path(relative: str) -> str:
    """
    获取资源文件的绝对路径：
    - 打包后：sys._MEIPASS 指向临时解压目录
    - 开发环境：使用脚本启动参数 argv[0] 的目录
    """
    try:
        if hasattr(sys, '_MEIPASS'):
            base = sys._MEIPASS
        else:
            base = os.path.dirname(os.path.abspath(sys.argv[0]))
        full = os.path.join(base, relative)
        return os.path.normpath(full)
    except Exception as e:
        print("resource_path 解析失败：", e)
        traceback.print_exc()
        sys.exit(1)

def is_port_open(port: int) -> bool:
    """检测 localhost:port 是否可连通"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(('localhost', port))
        return True
    except Exception:
        return False
    finally:
        s.close()

if __name__ == "__main__":
    try:
        # Streamlit 内部重启时的分支
        if os.environ.get("_LAUNCHED_BY_STREAMLIT") == "1":
            import streamlit.web.cli as stcli
            target = resource_path("main.py")
            print(f"Streamlit 内部调用，运行：streamlit run {target}")
            sys.argv = ["streamlit", "run", target]
            sys.exit(stcli.main())

        # 外部脚本启动分支
        os.environ["BROWSER"] = "none"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["_LAUNCHED_BY_STREAMLIT"] = "1"

        # 确定 main.py 路径
        main_py = resource_path("main.py")
        if not os.path.isfile(main_py):
            print(f"找不到 main.py，请检查路径是否正确：{main_py}")
            sys.exit(1)
        print(f"定位到 main.py：{main_py}")

        # 构造启动命令
        port = 8501
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            main_py,
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        print("启动命令：", " ".join(cmd))

        # 启动子进程并捕获 stdout/stderr
        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                # Windows 下若需隐藏黑框，可取消下一行注释，但可能会屏蔽部分输出：
                # creationflags=subprocess.CREATE_NO_WINDOW
            )
        except Exception as e:
            print("无法启动子进程：", e)
            traceback.print_exc()
            sys.exit(1)

        # 自动轮询端口，等服务首次就绪时打开浏览器
        max_wait = 30
        print(f"等待 Streamlit 服务在 localhost:{port} 启动（最多 {max_wait}s）…")
        for _ in range(max_wait):
            if is_port_open(port):
                print(f"服务已监听端口 {port}，打开浏览器")
                webbrowser.open(f"http://localhost:{port}")
                break
            time.sleep(1)
        else:
            print(f"服务未在 {max_wait}s 内启动，可能启动失败或较慢，请手动访问 http://localhost:{port}")

        # 打印子进程输出，方便排查运行时错误
        if p.stdout:
            print("----- Streamlit 输出日志开始 -----")
            for line in p.stdout:
                print(line.rstrip("\n"))
            print("----- Streamlit 输出日志结束 -----")

        # 等待子进程结束
        try:
            exit_code = p.wait()
            if exit_code != 0:
                print(f"Streamlit 进程退出，返回码：{exit_code}")
            else:
                print("Streamlit 进程已正常退出")
        except KeyboardInterrupt:
            print("检测到 Ctrl+C，正在终止 Streamlit 服务…")
            p.terminate()
            p.wait()
            print("服务已关闭")

    except Exception as e:
        print("未捕获异常：", e)
        traceback.print_exc()
        sys.exit(1)
