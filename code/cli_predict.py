import subprocess
import os
import sys
import webbrowser
import time
import socket

def resource_path(relative):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.abspath("."), relative)

def is_port_open(port):
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
    if os.environ.get("_LAUNCHED_BY_STREAMLIT") == "1":
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", resource_path("main.py")]
        sys.exit(stcli.main())
    else:
        os.environ["BROWSER"] = "none"
        os.environ["_LAUNCHED_BY_STREAMLIT"] = "1"
        main_py = resource_path("main.py")
        port = 8501
        cmd = [sys.executable, "-m", "streamlit", "run", main_py, "--server.port", str(port)]

        print("启动命令：", cmd)
        if sys.platform == "win32":
            p = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            p = subprocess.Popen(cmd)
        # 只自动打开一次浏览器
        max_wait = 15
        for i in range(max_wait):
            if is_port_open(port):
                webbrowser.open(f"http://localhost:{port}")
                break
            time.sleep(1)
        else:
            print(f"Streamlit 服务未在 {max_wait} 秒内启动，未能自动打开浏览器，请手动访问 http://localhost:{port}")
        try:
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
        print("服务已关闭，窗口即将退出。")
