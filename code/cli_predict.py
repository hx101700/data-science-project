import sys
print('当前 sys.path:')
for p in sys.path:
    print(p)
try:
    import streamlit
    print('EXE 加载到的 streamlit 路径：', streamlit.__file__)
    print('EXE 加载到的 streamlit 版本：', streamlit.__version__)
except Exception as e:
    print('EXE 加载 streamlit 出错：', e)

import subprocess
import os
import sys
import webbrowser
import time
import socket
def resource_path(relative):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), relative)

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

def find_free_port(start_port=8501, max_port=8600):
    for port in range(start_port, max_port):
        if not is_port_open(port):
            return port
    raise RuntimeError("找不到可用端口！")

if __name__ == "__main__":
    main_py = resource_path("main.py")
    print("main.py real path:", main_py)
    print("main.py exists:", os.path.exists(main_py))

    if os.environ.get("_LAUNCHED_BY_STREAMLIT") == "1":
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", main_py]
        sys.exit(stcli.main())
    else:
        os.environ["BROWSER"] = "none"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["_LAUNCHED_BY_STREAMLIT"] = "1"

        # 自动检测可用端口，避免冲突
        port = find_free_port(8501, 8600)
        print(f"分配的可用端口为: {port}")

        cmd = [
            sys.executable, "-m", "streamlit", "run",
            main_py,
            "--server.port", str(port),
            "--server.headless", "true"
        ]

        print("启动命令：", cmd)
        if sys.platform == "win32":
            p = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            p = subprocess.Popen(cmd)

        # 自动弹一次浏览器
        max_wait = 30
        for i in range(max_wait):
            if is_port_open(port):
                print(f"Streamlit 服务已启动，自动打开浏览器: http://localhost:{port}")
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
