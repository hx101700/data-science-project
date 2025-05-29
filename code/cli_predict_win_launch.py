import os
import subprocess
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
    if os.environ.get("_LAUNCHED_BY_STREAMLIT") == "1":
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", main_py]
        sys.exit(stcli.main())
    else:
        os.environ["BROWSER"] = "none"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["_LAUNCHED_BY_STREAMLIT"] = "1"

        port = find_free_port(8501, 8600)
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            main_py,
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        if sys.platform == "win32":
            p = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            p = subprocess.Popen(cmd)
        for i in range(30):
            if is_port_open(port):
                webbrowser.open(f"http://localhost:{port}")
                break
            time.sleep(1)
        try:
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
