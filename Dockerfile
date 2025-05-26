FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制本地代码到镜像
COPY . /app

# 安装依赖
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 假设你的主入口是 code/main.py（如用streamlit）
ENTRYPOINT ["streamlit", "run", "code/main.py", "--server.port=8501", "--server.address=0.0.0.0"]