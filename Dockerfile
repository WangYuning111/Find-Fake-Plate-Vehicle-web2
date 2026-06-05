FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p static/uploads static/results data

# 暴露端口
EXPOSE 8090

# 使用 Gunicorn 运行（生产环境）
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8090", "--timeout", "120", "app:app"]
