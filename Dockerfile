# 多阶段构建Dockerfile - 支持CPU和GPU
# 使用ARG来选择CPU或GPU版本

ARG BASE_IMAGE=python:3.10-slim
ARG DEVICE_TYPE=cpu

# CPU版本使用Python基础镜像
FROM ${BASE_IMAGE} AS base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    libopencv-dev \
    python3-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 根据设备类型安装PyTorch
RUN if [ "${DEVICE_TYPE}" = "gpu" ]; then \
        pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# 安装其他Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 下载SAM3模型（约3.2GB）
RUN echo "下载SAM3模型..." && \
    git lfs install && \
    git clone https://www.modelscope.cn/facebook/sam3.git /tmp/sam3-model && \
    cp /tmp/sam3-model/sam3.pt ./ && \
    rm -rf /tmp/sam3-model && \
    echo "模型下载完成"

# 创建数据目录
RUN mkdir -p /app/data /app/uploads /app/exports

# 暴露端口
EXPOSE 5001

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/ || exit 1

# 启动命令
CMD ["python3", "app.py"]
