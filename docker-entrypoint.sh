#!/bin/bash
set -e

echo "=========================================="
echo "SAM3-AN Docker容器启动"
echo "=========================================="

# 检查模型文件是否存在
if [ ! -f "./models/sam3.pt" ]; then
    echo "模型文件不存在，开始下载..."

    # 初始化git-lfs
    git lfs install

    # 克隆模型仓库
    echo "从ModelScope克隆SAM3模型..."
    git clone https://www.modelscope.cn/facebook/sam3.git /tmp/sam3-model

    # 复制模型文件
    mkdir -p ../models
    if [ -f "/tmp/sam3-model/sam3.pt" ]; then
        cp /tmp/sam3-model/sam3.pt ./models/sam3.pt
        echo "模型下载完成!"
        rm -rf /tmp/sam3-model
    else
        echo "错误: 模型文件下载失败"
        exit 1
    fi
else
    echo "模型文件已存在，跳过下载"
fi

# 检查模型文件大小
MODEL_SIZE=$(stat -c%s ./models/sam3.pt 2>/dev/null || echo "0")
if [ "$MODEL_SIZE" -lt 1000000 ]; then
    echo "警告: 模型文件大小异常 (${MODEL_SIZE} bytes)"
    exit 1
fi

echo "模型文件大小: $(numfmt --to=iec $MODEL_SIZE)"

# 启动Flask服务
echo "启动SAM3-AN服务..."
echo "设备类型: ${DEVICE_TYPE:-cpu}"
echo "=========================================="

exec python3 app.py
