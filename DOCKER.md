# SAM3-AN Docker 部署指南

## 快速开始

### CPU版本（推荐用于测试）

```bash
# 使用docker-compose启动CPU版本
docker-compose up -d sam3-an-cpu

# 或者直接构建并运行
docker build -t sam3-an:cpu --build-arg DEVICE_TYPE=cpu .
docker run -d -p 5001:5001 --name sam3-an-cpu sam3-an:cpu
```

### GPU版本（需要NVIDIA GPU和nvidia-docker2）

```bash
# 使用docker-compose启动GPU版本
docker-compose --profile gpu up -d sam3-an-gpu

# 或者直接构建并运行
docker build -f Dockerfile.gpu -t sam3-an:gpu .
docker run -d --gpus all -p 5002:5001 --name sam3-an-gpu sam3-an:gpu
```

## 访问服务

- **CPU版本**: http://localhost:5001
- **GPU版本**: http://localhost:5002

## 数据持久化

默认情况下，以下目录会被挂载到宿主机：

- `./data` - 项目数据
- `./uploads` - 上传文件
- `./exports` - 导出文件

## 性能说明

### CPU版本
- **适用场景**: 测试、小批量标注
- **性能**: 较慢，单张图片推理约30-50秒
- **内存需求**: 最小4GB，推荐8GB

### GPU版本
- **适用场景**: 生产环境、大批量标注
- **性能**: 快速，单张图片推理约1-3秒
- **显存需求**: 最小6GB，推荐8GB+

## 环境变量

| 变量 | 默认值 | 说明 |
|-------|---------|------|
| DEVICE_TYPE | cpu | 设备类型（cpu/gpu） |
| FLASK_ENV | production | Flask环境 |
| PYTHONUNBUFFERED | 1 | Python输出不缓冲 |

## 故障排查

### CPU版本构建失败
```bash
# 清理缓存重新构建
docker-compose build --no-cache sam3-an-cpu
```

### GPU版本无法访问GPU
```bash
# 检查nvidia-docker2是否安装
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 检查Docker运行时
docker info | grep -i runtime
```

### 模型下载失败
```bash
# 手动下载模型并挂载
wget https://www.modelscope.cn/models/facebook/sam3/sam3.pt
docker run -v ./sam3.pt:/app/sam3.pt -d -p 5001:5001 sam3-an:cpu
```

## 开发模式

```bash
# 挂载源代码进行开发
docker run -d \
  -p 5001:5001 \
  -v $(pwd):/app \
  -v ./data:/app/data \
  sam3-an:cpu
```

## 生产部署建议

1. **使用GPU版本**: 显著提升推理速度
2. **配置Nginx反向代理**: 处理静态文件和负载均衡
3. **设置资源限制**: 防止容器占用过多资源
4. **使用数据卷**: 确保数据持久化
5. **监控日志**: 使用Docker日志驱动或ELK Stack

## License

MIT License
