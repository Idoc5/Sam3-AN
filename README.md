# SAM3 AN - 智能数据标注工具

基于 **SAM3 (Segment Anything Model 3)** 的智能数据标注工具，在 https://github.com/fivif/Sam3-AN 的基础上进行魔改。

支持图像分割标注。通过文本提示、点击、框选等多种方式快速生成高质量标注数据。

## 目录

- [功能特性](#-功能特性)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [项目结构](#-项目结构)
- [配置说明](#-配置说明)
- [API 文档](#-api-文档)
- [常见问题](#-常见问题)
- [技术架构](#-技术架构)
- [性能优化](#-性能优化)

## ✨ 功能特性

### 🖼️ 图像标注

| 功能 | 描述 |
|------|------|
| **文本提示分割** | 输入中/英文描述，AI 自动识别并分割目标对象 |
| **点击分割** | 通过点击添加正/负样本点进行精确分割 |
| **框选分割** | 绘制边界框指定分割区域，支持正/负样本框 |
| **手动绘制** | 多边形工具手动绘制标注区域 |
| **批量分割** | 对多张图片进行批量自动分割，支持并发处理 |

### 🎯 正负样本系统

- **正样本 (绿色)**: 指示要分割的目标区域
- **负样本 (红色)**: 指示要排除的区域，用于精细化分割结果
- **智能过滤**: 使用 Mask 级别的重叠检测（NMS），精确排除不需要的分割结果

### 🤖 AI 翻译功能

- 支持中文输入，自动翻译为英文提示词
- 兼容 OpenAI API 格式（DeepSeek、通义千问、Moonshot 等）
- 可配置 API 地址、密钥和模型
- 内置连接测试功能

### 📦 数据导出

| 格式 | 说明 |
|------|------|
| **YOLO** | 支持 YOLOv5/v8/v11/v26 检测和分割格式 |
| **COCO** | 标准 COCO 实例分割格式 |

- 自动按 8:1:1 比例分割 train/val/test 数据集
- 支持多边形平滑处理（none/low/medium/high/ultra）
- 提供导出预览功能

### ⚡ 性能优化

- **并发处理**: 支持多线程批量分割，大幅提升处理速度
- **缓存机制**: 智能缓存已处理图片，避免重复计算
- **实时保存**: 批量处理时实时保存结果，防止数据丢失
- **进度追踪**: 实时显示处理进度和统计信息

### 🗄️ 数据管理

- **SQLite 数据库**: 高效存储标注数据，支持大规模项目
- **标注管理**: 支持标注的增删改查、批量操作
- **类别管理**: 灵活管理标注类别
- **数据清理**: 一键清理低置信度标注、非手动标注

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.6 (推荐，CPU 也可运行但较慢)
- 8GB+ GPU 显存 (推荐)

### 安装步骤

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 2. 进入项目目录
cd sam3-an

# 3. 安装 PyTorch (根据你的 CUDA 版本选择)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 4. 安装所有依赖
pip install -r requirements.txt

# 5. 下载 SAM3 模型权重
# 将 sam3.pt 放置在项目根目录下
# 模型下载: https://www.modelscope.cn/models/facebook/sam3
```

> **注意**: SAM3 核心代码已包含在 `SAM_src/` 目录中，无需额外安装。

### Docker 部署

```bash
# 使用 GPU
docker build -f Dockerfile.gpu -t sam3-an .
docker run -it --gpus all -p 5001:5001 -v $(pwd)/data:/workspace/data sam3-an

# 使用 CPU
docker build -f Dockerfile.cpu -t sam3-an .
docker run -it -p 5001:5001 -v $(pwd)/data:/workspace/data sam3-an
```

### 启动服务

```bash
python app.py
```

启动后会自动打开浏览器访问 http://localhost:5001

## 📖 使用指南

### 基本工作流程

```
创建项目 → 加载图片 → 添加类别 → 标注 → 保存 → 导出
```

1. **创建项目**: 点击左上角项目名称，选择"项目管理"
2. **设置目录**: 配置图片目录和输出目录
3. **添加类别**: 在右侧面板添加标注类别
4. **开始标注**: 选择工具进行标注
5. **保存导出**: 保存标注并导出数据集

### 标注工具

#### 文本提示分割 (推荐)

1. 在工具栏输入框中输入目标描述（如 "apple" 或 "苹果"）
2. 调整置信度阈值（默认 0.5）
3. 点击"分割"按钮或按 Enter

#### 点击分割

1. 选择"点击"工具
2. 选择正样本(+)或负样本(-)模式
3. 在图像上点击目标位置
4. 点击"分割"按钮执行分割

#### 框选分割

1. 选择"框选"工具
2. 选择正样本(+)或负样本(-)模式
3. 在图像上绘制边界框
4. 可添加多个正/负样本框
5. 点击"分割"按钮执行分割

#### 手动绘制

1. 选择"多边形"工具
2. 点击添加多边形顶点
3. 双击或点击起点闭合多边形

### 正负样本使用技巧

```
场景：图片中有多个苹果，只想标注其中一个

方法1：框选
1. 用正样本框(+)框选目标苹果
2. 用负样本框(-)框选不想要的苹果
3. 点击分割

方法2：点击
1. 用正样本点(+)点击目标苹果
2. 用负样本点(-)点击不想要的苹果
3. 点击分割
```

### 批量分割

1. 展开右侧"批量分割"面板
2. 输入提示词和目标类别
3. 设置图片范围（起始/结束索引）
4. 勾选"跳过已标注"保护已有标注
5. 点击"开始批量分割"（支持并发模式）
6. 查看实时进度和统计信息

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `←` / `→` | 上一张/下一张图片 |
| `Ctrl+S` | 保存当前标注 |
| `Delete` | 删除选中的标注 |
| `Escape` | 取消当前操作 |
| `+` / `-` | 放大/缩小 |
| `0` | 重置缩放 |
| `F` | 适应窗口 |

## 🏗️ 项目结构

```
sam3-an/
├── app.py                      # Flask 主应用
├── requirements.txt            # 依赖列表
├── sam3.pt                     # SAM3 模型权重（需单独下载）
├── README.md                   # 项目文档
│
├── services/
│   ├── sam3_service.py         # SAM3 模型服务封装
│   ├── sam3_concurrent_service.py  # 并发处理服务
│   ├── annotation_manager.py   # JSON 标注管理器
│   └── db_annotation_manager.py    # SQLite 标注管理器（默认）
│
├── exports/
│   ├── yolo_exporter.py        # YOLO 格式导出
│   └── coco_exporter.py        # COCO 格式导出
│
├── config/
│   ├── segmentation_config.py  # 分割配置
│   └── performance_config.py   # 性能配置
│
├── utils/
│   ├── performance_monitor.py  # 性能监控工具
│   └── migrate_to_sqlite.py    # 数据迁移工具
│
├── templates/
│   ├── index.html              # 图像标注页面
│   └── video.html              # 视频标注页面
│
├── data/
│   ├── annotations.db          # SQLite 数据库
│   └── projects.json           # 项目数据备份
│
├── uploads/                    # 上传文件临时目录
│
└── SAM_src/                    # SAM3 源码（本地副本）
```

## ⚙️ 配置说明

### AI 翻译配置

点击工具栏的 AI 翻译配置按钮：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| API 地址 | OpenAI 格式 API 地址 | `https://api.deepseek.com` |
| API 密钥 | 你的 API Key | `sk-xxx...` |
| 模型名称 | 使用的模型 | `deepseek-chat` |

支持的 API 服务：
- DeepSeek: `https://api.deepseek.com`
- 通义千问: `https://dashscope.aliyuncs.com/compatible-mode`
- Moonshot: `https://api.moonshot.cn`
- OpenAI: `https://api.openai.com`

### NMS 配置

通过配置文件或 API 调整 NMS 参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| enabled | 是否启用 NMS | true |
| iou_threshold | IOU 阈值 | 0.4 |
| overlap_mode | 重叠模式 (iou/area) | iou |
| min_area_ratio | 最小面积比例 | 0.1 |
| mask_level | 是否使用 Mask 级别 NMS | true |

### 置信度阈值

- 范围: 0.01 - 1.0
- 默认: 0.5
- 较高值: 更精确但可能漏检
- 较低值: 更全面但可能误检

### 性能配置

编辑 `config/performance_config.py`：

```python
# 并发工作线程数
NUM_WORKERS = 2

# 批处理大小
BATCH_SIZE = 4

# 缓存大小
CACHE_SIZE = 100

# 最大图片尺寸
MAX_IMAGE_SIZE = 2048
```

## 📡 API 文档

### 项目管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/project/create` | POST | 创建新项目 |
| `/api/project/<id>` | GET | 获取项目信息 |
| `/api/project/list` | GET | 列出所有项目 |
| `/api/project/<id>/update` | POST | 更新项目 |
| `/api/project/<id>/delete` | POST | 删除项目 |
| `/api/project/<id>/load_images` | POST | 加载图片 |

### 分割接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/segment/text` | POST | 文本提示分割 |
| `/api/segment/point` | POST | 点击分割 |
| `/api/segment/box` | POST | 框选分割 |
| `/api/segment/batch` | POST | 批量分割（串行） |
| `/api/segment/batch/concurrent` | POST | 批量分割（并发） |
| `/api/segment/batch/progress` | GET | 查询批量处理进度 |

### 标注管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/annotation/save` | POST | 保存标注 |
| `/api/annotation/add` | POST | 添加标注 |
| `/api/annotation/get` | GET | 获取标注 |
| `/api/annotation/update` | POST | 更新标注 |
| `/api/annotation/delete` | POST | 删除标注 |
| `/api/annotation/clear_low_confidence` | POST | 清理低置信度标注 |

### 导出接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/export/yolo` | POST | 导出 YOLO 格式 |
| `/api/export/coco` | POST | 导出 COCO 格式 |
| `/api/export/preview` | POST | 生成导出预览 |

### 配置接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/config/nms` | GET | 获取 NMS 配置 |
| `/api/config/nms` | POST | 更新 NMS 配置 |

## ❓ 常见问题

### Q: 首次启动很慢？

A: 首次启动需要加载 SAM3 模型（约 3.2GB），请耐心等待。后续启动会更快。

### Q: 显存不足？

A: SAM3 需要约 6-8GB 显存。可尝试：
- 关闭其他 GPU 程序
- 使用较小的图片
- 调整 `MAX_IMAGE_SIZE` 配置
- 使用 CPU 模式（较慢）

### Q: 分割结果不准确？

A: 尝试以下方法：
- 调整置信度阈值
- 使用更精确的提示词
- 使用正负样本框/点进行精细化
- 使用英文提示词（更准确）
- 调整 NMS 参数

### Q: 中文提示词不工作？

A: 配置 AI 翻译功能，自动将中文翻译为英文。

吊机 crane
挖掘机   digger or excavator
工程机械 Construction machine or crane or digger or excavator
工人 worker

### Q: 如何批量处理大量图片？

A: 使用并发批量分割功能：
1. 使用 `/api/segment/batch/concurrent` 接口
2. 调整 `NUM_WORKERS` 和 `BATCH_SIZE` 参数
3. 监控处理进度，结果会实时保存

### Q: 如何导出数据？

A: 在项目管理中选择导出格式（YOLO/COCO），选择平滑级别，点击导出。数据会自动按 8:1:1 分割。

### Q: 如何清理低质量标注？

A: 使用标注清理功能：
- 清理低置信度标注：设置阈值批量删除
- 清理非手动标注：保留手动标注，删除 AI 生成的
- 按类别清理：删除特定类别的所有标注

### Q: 数据库文件过大？

A: 可以定期备份数据到 JSON，然后清理数据库：
```bash
python utils/migrate_to_sqlite.py --backup-only
```

## 🛠️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (Browser)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Canvas 渲染 │  │  工具栏交互  │  │  标注管理   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                           │ HTTP/REST API
┌─────────────────────────────────────────────────────────┐
│                   后端 (Flask)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  路由处理    │  │  SAM3 服务   │  │  数据管理   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                       │                   │             │
│            ┌──────────┴─────┐   ┌─────────┴─────┐      │
│            │  并发服务      │   │  SQLite 数据库 │      │
│            └────────────────┘   └───────────────┘      │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   SAM3 模型层                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  图像编码器  │  │  文本编码器  │  │  Mask 解码器 │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## ⚡ 性能优化

### 并发处理

工具内置并发处理支持，可显著提升批量处理速度：

```python
# 自动使用并发接口
POST /api/segment/batch/concurrent
{
  "project_id": "xxx",
  "prompt": "car",
  "max_workers": 2,
  "use_cache": true
}
```

### 缓存机制

- 自动缓存已处理图片的特征
- 避免重复计算，提升处理速度
- 可通过配置调整缓存大小

### 性能监控

使用内置性能监控工具分析瓶颈：

```bash
python utils/performance_monitor.py
```

### 优化建议

1. **批量处理**: 优先使用并发批量接口
2. **图片尺寸**: 适当缩小图片可提升速度
3. **缓存启用**: 确保缓存机制开启
4. **GPU 利用**: 使用 GPU 模式，关闭其他 GPU 程序

## 📄 许可证

MIT License

## 🙏 致谢

- [SAM3 - Segment Anything Model 3](https://github.com/facebookresearch/sam3)
- [Linux.do](https://linux.do/)
- [Flask](https://flask.palletsprojects.com/)
- [PyTorch](https://pytorch.org/)

---

<p align="center">
  ⭐ Star this repo if it helps you!
</p>
