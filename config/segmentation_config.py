"""
分割服务配置文件
"""
import os
from pathlib import Path


class SegmentationConfig:
    # ==================== 设备配置 ====================
    USE_GPU = True
    GPU_DEVICE_ID = 0

    # ==================== 并发配置 ====================
    NUM_WORKERS = 6              # 并发worker数（L40建议6-8）
    BATCH_SIZE = 4               # 批处理大小（根据显存调整）

    # ==================== 缓存配置 ====================
    ENABLE_CACHE = True
    CACHE_SIZE = 20              # 特征缓存大小
    CLEAR_CACHE_INTERVAL = 50    # 每N张图片清理一次缓存

    # ==================== 图像处理 ====================
    MAX_IMAGE_SIZE = 1024        # 最大图像尺寸（可降768或512提速）
    IMAGE_RESIZE_METHOD = 'LANCZOS'

    # ==================== 推理配置 ====================
    ENABLE_AMP = True            # 自动混合精度
    ENABLE_CUDNN_BENCHMARK = True # cuDNN自动优化
    DEFAULT_CONFIDENCE = 0.5     # 置信度阈值

    # ==================== NMS配置（解决重复识别问题）====================
    ENABLE_NMS = True            # 启用NMS过滤重复检测
    NMS_IOU_THRESHOLD = 0.4      # NMS IoU阈值（越小过滤越严格，建议0.3-0.5）
    NMS_MIN_AREA_RATIO = 0.1     # 最小面积比例过滤（相对于图像面积）
    NMS_OVERLAP_MODE = 'iou'     # 重叠计算模式: 'iou'(标准IoU), 'min_ratio'(相对于小框), 'both'(两者取最大)
    NMS_MASK_LEVEL = True        # 启用mask级别NMS（更精确但更慢）

    # ==================== 数据库配置 ====================
    USE_SQLITE = True            # 使用SQLite替代JSON
    DB_PATH = str(Path(__file__).parent.parent / 'data' / 'annotations.db')
    BATCH_SAVE_SIZE = 10         # 批量保存大小

    # ==================== 日志配置 ====================
    ENABLE_DEBUG_LOG = True
    ENABLE_PERFORMANCE_LOG = True


# 环境变量覆盖（支持通过环境变量动态调整）
USE_GPU = os.getenv('SAM3_USE_GPU', 'true').lower() == 'true'
if not USE_GPU:
    SegmentationConfig.USE_GPU = False
    SegmentationConfig.NUM_WORKERS = 2  # CPU模式减少并发数

NUM_WORKERS = int(os.getenv('SAM3_NUM_WORKERS', SegmentationConfig.NUM_WORKERS))
if NUM_WORKERS != SegmentationConfig.NUM_WORKERS:
    SegmentationConfig.NUM_WORKERS = NUM_WORKERS

BATCH_SIZE = int(os.getenv('SAM3_BATCH_SIZE', SegmentationConfig.BATCH_SIZE))
if BATCH_SIZE != SegmentationConfig.BATCH_SIZE:
    SegmentationConfig.BATCH_SIZE = BATCH_SIZE

CACHE_SIZE = int(os.getenv('SAM3_CACHE_SIZE', SegmentationConfig.CACHE_SIZE))
if CACHE_SIZE != SegmentationConfig.CACHE_SIZE:
    SegmentationConfig.CACHE_SIZE = CACHE_SIZE

MAX_IMAGE_SIZE = int(os.getenv('SAM3_MAX_IMAGE_SIZE', SegmentationConfig.MAX_IMAGE_SIZE))
if MAX_IMAGE_SIZE != SegmentationConfig.MAX_IMAGE_SIZE:
    SegmentationConfig.MAX_IMAGE_SIZE = MAX_IMAGE_SIZE

DB_PATH = os.getenv('SAM3_DB_PATH', SegmentationConfig.DB_PATH)
if DB_PATH != SegmentationConfig.DB_PATH:
    SegmentationConfig.DB_PATH = DB_PATH

# 支持通过环境变量明确指定使用SQLite或JSON
USE_SQLITE_ENV = os.getenv('SAM3_USE_SQLITE', '').lower()
if USE_SQLITE_ENV == 'false':
    SegmentationConfig.USE_SQLITE = False
elif USE_SQLITE_ENV == 'true':
    SegmentationConfig.USE_SQLITE = True

# NMS参数环境变量覆盖
ENABLE_NMS_ENV = os.getenv('SAM3_ENABLE_NMS', '').lower()
if ENABLE_NMS_ENV == 'false':
    SegmentationConfig.ENABLE_NMS = False
elif ENABLE_NMS_ENV == 'true':
    SegmentationConfig.ENABLE_NMS = True

NMS_IOU_THRESHOLD = os.getenv('SAM3_NMS_IOU_THRESHOLD', '')
if NMS_IOU_THRESHOLD:
    try:
        SegmentationConfig.NMS_IOU_THRESHOLD = float(NMS_IOU_THRESHOLD)
    except ValueError:
        pass

NMS_OVERLAP_MODE = os.getenv('SAM3_NMS_OVERLAP_MODE', '')
if NMS_OVERLAP_MODE in ('iou', 'min_ratio', 'both'):
    SegmentationConfig.NMS_OVERLAP_MODE = NMS_OVERLAP_MODE

NMS_MASK_LEVEL_ENV = os.getenv('SAM3_NMS_MASK_LEVEL', '').lower()
if NMS_MASK_LEVEL_ENV == 'false':
    SegmentationConfig.NMS_MASK_LEVEL = False
elif NMS_MASK_LEVEL_ENV == 'true':
    SegmentationConfig.NMS_MASK_LEVEL = True

# 打印配置
if SegmentationConfig.ENABLE_DEBUG_LOG:
    print("=" * 60)
    print("SAM3分割服务配置")
    print("=" * 60)
    print(f"使用GPU: {SegmentationConfig.USE_GPU}")
    print(f"并发数: {SegmentationConfig.NUM_WORKERS}")
    print(f"批处理大小: {SegmentationConfig.BATCH_SIZE}")
    print(f"缓存大小: {SegmentationConfig.CACHE_SIZE}")
    print(f"最大图像尺寸: {SegmentationConfig.MAX_IMAGE_SIZE}")
    print(f"使用SQLite: {SegmentationConfig.USE_SQLITE}")
    print(f"数据库路径: {SegmentationConfig.DB_PATH}")
    print(f"启用NMS: {SegmentationConfig.ENABLE_NMS}")
    print(f"NMS IoU阈值: {SegmentationConfig.NMS_IOU_THRESHOLD}")
    print(f"NMS重叠模式: {SegmentationConfig.NMS_OVERLAP_MODE}")
    print(f"NMS Mask级别: {SegmentationConfig.NMS_MASK_LEVEL}")
    print("=" * 60)
