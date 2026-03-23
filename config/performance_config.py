"""
SAM3-AN 性能配置
可以通过修改此文件来调整性能参数
"""

class PerformanceConfig:
    # ==================== 设备配置 ====================
    # 是否使用GPU（如果有GPU可用）
    USE_GPU = True

    # GPU设备ID（多GPU时使用）
    GPU_DEVICE_ID = 0

    # ==================== 混合精度配置 ====================
    # 是否启用自动混合精度（仅GPU，可提升2-3倍速度）
    ENABLE_AMP = True

    # ==================== 图像处理配置 ====================
    # 最大图像尺寸（越大越精确，但越慢）
    # 推荐：GPU: 1024-1536, CPU: 512-768
    MAX_IMAGE_SIZE = 1024

    # 图像缩放时的插值方法
    # 'NEAREST', 'LANCZOS', 'BILINEAR', 'BICUBIC'
    IMAGE_RESIZE_METHOD = 'LANCZOS'

    # ==================== 缓存配置 ====================
    # 是否启用特征缓存
    ENABLE_CACHE = True

    # 最大缓存数量（每个缓存占用约100-300MB显存）
    CACHE_MAX_SIZE = 10

    # ==================== 批量处理配置 ====================
    # 批量处理时的并发数（仅限CPU模式）
    BATCH_CONCURRENT_WORKERS = 2

    # 是否启用批量图像预处理
    ENABLE_BATCH_PREPROCESSING = True

    # ==================== 推理优化配置 ====================
    # 是否启用 cudnn benchmark（可能加速首次推理）
    ENABLE_CUDNN_BENCHMARK = True

    # 文本分割时的置信度阈值（默认0.5）
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    # ==================== 模型优化（CPU模式）====================
    # 是否对模型进行量化（仅CPU，可减少2-4倍推理时间，精度略降）
    ENABLE_MODEL_QUANTIZATION = False

    # ==================== 内存管理 ====================
    # 每次推理后是否清理GPU缓存
    CLEAR_CACHE_AFTER_INFERENCE = False

    # 缓存清理间隔（推理次数）
    CACHE_CLEAR_INTERVAL = 10

    # ==================== 日志配置 ====================
    # 是否显示性能日志
    SHOW_PERFORMANCE_LOGS = True

    # 是否显示详细的调试信息
    VERBOSE_DEBUG = False
