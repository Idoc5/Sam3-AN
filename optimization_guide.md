# SAM3-AN 性能优化指南

基于性能分析，以下是提升分割性能的多种方案：

## 🔍 当前性能瓶颈分析

1. **主要瓶颈**: 强制使用CPU模式（即使CUDA可用）
2. **模型大小**: 3.2GB 大模型，在CPU上推理非常慢
3. **图像处理**: 缺少图像预处理优化
4. **缓存机制**: 缺少图像特征缓存

## 🚀 优化方案

### 方案1: 启用GPU加速（最高优先级）

**问题**: 当前代码强制使用CPU模式
**解决方案**: 修改 `sam3_service.py`，启用GPU支持

```python
# 修改 _init_image_model 方法
def _init_image_model(self):
    """初始化图像分割模型"""
    if self.image_model is not None:
        return

    print("正在加载SAM3图像模型...")

    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 优化CUDA设置
    if device == "cuda":
        if hasattr(torch, "backends"):
            if hasattr(torch.backends, "cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        print("已启用CUDA加速")
    else:
        print("⚠️ CUDA不可用，使用CPU模式（性能较差）")

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe_path = sam3_src / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    self.image_model = build_sam3_image_model(bpe_path=str(bpe_path), device=device, eval_mode=True)
    self.image_processor = Sam3Processor(self.image_model, device=device)
    
    # 移动模型到正确设备
    self.image_model = self.image_model.to(device)

    print("SAM3图像模型加载完成")
```

**预期效果**: GPU模式下性能提升 10-50倍

### 方案2: 图像预处理优化

**问题**: 大图像处理慢，缺少尺寸优化
**解决方案**: 添加图像尺寸优化逻辑

```python
def _load_image(self, image_path: str, max_size: int = 1024):
    """加载并优化图像尺寸"""
    if self.current_image_path != image_path or self.inference_state is None:
        print(f"[DEBUG] 加载图像: {image_path}")
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        
        # 优化图像尺寸
        width, height = image.size
        if max(width, height) > max_size:
            # 保持宽高比缩放
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            print(f"[OPTIMIZE] 缩放图像: {width}x{height} -> {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self._image_size = image.size
        self.inference_state = self.image_processor.set_image(image)
        self.current_image_path = image_path
        print(f"[DEBUG] 图像尺寸: {self._image_size}")
```

**预期效果**: 减少30-70%推理时间

### 方案3: 特征缓存优化

**问题**: 重复加载相同图像时重复计算特征
**解决方案**: 添加特征缓存机制

```python
class SAM3Service:
    def __init__(self):
        self.image_model = None
        self.image_processor = None
        self.video_predictor = None
        self.current_image_path = None
        self.inference_state = None
        self.video_sessions = {}
        self._image_size = None
        self._feature_cache = {}  # 新增：特征缓存
        
    def _load_image(self, image_path: str, max_size: int = 1024):
        """加载图像，使用特征缓存"""
        if image_path in self._feature_cache:
            print(f"[CACHE] 使用缓存特征: {image_path}")
            self.inference_state = self._feature_cache[image_path]['state']
            self._image_size = self._feature_cache[image_path]['size']
            self.current_image_path = image_path
            return
        
        # ... 原有的图像加载逻辑 ...
        
        # 缓存特征
        self._feature_cache[image_path] = {
            'state': self.inference_state,
            'size': self._image_size
        }
        # 限制缓存大小
        if len(self._feature_cache) > 10:
            # 移除最旧的缓存
            oldest_key = next(iter(self._feature_cache))
            del self._feature_cache[oldest_key]
```

**预期效果**: 相同图像二次处理速度提升 90%+

### 方案4: 批量处理优化

**问题**: 批量分割时串行处理
**解决方案**: 添加批量处理接口和异步支持

```python
def batch_segment_by_text(self, image_paths: list, prompt: str, confidence: float = 0.5):
    """批量文本分割"""
    results = {}
    
    for image_path in image_paths:
        try:
            result = self.segment_by_text(image_path, prompt, confidence)
            results[image_path] = result
        except Exception as e:
            print(f"[ERROR] 批量分割失败 {image_path}: {e}")
            results[image_path] = []
    
    return results

# 可选：添加异步版本
import asyncio
async def async_batch_segment(self, image_paths: list, prompt: str):
    """异步批量分割"""
    loop = asyncio.get_event_loop()
    tasks = []
    
    for image_path in image_paths:
        task = loop.run_in_executor(
            None, self.segment_by_text, image_path, prompt, 0.5
        )
        tasks.append((image_path, task))
    
    results = {}
    for image_path, task in tasks:
        try:
            result = await task
            results[image_path] = result
        except Exception as e:
            print(f"[ERROR] 异步分割失败 {image_path}: {e}")
            results[image_path] = []
    
    return results
```

### 方案5: 模型量化（CPU模式专用）

**问题**: CPU模式下模型推理慢
**解决方案**: 使用量化模型减少计算量

```python
def _init_quantized_model(self):
    """初始化量化模型（CPU专用）"""
    if self.image_model is not None:
        return
    
    print("正在加载量化SAM3模型...")
    
    device = "cpu"
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    bpe_path = sam3_src / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    self.image_model = build_sam3_image_model(bpe_path=str(bpe_path), device=device, eval_mode=True)
    
    # 应用动态量化
    import torch.quantization
    self.image_model = torch.quantization.quantize_dynamic(
        self.image_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    self.image_processor = Sam3Processor(self.image_model, device=device)
    print("量化SAM3模型加载完成")
```

**预期效果**: CPU模式下性能提升 2-4倍，精度略有下降

### 方案6: 配置系统优化

创建配置文件，让用户选择优化选项：

```python
# config/performance_config.py
PERFORMANCE_CONFIG = {
    'use_gpu': True,  # 是否使用GPU
    'max_image_size': 1024,  # 最大图像尺寸
    'enable_cache': True,  # 是否启用缓存
    'cache_size': 10,  # 缓存大小
    'quantize_model': False,  # 是否量化模型（CPU模式）
    'batch_size': 4,  # 批量处理大小
    'async_processing': True,  # 是否异步处理
}
```

## 📊 性能提升预期

| 优化方案 | 预期性能提升 | 实现难度 | 适用场景 |
|---------|-------------|---------|---------|
| GPU加速 | 10-50倍 | 低 | 有NVIDIA GPU |
| 图像尺寸优化 | 30-70% | 低 | 大图像处理 |
| 特征缓存 | 90%+（缓存命中） | 中 | 重复处理相同图像 |
| 批量处理 | 2-5倍（批量） | 中 | 批量标注任务 |
| 模型量化 | 2-4倍（CPU） | 高 | 无GPU环境 |
| 异步处理 | 2-3倍（IO密集） | 中 | 网络或磁盘IO瓶颈 |

## 🛠️ 实施步骤

### 第一阶段：快速优化（1-2小时）
1. 启用GPU支持（修改 `_init_image_model`）
2. 添加图像尺寸优化（修改 `_load_image`）
3. 测试GPU环境

### 第二阶段：中级优化（2-4小时）
1. 实现特征缓存
2. 添加批量处理接口
3. 创建性能配置文件

### 第三阶段：高级优化（4-8小时）
1. 实现模型量化（CPU专用）
2. 添加异步处理支持
3. 性能监控和调优

## 🔧 环境检查脚本

创建 `check_environment.py` 检查优化环境：

```bash
python check_environment.py
```

检查项目：
- CUDA可用性和版本
- GPU显存大小
- PyTorch配置
- OpenCV安装
- 内存和磁盘空间

## 📈 性能监控

添加性能监控功能：

```python
class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        
    def start_timing(self, operation):
        self.timings[operation] = time.time()
    
    def end_timing(self, operation):
        if operation in self.timings:
            duration = time.time() - self.timings[operation]
            print(f"[PERF] {operation}: {duration:.3f}s")
            # 可记录到日志文件
            
    def get_report(self):
        return self.timings
```

## 🎯 推荐优化路径

根据你的硬件环境选择：

### 有NVIDIA GPU环境：
1. **首要**: 启用GPU加速
2. **次要**: 图像尺寸优化 + 特征缓存
3. **进阶**: 批量处理优化

### 无GPU（CPU only）环境：
1. **首要**: 图像尺寸优化（缩小到512px）
2. **次要**: 特征缓存 + 模型量化
3. **进阶**: 使用更小的SAM3变体

### 批量处理场景：
1. **首要**: 批量处理接口
2. **次要**: 异步处理 + 特征缓存
3. **进阶**: 分布式处理

## 📝 注意事项

1. **GPU内存**: SAM3需要6-8GB显存，确保足够
2. **精度权衡**: 量化会略微降低精度
3. **缓存有效性**: 特征缓存对重复图像有效
4. **兼容性**: 确保优化不影响现有功能

## 🚨 故障排除

### GPU启用失败：
- 检查CUDA驱动版本
- 检查PyTorch CUDA版本匹配
- 检查显存是否足够

### 性能未提升：
- 检查是否真正使用GPU
- 检查图像尺寸是否优化
- 检查缓存是否生效

### 内存泄漏：
- 定期清理缓存
- 使用 `torch.cuda.empty_cache()`
- 监控内存使用情况

## 📚 参考资料

1. [PyTorch GPU加速指南](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
2. [SAM3官方性能优化](https://github.com/facebookresearch/sam3)
3. [模型量化教程](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
4. [OpenCV性能优化](https://docs.opencv.org/master/dc/d71/tutorial_py_optimization.html)

---

**开始优化前，请备份原始代码！**

建议从方案1（GPU加速）开始，这是性价比最高的优化方案。