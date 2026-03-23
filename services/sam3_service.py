"""
SAM3模型服务封装 - 修正版
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
import uuid
import traceback

# 使用本地 SAM_src 目录
sam3_src = Path(__file__).parent.parent / "SAM_src"
sys.path.insert(0, str(sam3_src))


class SAM3Service:
    """SAM3模型服务"""

    def __init__(self):
        self.image_model = None
        self.image_processor = None
        self.video_predictor = None
        self.current_image_path = None
        self.inference_state = None
        self.video_sessions = {}
        self._image_size = None
        self._feature_cache = {}  # 特征缓存：{image_path: {'state': inference_state, 'size': image_size}}
        self._cache_max_size = 10  # 最大缓存数量
        
        # 显存管理相关
        self._current_device = None  # 当前使用的设备 (cuda/cpu)
        self._gpu_failed_count = 0   # GPU失败计数
        self._use_cpu_fallback = False  # 是否临时使用CPU
        self._max_gpu_retries = 3    # GPU最大重试次数

    def _check_cuda_available(self) -> bool:
        """检查CUDA是否可用且有足够显存"""
        if not torch.cuda.is_available():
            return False
        
        try:
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            free_memory = total_memory - reserved
            
            # 需要至少2GB可用显存
            if free_memory >= 2.0:
                return True
            else:
                print(f"[GPU] 可用显存不足: {free_memory:.2f}GB (需要≥2GB)")
                return False
        except Exception as e:
            print(f"[GPU] 显存检测失败: {e}")
            return False

    def _clear_gpu_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[GPU] 已清理GPU缓存")

    def _release_gpu_resources(self):
        """释放GPU资源"""
        if self.image_model is not None:
            try:
                # 将模型移到CPU
                self.image_model = self.image_model.to('cpu')
            except:
                pass
        self._clear_gpu_cache()

    def _init_image_model(self, force_device: str = None):
        """初始化图像分割模型
        
        Args:
            force_device: 强制使用的设备 ('cuda' 或 'cpu')，None则自动选择
        """
        if self.image_model is not None and force_device is None:
            return

        print("正在加载SAM3图像模型...")

        # 确定使用的设备
        if force_device:
            device = force_device
        elif self._use_cpu_fallback:
            # 之前GPU失败过，临时使用CPU
            device = "cpu"
            print("[DEVICE] 临时使用CPU模式（上次GPU失败）")
        elif not self._check_cuda_available():
            device = "cpu"
            print("[DEVICE] CUDA不可用或显存不足，使用CPU模式")
        else:
            device = "cuda"
        
        self._current_device = device
        print(f"[DEVICE] 使用设备: {device}")

        # 优化CUDA设置
        if device == "cuda":
            if hasattr(torch, "backends"):
                if hasattr(torch.backends, "cuda"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = True
            # 启用自动混合精度（AMP）
            self.use_amp = True
            self.autocast = torch.cuda.amp.autocast(enabled=True)
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            print("[GPU] 已启用CUDA加速 + 混合精度")
        else:
            self.use_amp = False
            print("[CPU] 使用CPU模式（性能较差）")

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from config.segmentation_config import SegmentationConfig

        bpe_path = sam3_src / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        self.image_model = build_sam3_image_model(bpe_path=str(bpe_path), device=device, eval_mode=True)
        self.image_processor = Sam3Processor(
            self.image_model,
            device=device,
            confidence_threshold=SegmentationConfig.DEFAULT_CONFIDENCE,
            enable_nms=SegmentationConfig.ENABLE_NMS,
            nms_iou_threshold=SegmentationConfig.NMS_IOU_THRESHOLD,
        )
        
        # 移动模型到正确设备
        self.image_model = self.image_model.to(device)

        print(f"SAM3图像模型加载完成 (设备: {device})")

    def _handle_oom_error(self, operation_name: str) -> bool:
        """处理显存不足错误
        
        Returns:
            True 如果可以重试，False 如果应该放弃
        """
        self._gpu_failed_count += 1
        print(f"[OOM] 显存不足 ({operation_name}), 失败次数: {self._gpu_failed_count}/{self._max_gpu_retries}")
        
        # 清理GPU缓存
        self._clear_gpu_cache()
        
        if self._gpu_failed_count >= self._max_gpu_retries:
            # 达到最大重试次数，切换到CPU
            print(f"[OOM] 重试{self._max_gpu_retries}次后仍失败，切换到CPU模式")
            self._use_cpu_fallback = True
            self._gpu_failed_count = 0
            return False
        
        return True  # 可以重试

    def _reset_gpu_state(self):
        """重置GPU状态（下一个任务重新尝试GPU）"""
        if self._use_cpu_fallback:
            print("[GPU] 尝试重新启用GPU模式")
            self._use_cpu_fallback = False
            self._gpu_failed_count = 0
            self._current_device = None
            # 释放当前模型，下次使用时重新加载到GPU
            self.image_model = None
            self.image_processor = None

    def _load_image(self, image_path: str, max_size: int = 1024, use_cache: bool = True):
        """加载并优化图像尺寸，支持特征缓存"""
        # 检查缓存
        if use_cache and image_path in self._feature_cache:
            print(f"[CACHE] 使用缓存特征: {image_path}")
            cached = self._feature_cache[image_path]
            self.inference_state = cached['state']
            self._image_size = cached['size']
            self._original_size = cached['original_size']
            self._scale_factor = cached['scale_factor']
            self.current_image_path = image_path
            return

        print(f"[DEBUG] 加载图像: {image_path}")
        image = Image.open(image_path)
        # 处理 EXIF 旋转信息，修复竖屏图像分割偏移问题
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')

        # 优化图像尺寸（如果过大）
        width, height = image.size
        original_size = (width, height)
        scale_factor = 1.0

        if max(width, height) > max_size:
            # 保持宽高比缩放
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            scale_factor = width / new_width if width > height else height / new_height
            print(f"[OPTIMIZE] 缩放图像: {width}x{height} -> {new_width}x{new_height}, 缩放因子: {scale_factor:.4f}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self._image_size = image.size
        self._original_size = original_size
        self._scale_factor = scale_factor
        self.inference_state = self.image_processor.set_image(image)
        self.current_image_path = image_path

        # 缓存特征
        if use_cache:
            self._feature_cache[image_path] = {
                'state': self.inference_state,
                'size': self._image_size,
                'original_size': original_size,
                'scale_factor': scale_factor
            }
            # 限制缓存大小
            if len(self._feature_cache) > self._cache_max_size:
                # 移除最旧的缓存（FIFO）
                oldest_key = next(iter(self._feature_cache))
                del self._feature_cache[oldest_key]
                print(f"[CACHE] 移除旧缓存: {oldest_key}")

        print(f"[DEBUG] 图像尺寸: {self._image_size} (原始: {original_size}), 缩放因子: {scale_factor:.4f}")
    
    def clear_cache(self):
        """清空特征缓存"""
        cache_size = len(self._feature_cache)
        self._feature_cache.clear()
        print(f"[CACHE] 已清空缓存，释放了 {cache_size} 个特征")

    def _smooth_mask(self, mask: np.ndarray, smooth_level: str) -> np.ndarray:
        """在 mask 级别进行形态学平滑，从根本上消除锯齿

        Args:
            mask: 二值mask (0/255)
            smooth_level: 平滑级别

        Returns:
            平滑后的mask
        """
        import cv2

        # 平滑参数：kernel_size 越大，平滑效果越强
        smooth_params = {
            'none': {'kernel_size': 0},
            'low': {'kernel_size': 3},
            'medium': {'kernel_size': 5},
            'high': {'kernel_size': 7},
            'ultra': {'kernel_size': 11},
        }

        params = smooth_params.get(smooth_level, smooth_params['medium'])
        kernel_size = params['kernel_size']

        if kernel_size == 0:
            return mask

        # 使用形态学操作平滑边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 先闭运算（填充小孔），再开运算（去除毛刺）
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        # 使用高斯模糊 + 阈值进一步平滑边缘
        if kernel_size >= 5:
            blur_size = kernel_size | 1  # 确保是奇数
            smoothed = cv2.GaussianBlur(smoothed, (blur_size, blur_size), 0)
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        return smoothed

    def _adaptive_simplify(self, points: np.ndarray, epsilon_factor: float) -> np.ndarray:
        """自适应简化多边形"""
        import cv2
        if len(points) < 3:
            return points

        contour = points.reshape(-1, 1, 2).astype(np.float32)
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        return approx.reshape(-1, 2)

    def _mask_to_polygon(self, mask: np.ndarray, smooth_level: str = 'medium') -> list:
        """将mask转换为多边形，在mask级别进行平滑处理

        Args:
            mask: 二值mask
            smooth_level: 平滑级别 'none', 'low', 'medium', 'high', 'ultra'

        Returns:
            多边形点列表 [[x, y], ...]
        """
        import cv2
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255

        # 在 mask 级别进行形态学平滑（关键步骤！）
        smoothed_mask = self._smooth_mask(mask, smooth_level)

        # 使用 CHAIN_APPROX_TC89_KCOS 算法提取更平滑的轮廓
        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        points = largest.reshape(-1, 2).astype(np.float64)

        # 简化参数
        simplify_params = {
            'none': 0.002,
            'low': 0.0015,
            'medium': 0.001,
            'high': 0.0008,
            'ultra': 0.0005,
        }

        epsilon_factor = simplify_params.get(smooth_level, 0.001)
        result = self._adaptive_simplify(points, epsilon_factor)

        return result.tolist()

    def _chaikin_smooth(self, points: np.ndarray, iterations: int) -> np.ndarray:
        """使用 Chaikin 算法平滑多边形
        
        Args:
            points: 多边形点数组 (N, 2)
            iterations: 迭代次数
            
        Returns:
            平滑后的点数组
        """
        if len(points) < 3:
            return points
            
        result = points.copy()
        for _ in range(iterations):
            new_points = []
            n = len(result)
            
            for i in range(n):
                p0 = result[i]
                p1 = result[(i + 1) % n]
                
                # Chaikin 算法: 生成两个新点
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                
                new_points.append(q)
                new_points.append(r)
            
            result = np.array(new_points, dtype=np.float64)
        
        return result

    def smooth_polygon(self, polygon: list, smooth_level: str = 'medium') -> list:
        """对已有多边形进行平滑处理（使用 Chaikin 算法，不会收缩）

        Args:
            polygon: 多边形点列表 [[x, y], ...]
            smooth_level: 平滑级别 'none', 'low', 'medium', 'high', 'ultra'

        Returns:
            平滑后的多边形点列表
        """
        if not polygon or len(polygon) < 3:
            return polygon

        smooth_params = {
            'none': {'chaikin_iterations': 0, 'simplify_epsilon': 0.003},
            'low': {'chaikin_iterations': 1, 'simplify_epsilon': 0.002},
            'medium': {'chaikin_iterations': 2, 'simplify_epsilon': 0.0015},
            'high': {'chaikin_iterations': 3, 'simplify_epsilon': 0.001},
            'ultra': {'chaikin_iterations': 4, 'simplify_epsilon': 0.0008},
        }

        params = smooth_params.get(smooth_level, smooth_params['medium'])
        points = np.array(polygon, dtype=np.float64)

        if params['chaikin_iterations'] == 0:
            result = self._adaptive_simplify(points, params['simplify_epsilon'])
            return result.tolist()

        # Chaikin 平滑 + 简化
        smoothed = self._chaikin_smooth(points, params['chaikin_iterations'])
        result = self._adaptive_simplify(smoothed, params['simplify_epsilon'])

        return result.tolist()

    def _execute_with_oom_retry(self, operation_func, operation_name: str, *args, **kwargs):
        """执行操作并在显存不足时自动重试
        
        Args:
            operation_func: 要执行的操作函数
            operation_name: 操作名称（用于日志）
            *args, **kwargs: 传递给操作函数的参数
            
        Returns:
            操作结果
        """
        max_attempts = self._max_gpu_retries + 1  # GPU重试次数 + 1次CPU
        
        for attempt in range(max_attempts):
            try:
                # 尝试重置GPU状态（下一个任务重新尝试GPU）
                if attempt > 0 and self._use_cpu_fallback:
                    self._init_image_model(force_device='cpu')
                else:
                    self._init_image_model()
                
                result = operation_func(*args, **kwargs)
                
                # 成功后重置GPU状态
                if not self._use_cpu_fallback:
                    self._gpu_failed_count = 0
                
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                is_oom = 'out of memory' in error_str or 'cuda out of memory' in error_str
                
                if is_oom:
                    print(f"[OOM] {operation_name} 显存不足 (尝试 {attempt + 1}/{max_attempts})")
                    
                    if self._handle_oom_error(operation_name):
                        # 可以重试，等待并清理
                        self._clear_gpu_cache()
                        time.sleep(2)  # 等待2秒
                        
                        # 如果之前在GPU上失败，这次尝试CPU
                        if self._use_cpu_fallback:
                            print(f"[OOM] 切换到CPU模式重试")
                            self.image_model = None
                            self.image_processor = None
                        continue
                    else:
                        # 已切换到CPU，用CPU重试一次
                        print(f"[OOM] 使用CPU模式重试 {operation_name}")
                        self.image_model = None
                        self.image_processor = None
                        try:
                            self._init_image_model(force_device='cpu')
                            return operation_func(*args, **kwargs)
                        except Exception as cpu_e:
                            print(f"[ERROR] CPU模式也失败: {cpu_e}")
                            traceback.print_exc()
                            return None
                else:
                    # 非OOM错误，直接抛出
                    raise e
        
        return None

    def segment_by_text(self, image_path: str, prompt: str, confidence: float = 0.5) -> list:
        """文本提示分割（支持OOM自动重试）"""
        # 尝试重置GPU状态
        self._reset_gpu_state()
        
        def _do_segment():
            self._load_image(image_path)
            
            print(f"[DEBUG] 文本分割: prompt='{prompt}', confidence={confidence}")

            # 设置置信度
            self.image_processor.confidence_threshold = confidence
            print(f"[DEBUG] 置信度阈值已设置为: {self.image_processor.confidence_threshold}")

            # 执行文本分割 - 直接使用 set_text_prompt，它会返回结果
            output = self.image_processor.set_text_prompt(
                state=self.inference_state,
                prompt=prompt
            )

            print(f"[DEBUG] 输出keys: {list(output.keys()) if output else 'None'}")
            if output:
                print(f"[DEBUG] output['masks'] shape: {output['masks'].shape if 'masks' in output else 'N/A'}")
                print(f"[DEBUG] output['boxes'] shape: {output['boxes'].shape if 'boxes' in output else 'N/A'}")
                print(f"[DEBUG] output['scores'] shape: {output['scores'].shape if 'scores' in output else 'N/A'}")
                if 'scores' in output:
                    scores_np = output['scores'].cpu().numpy()
                    print(f"[DEBUG] scores values: {scores_np[:10] if len(scores_np) > 0 else 'empty'}")

            return self._extract_results(output, prompt)

        try:
            result = self._execute_with_oom_retry(_do_segment, "文本分割")
            return result if result else []
        except Exception as e:
            print(f"[ERROR] segment_by_text: {e}")
            traceback.print_exc()
            return []

    def segment_by_points(self, image_path: str, points: list) -> list:
        """点击分割 - 支持正负样本点（支持OOM自动重试）

        正样本点：指示要分割的对象位置
        负样本点：指示不想要的区域（用于排除）

        策略：
        1. 正样本点转换为小框，用于触发分割
        2. 负样本点转换为小框，用于过滤结果
        """
        # 尝试重置GPU状态
        self._reset_gpu_state()
        
        def _do_segment():
            self._load_image(image_path)

            # 分离正负样本点
            positive_points = []
            negative_points = []

            print(f"[DEBUG] 点击分割: 共 {len(points)} 个点")
            for i, p in enumerate(points):
                x, y, label = p
                is_positive = label == 1
                label_str = "正样本(+)" if is_positive else "负样本(-)"
                print(f"[DEBUG]   点{i}: ({x:.1f}, {y:.1f}) {label_str}")

                if is_positive:
                    positive_points.append([x, y])
                else:
                    negative_points.append([x, y])

            print(f"[DEBUG] 正样本点: {len(positive_points)}, 负样本点: {len(negative_points)}")

            # 如果没有正样本点，无法分割
            if not positive_points:
                print("[DEBUG] 没有正样本点，无法分割")
                return []

            width, height = self._image_size

            # 将点转换为框
            boxes = []

            # 正样本点 -> 正样本框
            for x, y in positive_points:
                box_size = 15  # 稍小的框，更精确
                x1 = max(0, x - box_size)
                y1 = max(0, y - box_size)
                x2 = min(width, x + box_size)
                y2 = min(height, y + box_size)
                boxes.append([x1, y1, x2, y2, 1])  # label=1 正样本

            # 负样本点 -> 负样本框（用于过滤）
            for x, y in negative_points:
                box_size = 30  # 稍大的框，确保覆盖不想要的区域
                x1 = max(0, x - box_size)
                y1 = max(0, y - box_size)
                x2 = min(width, x + box_size)
                y2 = min(height, y + box_size)
                boxes.append([x1, y1, x2, y2, 0])  # label=0 负样本

            return self._segment_by_boxes_internal(image_path, boxes)
        
        try:
            result = self._execute_with_oom_retry(_do_segment, "点击分割")
            return result if result else []
        except Exception as e:
            print(f"[ERROR] segment_by_points: {e}")
            traceback.print_exc()
            return []

    def _mask_in_negative_region(self, mask: np.ndarray, negative_boxes: list, threshold: float = 0.5) -> bool:
        """检查 mask 是否主要位于负样本区域内

        Args:
            mask: 二值 mask (H, W)
            negative_boxes: 负样本框列表 [[x1, y1, x2, y2], ...]
            threshold: mask 在负样本区域内的比例阈值

        Returns:
            True 如果 mask 主要在负样本区域内，应该被排除
        """
        if not negative_boxes:
            return False

        mask_area = mask.sum()
        if mask_area == 0:
            return True  # 空 mask 直接排除

        # 创建负样本区域的 mask
        h, w = mask.shape
        negative_region = np.zeros((h, w), dtype=np.uint8)

        for box in negative_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            negative_region[y1:y2, x1:x2] = 1

        # 计算 mask 与负样本区域的重叠
        overlap = (mask > 0) & (negative_region > 0)
        overlap_area = overlap.sum()

        # 计算重叠比例（相对于 mask 面积）
        overlap_ratio = overlap_area / mask_area

        return overlap_ratio > threshold

    def _compute_iou(self, box1: list, box2: list) -> float:
        """计算两个框的标准 IoU (Intersection over Union)
        box 格式: [x1, y1, x2, y2]
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _compute_overlap_ratio(self, box1: list, box2: list) -> float:
        """计算重叠比例（相对于较小框的面积）
        box 格式: [x1, y1, x2, y2]
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        min_area = min(box1_area, box2_area)

        if min_area <= 0:
            return 0.0

        return inter_area / min_area

    def _boxes_overlap(self, box1: list, box2: list, threshold: float = 0.3, mode: str = 'iou') -> bool:
        """检查两个框是否重叠
        box 格式: [x1, y1, x2, y2]
        mode: 'iou' - 标准IoU, 'min_ratio' - 相对于小框, 'both' - 两者取最大
        """
        if mode == 'iou':
            iou = self._compute_iou(box1, box2)
            return iou > threshold
        elif mode == 'min_ratio':
            ratio = self._compute_overlap_ratio(box1, box2)
            return ratio > threshold
        else:  # 'both'
            iou = self._compute_iou(box1, box2)
            ratio = self._compute_overlap_ratio(box1, box2)
            return max(iou, ratio) > threshold

    def _mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个 mask 的 IoU"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def _deduplicate_results(self, results: list, iou_threshold: float = 0.4) -> list:
        """框级别的去重（作为NMS的后备机制）
        
        基于 bbox IoU 过滤重叠的检测结果，保留分数更高的
        """
        from config.segmentation_config import SegmentationConfig
        
        if len(results) <= 1:
            return results
        
        # 获取配置
        overlap_mode = getattr(SegmentationConfig, 'NMS_OVERLAP_MODE', 'iou')
        
        # 按分数降序排序
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        kept = []
        suppressed = set()
        
        for i, result_i in enumerate(sorted_results):
            if i in suppressed:
                continue
            
            kept.append(result_i)
            
            # 检查与后续结果的重叠
            for j in range(i + 1, len(sorted_results)):
                if j in suppressed:
                    continue
                    
                result_j = sorted_results[j]
                
                # 使用配置的重叠模式计算框重叠
                if self._boxes_overlap(result_i['bbox'], result_j['bbox'], 
                                       threshold=iou_threshold, mode=overlap_mode):
                    suppressed.add(j)
        
        if len(suppressed) > 0:
            print(f"[DEDUP] 框级别去重: 过滤了 {len(suppressed)} 个重叠结果")
        
        return kept

    def _deduplicate_results_with_mask(self, results: list, output: dict, iou_threshold: float = 0.4) -> list:
        """Mask 级别的去重（更精确）
        
        基于 mask IoU 过滤重叠的检测结果，保留分数更高的
        """
        if len(results) <= 1:
            return results
        
        masks = output.get('masks', [])
        if len(masks) != len(results):
            print("[DEDUP] mask数量与结果数量不匹配，回退到框级别去重")
            return self._deduplicate_results(results, iou_threshold)
        
        # 预处理所有 mask
        mask_list = []
        for mask in masks:
            mask_np = mask[0].cpu().numpy() if mask.dim() == 3 else mask.cpu().numpy()
            mask_list.append(mask_np > 0.5)
        
        # 按分数降序排序（同时保持索引对应）
        indexed_results = list(enumerate(results))
        sorted_results = sorted(indexed_results, key=lambda x: x[1]['score'], reverse=True)
        
        kept = []
        suppressed = set()
        
        for orig_i, result_i in sorted_results:
            if orig_i in suppressed:
                continue
            
            kept.append(result_i)
            mask_i = mask_list[orig_i]
            
            # 检查与后续结果的 mask 重叠
            for orig_j, result_j in sorted_results:
                if orig_j == orig_i or orig_j in suppressed:
                    continue
                
                mask_j = mask_list[orig_j]
                mask_iou = self._mask_iou(mask_i, mask_j)
                
                if mask_iou > iou_threshold:
                    suppressed.add(orig_j)
        
        if len(suppressed) > 0:
            print(f"[DEDUP] Mask级别去重: 过滤了 {len(suppressed)} 个重叠结果")
        
        return kept

    def segment_by_boxes(self, image_path: str, boxes: list) -> list:
        """框选分割 - 支持正负样本

        正样本框：用于指示要分割的区域
        负样本框：用于排除不想要的分割结果

        策略：
        1. 同时将正样本和负样本框传递给 SAM3（利用原生支持）
        2. 使用 mask 级别的后处理过滤（更精确）
        """
        try:
            self._init_image_model()
            self._load_image(image_path)

            # 分离正样本框和负样本框（原始像素坐标）
            positive_boxes_px = []
            negative_boxes_px = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box[:4]
                label = box[4] if len(box) > 4 else 1
                is_positive = bool(label)

                label_str = "正样本(+)" if is_positive else "负样本(-)"
                print(f"[DEBUG] 框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] {label_str}")

                if is_positive:
                    positive_boxes_px.append([x1, y1, x2, y2])
                else:
                    negative_boxes_px.append([x1, y1, x2, y2])

            print(f"[DEBUG] 正样本框: {len(positive_boxes_px)}, 负样本框: {len(negative_boxes_px)}")

            # 如果没有正样本框，无法分割
            if not positive_boxes_px:
                print("[DEBUG] 没有正样本框，无法分割")
                return []

            # 重置 geometric_prompt
            if "geometric_prompt" in self.inference_state:
                del self.inference_state["geometric_prompt"]
            print("[DEBUG] 已重置 geometric_prompt")

            width, height = self._image_size
            output = None

            # 先添加所有正样本框
            for i, box in enumerate(positive_boxes_px):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                norm_box = [cx, cy, w, h]
                print(f"[DEBUG] 添加正样本框{i}: {norm_box}")
                output = self.image_processor.add_geometric_prompt(
                    norm_box, True, self.inference_state
                )

            # 再添加所有负样本框（SAM3 原生支持）
            for i, box in enumerate(negative_boxes_px):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                norm_box = [cx, cy, w, h]
                print(f"[DEBUG] 添加负样本框{i}: {norm_box}")
                output = self.image_processor.add_geometric_prompt(
                    norm_box, False, self.inference_state
                )

            if output is None:
                return []

            print(f"[DEBUG] 输出keys: {list(output.keys()) if output else 'None'}")

            # 提取结果（带 mask 数据用于后处理）
            results = self._extract_results_with_mask(output, "box_prompt", negative_boxes_px)

            return results

        except Exception as e:
            print(f"[ERROR] segment_by_boxes: {e}")
            traceback.print_exc()
            return []

    def _segment_by_boxes_internal(self, image_path: str, boxes: list) -> list:
        """框选分割的内部实现（不含OOM重试逻辑）"""
        self._load_image(image_path)

        # 分离正样本框和负样本框（原始像素坐标）
        positive_boxes_px = []
        negative_boxes_px = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            label = box[4] if len(box) > 4 else 1
            is_positive = bool(label)

            label_str = "正样本(+)" if is_positive else "负样本(-)"
            print(f"[DEBUG] 框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] {label_str}")

            if is_positive:
                positive_boxes_px.append([x1, y1, x2, y2])
            else:
                negative_boxes_px.append([x1, y1, x2, y2])

        print(f"[DEBUG] 正样本框: {len(positive_boxes_px)}, 负样本框: {len(negative_boxes_px)}")

        # 如果没有正样本框，无法分割
        if not positive_boxes_px:
            print("[DEBUG] 没有正样本框，无法分割")
            return []

        # 重置 geometric_prompt
        if "geometric_prompt" in self.inference_state:
            del self.inference_state["geometric_prompt"]
        print("[DEBUG] 已重置 geometric_prompt")

        width, height = self._image_size
        output = None

        # 先添加所有正样本框
        for i, box in enumerate(positive_boxes_px):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            norm_box = [cx, cy, w, h]
            print(f"[DEBUG] 添加正样本框{i}: {norm_box}")
            output = self.image_processor.add_geometric_prompt(
                norm_box, True, self.inference_state
            )

        # 再添加所有负样本框（SAM3 原生支持）
        for i, box in enumerate(negative_boxes_px):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            norm_box = [cx, cy, w, h]
            print(f"[DEBUG] 添加负样本框{i}: {norm_box}")
            output = self.image_processor.add_geometric_prompt(
                norm_box, False, self.inference_state
            )

        if output is None:
            return []

        print(f"[DEBUG] 输出keys: {list(output.keys()) if output else 'None'}")

        # 提取结果（带 mask 数据用于后处理）
        results = self._extract_results_with_mask(output, "box_prompt", negative_boxes_px)

        return results

    def segment_by_boxes(self, image_path: str, boxes: list) -> list:
        """框选分割 - 支持正负样本（支持OOM自动重试）

        正样本框：用于指示要分割的区域
        负样本框：用于排除不想要的分割结果

        策略：
        1. 同时将正样本和负样本框传递给 SAM3（利用原生支持）
        2. 使用 mask 级别的后处理过滤（更精确）
        """
        # 尝试重置GPU状态
        self._reset_gpu_state()
        
        try:
            result = self._execute_with_oom_retry(
                self._segment_by_boxes_internal, "框选分割", 
                image_path, boxes
            )
            return result if result else []
        except Exception as e:
            print(f"[ERROR] segment_by_boxes: {e}")
            traceback.print_exc()
            return []

    def _extract_results_with_mask(self, output: dict, label: str, negative_boxes: list) -> list:
        """从输出提取结果，并使用 mask 级别过滤负样本区域"""
        results = []

        if output is None:
            print("[DEBUG] output is None")
            return results

        masks = output.get('masks', [])
        boxes = output.get('boxes', [])
        scores = output.get('scores', [])

        print(f"[DEBUG] 原始结果: masks={len(masks)}, boxes={len(boxes)}, scores={len(scores)}")

        # 还原负样本框到缩放后的图像尺寸（用于过滤）
        scale_factor = getattr(self, '_scale_factor', 1.0)
        negative_boxes_scaled = []
        if negative_boxes and scale_factor != 1.0:
            inverse_scale = 1.0 / scale_factor
            negative_boxes_scaled = [[v * inverse_scale for v in box] for box in negative_boxes]
        else:
            negative_boxes_scaled = negative_boxes

        filtered_count = 0
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            try:
                mask_np = mask[0].cpu().numpy()
                box_np = box.cpu().numpy().tolist()

                # 使用 mask 级别的负样本过滤
                if negative_boxes_scaled and self._mask_in_negative_region(mask_np, negative_boxes_scaled, threshold=0.4):
                    print(f"[DEBUG] 结果{i}: score={float(score):.4f} - 被负样本区域过滤")
                    filtered_count += 1
                    continue

                print(f"[DEBUG] 结果{i}: score={float(score):.4f}, bbox={[f'{v:.1f}' for v in box_np]}")

                polygon = self._mask_to_polygon(mask_np)

                # 还原坐标到原始图像尺寸
                if scale_factor != 1.0:
                    print(f"[DEBUG] 还原坐标，缩放因子: {scale_factor:.4f}")
                    # 还原 bbox
                    box_np = [v * scale_factor for v in box_np]
                    # 还原 polygon
                    polygon = [[x * scale_factor, y * scale_factor] for x, y in polygon]
                    # 还原 area
                    area = float(mask_np.sum() * scale_factor * scale_factor)
                else:
                    area = float(mask_np.sum())

                results.append({
                    'id': str(uuid.uuid4())[:8],
                    'label': label,
                    'score': float(score),
                    'bbox': box_np,
                    'polygon': polygon,
                    'area': area,
                })
            except Exception as e:
                print(f"[ERROR] 提取结果{i}失败: {e}")

        if (filtered_count > 0):
            print(f"[DEBUG] 负样本过滤: 排除了 {filtered_count} 个结果")

        return results

    def _extract_results(self, output: dict, label: str) -> list:
        """从输出提取结果"""
        results = []

        if output is None:
            print("[DEBUG] output is None")
            return results

        masks = output.get('masks', [])
        boxes = output.get('boxes', [])
        scores = output.get('scores', [])

        print(f"[DEBUG] 结果: masks={len(masks)}, boxes={len(boxes)}, scores={len(scores)}")

        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            try:
                mask_np = mask[0].cpu().numpy()
                box_np = box.cpu().numpy().tolist()

                print(f"[DEBUG] 结果{i}: score={float(score):.4f}, bbox={box_np}")

                polygon = self._mask_to_polygon(mask_np)

                # 还原坐标到原始图像尺寸
                scale_factor = getattr(self, '_scale_factor', 1.0)
                if scale_factor != 1.0:
                    print(f"[DEBUG] 还原坐标，缩放因子: {scale_factor:.4f}")
                    # 还原 bbox
                    box_np = [v * scale_factor for v in box_np]
                    # 还原 polygon
                    polygon = [[x * scale_factor, y * scale_factor] for x, y in polygon]
                    # 还原 area
                    area = float(mask_np.sum() * scale_factor * scale_factor)
                else:
                    area = float(mask_np.sum())

                results.append({
                    'id': str(uuid.uuid4())[:8],
                    'label': label,
                    'score': float(score),
                    'bbox': box_np,
                    'polygon': polygon,
                    'area': area,
                })
            except Exception as e:
                print(f"[ERROR] 提取结果{i}失败: {e}")

        # NMS去重
        from config.segmentation_config import SegmentationConfig
        if SegmentationConfig.ENABLE_NMS:
            if getattr(SegmentationConfig, 'NMS_MASK_LEVEL', False):
                # 使用更精确的 mask 级别 NMS
                results = self._deduplicate_results_with_mask(results, output, 
                                                              iou_threshold=SegmentationConfig.NMS_IOU_THRESHOLD)
            else:
                # 使用框级别 NMS
                results = self._deduplicate_results(results, iou_threshold=SegmentationConfig.NMS_IOU_THRESHOLD)

        return results

    # ==================== 视频分割 ====================

    def _init_video_model(self):
        if self.video_predictor is not None:
            return

        print("正在加载SAM3视频模型...")
        from sam3.model_builder import build_sam3_video_predictor

        gpus = range(torch.cuda.device_count()) if torch.cuda.is_available() else []
        self.video_predictor = build_sam3_video_predictor(gpus_to_use=gpus)
        print("SAM3视频模型加载完成")

    def start_video_session(self, video_path: str) -> str:
        self._init_video_model()
        response = self.video_predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = response["session_id"]
        self.video_sessions[session_id] = {'video_path': video_path, 'outputs': {}}
        return session_id

    def add_video_prompt(self, session_id: str, frame_index: int,
                         prompt_type: str, prompt_data) -> dict:
        self._init_video_model()

        request = {
            'type': 'add_prompt',
            'session_id': session_id,
            'frame_index': frame_index,
        }

        if prompt_type == 'text':
            request['text'] = prompt_data
        elif prompt_type == 'points':
            points = torch.tensor(prompt_data['points'], dtype=torch.float32)
            labels = torch.tensor(prompt_data['labels'], dtype=torch.int32)
            request['points'] = points
            request['point_labels'] = labels
            if 'obj_id' in prompt_data:
                request['obj_id'] = prompt_data['obj_id']

        response = self.video_predictor.handle_request(request=request)
        return response.get('outputs', {})

    def propagate_video(self, session_id: str) -> dict:
        self._init_video_model()
        outputs = {}
        for response in self.video_predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs[response["frame_index"]] = response["outputs"]
        return outputs

    def close_video_session(self, session_id: str):
        if self.video_predictor:
            self.video_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        self.video_sessions.pop(session_id, None)

    def shutdown(self):
        if self.video_predictor:
            self.video_predictor.shutdown()
        self.video_predictor = None
        self.image_model = None
        self.image_processor = None
    
    # ==================== 批量处理优化 ====================
    
    def batch_segment_by_text(self, image_paths: list, prompt: str, confidence: float = 0.5, 
                              max_size: int = 1024, use_cache: bool = True):
        """批量文本分割
        
        Args:
            image_paths: 图像路径列表
            prompt: 文本提示
            confidence: 置信度阈值
            max_size: 最大图像尺寸
            use_cache: 是否使用特征缓存
            
        Returns:
            dict: {image_path: [分割结果], ...}
        """
        results = {}
        
        print(f"[BATCH] 开始批量分割，共 {len(image_paths)} 张图像")
        print(f"[BATCH] 提示词: '{prompt}', 置信度: {confidence}")
        
        total_start = time.time()
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"\n[BATCH] 处理第 {i}/{len(image_paths)} 张: {image_path}")
                start_time = time.time()
                
                # 使用优化后的分割方法
                result = self.segment_by_text(image_path, prompt, confidence)
                seg_time = time.time() - start_time
                
                results[image_path] = result
                print(f"[BATCH] 完成: {seg_time:.2f}秒, 结果数: {len(result)}")
                
            except Exception as e:
                print(f"[ERROR] 批量分割失败 {image_path}: {e}")
                traceback.print_exc()
                results[image_path] = []
        
        total_time = time.time() - total_start
        avg_time = total_time / len(image_paths) if image_paths else 0
        
        print(f"\n[BATCH] 批量分割完成!")
        print(f"[BATCH] 总耗时: {total_time:.2f}秒")
        print(f"[BATCH] 平均每张: {avg_time:.2f}秒")
        print(f"[BATCH] 处理速度: {len(image_paths)/total_time:.2f} 张/秒")
        
        return results
    
    def segment_by_text_with_optimization(self, image_path: str, prompt: str, 
                                         confidence: float = 0.5, max_size: int = 1024):
        """优化版的文本分割方法，包含所有优化
        
        这是对原有 segment_by_text 方法的优化版本，集成了：
        1. 图像尺寸优化
        2. 特征缓存
        3. GPU加速（如果可用）
        
        Args:
            image_path: 图像路径
            prompt: 文本提示
            confidence: 置信度阈值
            max_size: 最大图像尺寸
            
        Returns:
            list: 分割结果列表
        """
        try:
            self._init_image_model()
            
            # 使用优化后的图像加载方法
            self._load_image(image_path, max_size=max_size, use_cache=True)
            
            print(f"[OPTIMIZED] 文本分割: prompt='{prompt}', confidence={confidence}")
            
            # 设置置信度
            self.image_processor.confidence_threshold = confidence
            
            # 执行文本分割
            output = self.image_processor.set_text_prompt(
                state=self.inference_state,
                prompt=prompt
            )
            
            print(f"[OPTIMIZED] 输出keys: {list(output.keys()) if output else 'None'}")
            
            return self._extract_results(output, prompt)
            
        except Exception as e:
            print(f"[ERROR] segment_by_text_with_optimization: {e}")
            traceback.print_exc()
            return []
