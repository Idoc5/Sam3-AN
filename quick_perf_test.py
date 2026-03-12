#!/usr/bin/env python3
"""
快速性能测试 - 只测试关键瓶颈
"""

import time
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# 添加SAM3到路径
sam3_src = Path(__file__).parent / "SAM_src"
sys.path.insert(0, str(sam3_src))

def quick_test():
    print("快速性能测试")
    print("="*50)
    
    # 1. 检查CUDA可用性
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("⚠️ 警告: CUDA不可用，将使用CPU模式")
    
    # 2. 检查模型文件
    model_path = Path(__file__).parent / "sam3.pt"
    print(f"模型文件存在: {model_path.exists()}")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024*1024)
        print(f"模型文件大小: {size_mb:.1f} MB")
    
    # 3. 测试图像处理性能
    print("\n测试图像处理性能:")
    
    # 创建测试图像
    test_img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([50, 50, 150, 150], fill='red')
    
    # 测试OpenCV导入（用于mask处理）
    try:
        import cv2
        print(f"OpenCV版本: {cv2.__version__}")
        
        # 测试mask处理速度
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        
        start = time.time()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        contour_time = time.time() - start
        
        print(f"轮廓检测时间: {contour_time:.4f}秒")
        
        if contours:
            # 测试多边形简化
            contour = contours[0]
            start = time.time()
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            simplify_time = time.time() - start
            
            print(f"多边形简化时间: {simplify_time:.4f}秒")
            print(f"简化后点数: {len(approx)}")
        
    except ImportError:
        print("❌ OpenCV未安装，mask处理性能会受影响")
    
    # 4. 分析可能的性能瓶颈
    print("\n性能瓶颈分析:")
    
    bottlenecks = []
    
    if not torch.cuda.is_available():
        bottlenecks.append("CPU模式 - SAM3模型在CPU上运行非常慢")
    
    # 检查是否有大图像处理
    img_size = test_img.size
    if img_size[0] * img_size[1] > 1024*1024:  # 超过1百万像素
        bottlenecks.append(f"大图像({img_size[0]}x{img_size[1]}) - 考虑缩小图像尺寸")
    
    # 检查mask处理复杂度
    if 'cv2' not in locals():
        bottlenecks.append("缺少OpenCV - mask处理使用纯Python实现，速度慢")
    
    if bottlenecks:
        print("⚠️ 发现以下性能瓶颈:")
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"  {i}. {bottleneck}")
    else:
        print("✅ 未发现明显性能瓶颈")
    
    # 5. 建议优化措施
    print("\n优化建议:")
    suggestions = []
    
    if not torch.cuda.is_available():
        suggestions.append("1. 配置CUDA环境，启用GPU加速")
        suggestions.append("2. 如果无法使用GPU，考虑使用更小的模型或量化模型")
    
    suggestions.append("3. 对于大图像，在分割前先缩放到合适尺寸")
    suggestions.append("4. 确保OpenCV已安装，用于快速mask处理")
    suggestions.append("5. 考虑缓存图像特征，避免重复计算")
    suggestions.append("6. 批量处理图像时，使用异步处理")
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

if __name__ == "__main__":
    quick_test()