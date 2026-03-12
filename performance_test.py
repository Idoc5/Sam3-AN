#!/usr/bin/env python3
"""
SAM3-AN 性能测试脚本
用于分析分割性能并找出瓶颈
"""

import time
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import traceback

# 添加SAM3到路径
sam3_src = Path(__file__).parent / "SAM_src"
sys.path.insert(0, str(sam3_src))

from services.sam3_service import SAM3Service


def create_test_image():
    """创建一个简单的测试图像"""
    img_path = Path(__file__).parent / "data" / "test_image.png"
    img_path.parent.mkdir(exist_ok=True, parents=True)
    
    # 创建一个 512x512 的测试图像，包含几个简单形状
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    
    # 画几个简单形状
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
    draw.ellipse([200, 200, 300, 300], fill='blue', outline='black')
    draw.polygon([(350, 100), (400, 200), (300, 200)], fill='green', outline='black')
    
    img.save(img_path)
    print(f"创建测试图像: {img_path}")
    return str(img_path)


def test_model_loading():
    """测试模型加载性能"""
    print("\n" + "="*50)
    print("测试模型加载性能")
    print("="*50)
    
    start_time = time.time()
    service = SAM3Service()
    service._init_image_model()
    load_time = time.time() - start_time
    
    print(f"模型加载时间: {load_time:.2f}秒")
    return service


def test_image_loading(service, image_path):
    """测试图像加载性能"""
    print("\n" + "="*50)
    print("测试图像加载性能")
    print("="*50)
    
    start_time = time.time()
    service._load_image(image_path)
    load_time = time.time() - start_time
    
    print(f"图像加载时间: {load_time:.2f}秒")
    return load_time


def test_text_segmentation(service, image_path, prompt="red rectangle"):
    """测试文本分割性能"""
    print("\n" + "="*50)
    print("测试文本分割性能")
    print("="*50)
    
    start_time = time.time()
    results = service.segment_by_text(image_path, prompt, 0.5)
    seg_time = time.time() - start_time
    
    print(f"文本分割时间: {seg_time:.2f}秒")
    print(f"分割结果数量: {len(results)}")
    
    if results:
        print(f"第一个结果的置信度: {results[0]['score']:.4f}")
        print(f"第一个结果的面积: {results[0]['area']:.0f}像素")
        print(f"第一个结果的多边形点数: {len(results[0]['polygon'])}")
    
    return seg_time, results


def test_point_segmentation(service, image_path):
    """测试点击分割性能"""
    print("\n" + "="*50)
    print("测试点击分割性能")
    print("="*50)
    
    # 在红色矩形中心添加正样本点
    points = [[100, 100, 1]]  # (x, y, label=1 表示正样本)
    
    start_time = time.time()
    results = service.segment_by_points(image_path, points)
    seg_time = time.time() - start_time
    
    print(f"点击分割时间: {seg_time:.2f}秒")
    print(f"分割结果数量: {len(results)}")
    
    return seg_time, results


def test_box_segmentation(service, image_path):
    """测试框选分割性能"""
    print("\n" + "="*50)
    print("测试框选分割性能")
    print("="*50)
    
    # 在红色矩形周围添加正样本框
    boxes = [[40, 40, 160, 160, 1]]  # (x1, y1, x2, y2, label=1)
    
    start_time = time.time()
    results = service.segment_by_boxes(image_path, boxes)
    seg_time = time.time() - start_time
    
    print(f"框选分割时间: {seg_time:.2f}秒")
    print(f"分割结果数量: {len(results)}")
    
    return seg_time, results


def test_mask_to_polygon_performance(service):
    """测试mask到多边形转换性能"""
    print("\n" + "="*50)
    print("测试mask到多边形转换性能")
    print("="*50)
    
    # 创建一个简单的测试mask
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[50:150, 50:150] = 255  # 红色矩形区域
    
    # 测试不同平滑级别
    smooth_levels = ['none', 'low', 'medium', 'high', 'ultra']
    
    for level in smooth_levels:
        start_time = time.time()
        polygon = service._mask_to_polygon(test_mask, smooth_level=level)
        conv_time = time.time() - start_time
        
        print(f"平滑级别 '{level}': {conv_time:.4f}秒, 多边形点数: {len(polygon)}")


def profile_memory_usage():
    """分析内存使用情况"""
    print("\n" + "="*50)
    print("分析内存使用情况")
    print("="*50)
    
    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
        
        # 显存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"已分配显存: {allocated:.2f} GB")
        print(f"已保留显存: {reserved:.2f} GB")
    else:
        print("CUDA可用: 否")
        print("当前使用CPU模式")


def run_comprehensive_test():
    """运行全面的性能测试"""
    print("SAM3-AN 性能测试")
    print("="*60)
    
    # 创建测试图像
    image_path = create_test_image()
    
    try:
        # 1. 测试模型加载
        service = test_model_loading()
        
        # 2. 分析内存使用
        profile_memory_usage()
        
        # 3. 测试图像加载
        test_image_loading(service, image_path)
        
        # 4. 测试各种分割方法
        text_time, text_results = test_text_segmentation(service, image_path)
        point_time, point_results = test_point_segmentation(service, image_path)
        box_time, box_results = test_box_segmentation(service, image_path)
        
        # 5. 测试mask转换性能
        test_mask_to_polygon_performance(service)
        
        # 6. 性能总结
        print("\n" + "="*50)
        print("性能测试总结")
        print("="*50)
        print(f"文本分割平均时间: {text_time:.2f}秒")
        print(f"点击分割平均时间: {point_time:.2f}秒")
        print(f"框选分割平均时间: {box_time:.2f}秒")
        
        # 识别性能瓶颈
        if text_time > 5.0:
            print("\n⚠️ 警告: 文本分割时间过长 (>5秒)")
            print("建议优化: 检查模型推理配置，考虑使用GPU加速")
        
        if point_time > 3.0:
            print("\n⚠️ 警告: 点击分割时间过长 (>3秒)")
            print("建议优化: 简化点转框逻辑，减少不必要的计算")
        
        if box_time > 3.0:
            print("\n⚠️ 警告: 框选分割时间过长 (>3秒)")
            print("建议优化: 优化几何提示处理逻辑")
        
        # 检查是否使用CPU模式
        if not torch.cuda.is_available():
            print("\n⚠️ 注意: 当前运行在CPU模式下，性能会显著下降")
            print("建议: 确保CUDA环境正确配置，使用GPU加速")
        else:
            print("\n✅ 当前运行在GPU模式下")
        
        return {
            'text_time': text_time,
            'point_time': point_time,
            'box_time': box_time,
            'text_results': len(text_results),
            'point_results': len(point_results),
            'box_results': len(box_results)
        }
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("开始SAM3-AN性能测试...")
    results = run_comprehensive_test()
    
    if results:
        print("\n✅ 性能测试完成!")
    else:
        print("\n❌ 性能测试失败!")