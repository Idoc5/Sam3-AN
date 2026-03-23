#!/usr/bin/env python3
"""
SAM3-AN 性能优化脚本
提供一键优化和环境检查功能
"""
import sys
import torch
import psutil
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "SAM_src"))


def check_environment():
    """检查系统环境"""
    print("=" * 60)
    print("SAM3-AN 环境检查")
    print("=" * 60)

    # 1. Python版本
    print(f"\n✅ Python版本: {sys.version.split()[0]}")

    # 2. PyTorch版本
    print(f"✅ PyTorch版本: {torch.__version__}")

    # 3. CUDA检查
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: 是")
        print(f"   - CUDA版本: {torch.version.cuda}")
        print(f"   - GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   - GPU {i}: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print(f"⚠️  CUDA不可用，将使用CPU模式（性能较差）")

    # 4. 内存检查
    mem = psutil.virtual_memory()
    print(f"\n✅ 系统内存: {mem.total / 1024**3:.1f}GB")
    print(f"   - 可用内存: {mem.available / 1024**3:.1f}GB")

    # 5. CPU检查
    print(f"\n✅ CPU核心数: {psutil.cpu_count()}")

    # 6. 磁盘空间检查
    disk = psutil.disk_usage('/')
    print(f"✅ 磁盘空间: {disk.free / 1024**3:.1f}GB 可用")

    # 7. 模型文件检查
    model_path = project_root / "sam3.pt"
    if model_path.exists():
        model_size = model_path.stat().st_size / 1024**3
        print(f"\n✅ 模型文件: {model_path} ({model_size:.2f}GB)")
    else:
        print(f"\n⚠️  模型文件不存在: {model_path}")

    print("=" * 60)


def benchmark_segmentation(num_images=5):
    """性能测试"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    try:
        from services.sam3_service import SAM3Service

        service = SAM3Service()

        # 测试图像处理速度
        test_image = project_root / "test_image.png"
        if not test_image.exists():
            print(f"⚠️  测试图像不存在: {test_image}")
            print("跳过性能测试")
            return

        print(f"\n测试图像: {test_image}")

        # 预热
        print("预热模型...")
        service.segment_by_text(str(test_image), "test", 0.5)

        # 性能测试
        print(f"\n开始性能测试（{num_images}次）...")
        times = []

        for i in range(num_images):
            start = time.time()
            result = service.segment_by_text(str(test_image), "test", 0.5)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  测试 {i+1}: {elapsed:.3f}秒, 结果数: {len(result)}")

        # 统计
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\n结果:")
        print(f"  平均耗时: {avg_time:.3f}秒")
        print(f"  最快: {min_time:.3f}秒")
        print(f"  最慢: {max_time:.3f}秒")
        print(f"  吞吐量: {1/avg_time:.2f} 张/秒")

        # 性能评级
        if torch.cuda.is_available():
            if avg_time < 0.5:
                grade = "🚀 优秀"
            elif avg_time < 1.0:
                grade = "✅ 良好"
            elif avg_time < 2.0:
                grade = "⚠️  一般"
            else:
                grade = "❌ 较慢"
        else:
            if avg_time < 5.0:
                grade = "✅ CPU表现良好"
            elif avg_time < 10.0:
                grade = "⚠️  CPU表现一般"
            else:
                grade = "❌ CPU性能较差"

        print(f"  性能评级: {grade}")

        # 建议
        print(f"\n优化建议:")
        if torch.cuda.is_available():
            print("  ✅ 已启用GPU加速")
            if avg_time > 1.0:
                print("  - 尝试降低 MAX_IMAGE_SIZE")
                print("  - 检查GPU显存使用情况")
        else:
            print("  ⚠️  使用CPU模式，建议:")
            print("  - 降低 MAX_IMAGE_SIZE 到 512-768")
            print("  - 启用模型量化（ENABLE_MODEL_QUANTIZATION=True）")
            print("  - 如有条件，使用GPU可提升10-50倍性能")

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


def optimize_config():
    """根据硬件自动优化配置"""
    print("\n" + "=" * 60)
    print("自动优化配置")
    print("=" * 60)

    # 导入配置
    try:
        from config.performance_config import PerformanceConfig

        # 根据CUDA可用性调整配置
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n检测到GPU: {gpu_mem:.1f}GB")

            # 根据显存大小调整图像尺寸
            if gpu_mem >= 24:
                PerformanceConfig.MAX_IMAGE_SIZE = 1536
                print("  ✅ 设置 MAX_IMAGE_SIZE = 1536（高精度）")
            elif gpu_mem >= 12:
                PerformanceConfig.MAX_IMAGE_SIZE = 1024
                print("  ✅ 设置 MAX_IMAGE_SIZE = 1024（平衡）")
            elif gpu_mem >= 8:
                PerformanceConfig.MAX_IMAGE_SIZE = 768
                print("  ✅ 设置 MAX_IMAGE_SIZE = 768（节省显存）")
            else:
                PerformanceConfig.MAX_IMAGE_SIZE = 512
                print("  ✅ 设置 MAX_IMAGE_SIZE = 512（低显存）")

            # 启用AMP
            PerformanceConfig.ENABLE_AMP = True
            print("  ✅ 启用自动混合精度（AMP）")

        else:
            print(f"\n未检测到GPU，使用CPU模式")
            PerformanceConfig.MAX_IMAGE_SIZE = 512
            PerformanceConfig.ENABLE_AMP = False
            PerformanceConfig.ENABLE_MODEL_QUANTIZATION = True

            print("  ✅ 设置 MAX_IMAGE_SIZE = 512")
            print("  ✅ 启用模型量化")
            print("  ⚠️  CPU模式性能较差，建议使用GPU")

        # 缓存配置
        mem = psutil.virtual_memory().total / 1024**3
        if mem >= 32:
            PerformanceConfig.CACHE_MAX_SIZE = 15
            print(f"  ✅ 设置 CACHE_MAX_SIZE = 15")
        elif mem >= 16:
            PerformanceConfig.CACHE_MAX_SIZE = 10
            print(f"  ✅ 设置 CACHE_MAX_SIZE = 10")

        print(f"\n配置已优化！")
        print(f"提示: 这些配置可以在 config/performance_config.py 中手动调整")

    except ImportError:
        print("⚠️  性能配置文件不存在，跳过自动优化")

    print("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='SAM3-AN 性能优化工具')
    parser.add_argument('--check', action='store_true', help='检查环境')
    parser.add_argument('--benchmark', type=int, default=5, metavar='N',
                       help='性能测试（默认5次）')
    parser.add_argument('--optimize', action='store_true', help='自动优化配置')
    parser.add_argument('--all', action='store_true', help='执行所有检查')

    args = parser.parse_args()

    if args.all:
        check_environment()
        optimize_config()
        benchmark_segmentation(args.benchmark)
    elif args.check:
        check_environment()
    elif args.optimize:
        optimize_config()
    elif args.benchmark:
        benchmark_segmentation(args.benchmark)
    else:
        # 默认执行所有检查
        check_environment()
        optimize_config()
        benchmark_segmentation(args.benchmark)


if __name__ == '__main__':
    main()
