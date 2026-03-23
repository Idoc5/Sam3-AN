"""
性能监控工具
用于跟踪和报告系统性能指标
"""
import time
import torch
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, name: str = "Performance"):
        self.name = name
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.markers = defaultdict(float)
        self.total_operations = 0

    def start(self, operation: str) -> float:
        """开始计时"""
        self.start_times[operation] = time.time()
        return self.start_times[operation]

    def end(self, operation: str) -> Optional[float]:
        """结束计时并返回耗时"""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.metrics[operation].append(elapsed)
            del self.start_times[operation]
            self.total_operations += 1
            return elapsed
        return None

    def record(self, operation: str, value: float):
        """直接记录一个值"""
        self.metrics[operation].append(value)

    def marker(self, name: str):
        """设置时间标记"""
        self.markers[name] = time.time()

    def elapsed_since(self, marker: str) -> float:
        """获取从标记到现在的经过时间"""
        if marker in self.markers:
            return time.time() - self.markers[marker]
        return 0.0

    def get_stats(self, operation: str) -> Dict:
        """获取单个操作的统计信息"""
        values = self.metrics.get(operation, [])
        if not values:
            return {}

        return {
            'name': operation,
            'count': len(values),
            'total': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'std': self._stddev(values)
        }

    def get_all_stats(self) -> Dict[str, Dict]:
        """获取所有操作的统计信息"""
        return {
            op: self.get_stats(op)
            for op in self.metrics.keys()
        }

    def _stddev(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def get_gpu_memory(self) -> Dict:
        """获取GPU显存信息"""
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return {
                    'allocated_gb': round(allocated, 2),
                    'reserved_gb': round(reserved, 2),
                    'free_gb': round(total - reserved, 2),
                    'total_gb': round(total, 2),
                    'utilization_percent': round((reserved / total) * 100, 1)
                }
            except Exception as e:
                return {'error': str(e)}
        return {}

    def print_summary(self, show_gpu: bool = True):
        """打印性能摘要"""
        print("\n" + "=" * 70)
        print(f"{self.name} - 性能监控摘要")
        print("=" * 70)

        for op in sorted(self.metrics.keys()):
            stats = self.get_stats(op)
            if stats:
                print(f"\n【{op}】")
                print(f"  调用次数: {stats['count']}")
                print(f"  总耗时: {stats['total']:.2f}秒")
                print(f"  平均耗时: {stats['avg']:.3f}秒")
                print(f"  最小耗时: {stats['min']:.3f}秒")
                print(f"  最大耗时: {stats['max']:.3f}秒")
                if stats['std'] > 0:
                    print(f"  标准差: {stats['std']:.3f}秒")

        if show_gpu:
            gpu_info = self.get_gpu_memory()
            if gpu_info and 'error' not in gpu_info:
                print(f"\n【GPU显存】")
                print(f"  总显存: {gpu_info['total_gb']} GB")
                print(f"  已保留: {gpu_info['reserved_gb']} GB ({gpu_info['utilization_percent']}%)")
                print(f"  已分配: {gpu_info['allocated_gb']} GB")
                print(f"  可用: {gpu_info['free_gb']} GB")

        print("=" * 70 + "\n")

    def export_report(self, filepath: str = None):
        """导出报告到文件"""
        if filepath is None:
            filepath = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{self.name} - 性能报告\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")

            for op in sorted(self.metrics.keys()):
                stats = self.get_stats(op)
                if stats:
                    f.write(f"【{op}】\n")
                    f.write(f"  调用次数: {stats['count']}\n")
                    f.write(f"  总耗时: {stats['total']:.2f}秒\n")
                    f.write(f"  平均耗时: {stats['avg']:.3f}秒\n")
                    f.write(f"  最小耗时: {stats['min']:.3f}秒\n")
                    f.write(f"  最大耗时: {stats['max']:.3f}秒\n")
                    f.write(f"  标准差: {stats['std']:.3f}秒\n\n")

            gpu_info = self.get_gpu_memory()
            if gpu_info and 'error' not in gpu_info:
                f.write(f"【GPU显存】\n")
                f.write(f"  总显存: {gpu_info['total_gb']} GB\n")
                f.write(f"  已保留: {gpu_info['reserved_gb']} GB ({gpu_info['utilization_percent']}%)\n")
                f.write(f"  已分配: {gpu_info['allocated_gb']} GB\n")
                f.write(f"  可用: {gpu_info['free_gb']} GB\n\n")

        print(f"性能报告已导出到: {filepath}")


class BatchPerformanceMonitor(PerformanceMonitor):
    """批量操作专用性能监控器"""

    def __init__(self, name: str = "BatchPerformance", batch_size: int = 10):
        super().__init__(name)
        self.batch_size = batch_size
        self.completed_batches = 0

    def start_batch(self):
        """开始一个批次"""
        self.marker(f"batch_{self.completed_batches}")
        self.start(f"batch_{self.completed_batches}")

    def end_batch(self) -> float:
        """结束一个批次"""
        elapsed = self.end(f"batch_{self.completed_batches}")
        if elapsed is not None:
            self.completed_batches += 1
        return elapsed

    def get_batch_stats(self) -> List[Dict]:
        """获取所有批次的统计信息"""
        batch_stats = []
        for i in range(self.completed_batches):
            op = f"batch_{i}"
            stats = self.get_stats(op)
            if stats:
                stats['batch_index'] = i
                batch_stats.append(stats)
        return batch_stats

    def print_batch_summary(self):
        """打印批次摘要"""
        print("\n" + "=" * 70)
        print(f"{self.name} - 批次处理摘要")
        print("=" * 70)

        batch_stats = self.get_batch_stats()
        if batch_stats:
            total_time = sum(s['total'] for s in batch_stats)
            avg_time = sum(s['avg'] for s in batch_stats) / len(batch_stats)

            print(f"总批次数: {len(batch_stats)}")
            print(f"总耗时: {total_time:.2f}秒")
            print(f"平均每批次: {avg_time:.3f}秒")
            print(f"最快批次: {min(s['avg'] for s in batch_stats):.3f}秒")
            print(f"最慢批次: {max(s['avg'] for s in batch_stats):.3f}秒")
            print(f"平均吞吐量: {len(batch_stats) * self.batch_size / total_time:.2f} 张/秒")

        print("=" * 70 + "\n")


def timing_decorator(monitor: PerformanceMonitor = None, operation: str = None):
    """计时装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal operation
            if operation is None:
                operation = func.__name__

            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            if monitor:
                monitor.record(operation, elapsed)

            return result
        return wrapper
    return decorator
