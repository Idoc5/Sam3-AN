#!/bin/bash
# SAM3-AN 快速优化脚本

set -e

echo "=========================================="
echo "SAM3-AN 快速优化"
echo "=========================================="

# 1. 检查环境
echo ""
echo "1️⃣  检查系统环境..."
python3 optimize_performance.py --check

# 2. 自动优化配置
echo ""
echo "2️⃣  自动优化配置..."
python3 optimize_performance.py --optimize

# 3. 安装性能监控依赖
echo ""
echo "3️⃣  安装性能监控依赖..."
pip3 install psutil -q

# 4. 测试性能
echo ""
echo "4️⃣  性能基准测试..."
python3 optimize_performance.py --benchmark 3

echo ""
echo "=========================================="
echo "✅ 优化完成！"
echo "=========================================="
echo ""
echo "📚 查看详细说明: cat PERFORMANCE_TIPS.md"
echo "🔧 修改配置: vim config/performance_config.py"
echo "🚀 启动服务: python3 app.py"
echo "📊 查看性能: python3 optimize_performance.py --all"
echo ""
