#!/bin/bash
# Stock Analyzer Phase 1 优化部署脚本

echo "======================================================================"
echo "Stock Analyzer Phase 1 优化部署"
echo "======================================================================"

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ 请先激活虚拟环境: source venv/bin/activate"
    exit 1
fi

# 运行测试
echo "🧪 运行优化模块测试..."
python test_optimization.py
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo "⚠️  部分测试未通过，但继续部署"
fi

# 启动服务
echo ""
echo "🚀 启动优化后的服务..."
echo "======================================================================"
echo "服务地址: http://127.0.0.1:5002"
echo "API地址:  http://127.0.0.1:5002/api/stock/sz300620"
echo "健康检查: http://127.0.0.1:5002/api/health"
echo "优化状态: http://127.0.0.1:5002/api/optimization/status"
echo "======================================================================"
echo ""

# 启动服务
python run_server.py