"""
PEARL Benchmark Tool
主入口文件，调用其他模块执行 benchmark 测试
"""
import sys
import os

# 添加当前目录到路径，以便导入 benchmark 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from benchmark.bench_args import parse_args
from benchmark.bench_runner import run_benchmark


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)