"""
Benchmark Arguments Parser
处理 benchmark 工具的命令行参数
"""
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PEARL Benchmark Tool')
    
    # Model arguments
    parser.add_argument('--draft-model', '-d', type=str, required=True,
                       help='Draft model path (required)')
    parser.add_argument('--target-model', '-t', type=str, required=True,
                       help='Target model path (required)')
    parser.add_argument('--draft-tp', type=int, default=1,
                       help='Draft model tensor parallel size (default: 1)')
    parser.add_argument('--target-tp', type=int, default=2,
                       help='Target model tensor parallel size (default: 2)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization (default: 0.9)')
    
    # Generation arguments
    parser.add_argument('--temperature', '-temp', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate (default: 200)')
    parser.add_argument('--ignore-eos', '-noeos', action='store_true',
                       help='Ignore EOS token (default: False)')
    
    # Benchmark arguments
    parser.add_argument('--run-ar-benchmark', '-ar', action='store_true',
                       help='Run AR (Autoregressive) benchmark (default: False)')
    parser.add_argument('--custom-prompts', '-p', type=str, nargs='+',
                       help='Custom prompts for benchmark')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output (default: False)')
    
    # Power monitoring arguments
    parser.add_argument('--monitor-power', action='store_true',
                       help='Enable GPU power monitoring (default: True)')
    parser.add_argument('--power-sample-interval', type=float, default=0.1,
                       help='Power monitoring sample interval in seconds (default: 0.5)')
    parser.add_argument('--export-power-csv', type=str, default=None,
                       help='Export power monitoring data to CSV file')
    
    # Prompt loading arguments
    parser.add_argument('--max-prompts', type=int, default=100,
                       help='Maximum number of prompts to load (default: 100, set to 0 to load all)')
    parser.add_argument('--random-seed-prompts', type=int, default=None,
                       help='Random seed for prompt sampling (default: None, uses --seed)')
    
    # Rate and duration control arguments
    parser.add_argument('--target-duration', type=float, default=300,
                       help='Target experiment duration in seconds. Will estimate prompts needed based on throughput (default: None)')
    parser.add_argument('--max-duration', type=float, default=None,
                       help='Maximum experiment duration in seconds. Experiment will stop after this time (default: None)')
    parser.add_argument('--min-duration', type=float, default=None,
                       help='Minimum experiment duration in seconds. Experiment will continue until at least this time (default: None)')
    parser.add_argument('--request-rate', type=float, default=1.0,
                       help='Request sending rate (requests per second, QPS). If set, requests will be sent at this rate (default: None, send all at once)')
    parser.add_argument('--batch-interval', type=float, default=None,
                       help='Interval between batches in seconds. If set, requests will be sent in batches with this interval (default: None)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for sending requests. Used with --batch-interval (default: None, use all prompts as one batch)')
    
    return parser.parse_args()

