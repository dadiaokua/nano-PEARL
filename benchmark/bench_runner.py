"""
Benchmark Runner
执行 benchmark 测试的核心逻辑
"""
import copy
from random import seed
from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams, logger
from nano_pearl.utils.prompt_loader import get_default_prompts, estimate_prompts_needed
from nano_pearl.utils.gpu_power_monitor import GPUPowerMonitor


def run_warmup(engine):
    """
    运行 warmup 测试
    
    Returns:
        warmup_throughput: warmup 的吞吐量 (tokens/s)
    """
    prompt = "Benchmark:"
    sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=512)
    engine.add_request(prompt, sampling_params)
    output_text, num_tokens, num_acc_tokens, elapsed_time = engine.generate()
    MAT = [sum(n) / len(n) for n in num_acc_tokens]
    warmup_throughput = sum(num_tokens) / elapsed_time if elapsed_time > 0 else 0
    logger.info(f"[Warmup] Total: {sum(num_tokens)}tok, Time: {elapsed_time:.2f}s, Throughput: {warmup_throughput:.2f}tok/s, MAT: {MAT}")
    return warmup_throughput


def prepare_prompts(args, warmup_throughput=0):
    """
    准备 prompts，根据参数和目标时长进行智能采样
    
    Returns:
        prompts: prompts 列表
        estimated_tokens_per_prompt: 估算的每个 prompt 的 token 数
    """
    # 估算每个 prompt 会生成多少 token
    estimated_tokens_per_prompt = 100 + args.max_tokens  # prompt 平均 100 tokens + max_tokens
    
    # 根据目标时长和 warmup 吞吐量估算需要的 prompts 数量
    if args.target_duration and not args.custom_prompts:
        estimated_prompts_needed = estimate_prompts_needed(
            warmup_throughput, 
            args.target_duration, 
            estimated_tokens_per_prompt
        )
        logger.info(f"根据 warmup 吞吐量 ({warmup_throughput:.2f} tok/s) 和目标时长 ({args.target_duration}s)")
        logger.info(f"估算需要约 {estimated_prompts_needed} 条 prompts (假设每条 {estimated_tokens_per_prompt} tokens)")
        # 使用估算值或用户指定的 max_prompts，取较小值
        effective_max_prompts = min(estimated_prompts_needed, args.max_prompts) if args.max_prompts > 0 else estimated_prompts_needed
    else:
        effective_max_prompts = args.max_prompts if args.max_prompts > 0 else None
    
    # 加载 prompts
    if args.custom_prompts:
        prompts = args.custom_prompts
    else:
        prompts = get_default_prompts(
            max_prompts=effective_max_prompts,
            random_seed=args.random_seed_prompts if args.random_seed_prompts is not None else args.seed
        )
    
    if args.target_duration and not args.custom_prompts:
        estimated_duration = len(prompts) * estimated_tokens_per_prompt / warmup_throughput if warmup_throughput > 0 else 0
        logger.info(f"实际加载了 {len(prompts)} 条 prompts，预计实验时长约 {estimated_duration:.1f} 秒")
    
    return prompts, estimated_tokens_per_prompt


def run_pearl_benchmark(engine, prompts, sampling_params, verbose=False):
    """
    运行 PEARL benchmark 测试
    
    Returns:
        output_text: 生成的文本列表
        num_tokens: 每个序列的 token 数量列表
        num_acc_tokens: 每个序列的接受 token 数量列表
        elapsed_time: 耗时（秒）
        throughput: 吞吐量 (tokens/s)
    """
    # 添加所有请求
    for prompt in prompts:
        engine.add_request(prompt, copy.deepcopy(sampling_params))
    
    # 运行 benchmark
    output_text, num_tokens, num_acc_tokens, elapsed_time = engine.bench_generate(num_pearl_steps=100)
    MAT = [sum(n) / len(n) for n in num_acc_tokens]
    throughput = sum(num_tokens) / elapsed_time if elapsed_time > 0 else 0
    
    # 输出结果
    if verbose:
        for prompt, output_text_item in zip(prompts, output_text):
            logger.info(f"Prompt: \n{prompt}", color="yellow")
            logger.info(f"Completion: \n{output_text_item}")
    
    logger.info(f"num_tokens: {num_tokens}, MAT: {MAT}")
    logger.info(f"[PEARL Generate] Batch Size: {len(prompts)} Total: {sum(num_tokens)}tok, Time: {elapsed_time:.2f}s, Throughput: {throughput:.2f}tok/s, MAT: {MAT}")
    
    return output_text, num_tokens, num_acc_tokens, elapsed_time, throughput


def run_ar_benchmark(engine, prompts, sampling_params, verbose=False):
    """
    运行 AR (Auto-Regressive) benchmark 测试
    
    Returns:
        output_text: 生成的文本列表
        num_tokens: 每个序列的 token 数量列表
        elapsed_time: 耗时（秒）
        throughput: 吞吐量 (tokens/s)
    """
    # 添加所有请求
    for prompt in prompts:
        engine.add_request(prompt, copy.deepcopy(sampling_params))
    
    # 运行 AR 生成
    output_text, num_tokens, _, elapsed_time = engine.AR_generate()
    throughput = sum(num_tokens) / elapsed_time if elapsed_time > 0 else 0
    
    # 输出结果
    if verbose:
        for prompt, output_text_item in zip(prompts, output_text):
            logger.info(f"Prompt: \n{prompt}", color="yellow")
            logger.info(f"Completion: \n{output_text_item}")
    
    logger.info(f"[AR Generate] Batch Size: {len(prompts)} Total: {sum(num_tokens)}tok, Time: {elapsed_time:.2f}s, Throughput: {throughput:.2f}tok/s")
    
    return output_text, num_tokens, elapsed_time, throughput


def run_benchmark(args):
    """
    运行完整的 benchmark 测试
    
    Args:
        args: 命令行参数对象
    """
    # 设置随机种子
    seed(args.seed)
    
    # 初始化引擎
    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp,
        target_tensor_parallel_size=args.target_tp,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    engine = PEARLEngine(config)
    
    # 初始化功耗监控
    power_monitor = None
    if args.monitor_power:
        gpu_ids = list(range(args.draft_tp + args.target_tp))
        power_monitor = GPUPowerMonitor(gpu_ids=gpu_ids, sample_interval=args.power_sample_interval)
        power_monitor.start()
    
    try:
        # Warmup
        warmup_throughput = run_warmup(engine)
        
        # 准备 prompts
        prompts, estimated_tokens_per_prompt = prepare_prompts(args, warmup_throughput)
        
        # 准备采样参数
        sampling_params = SamplingParams(
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
            max_tokens=args.max_tokens
        )
        
        # 运行 PEARL benchmark
        output_text, num_tokens, num_acc_tokens, elapsed_time, pearl_throughput = run_pearl_benchmark(
            engine, prompts, sampling_params, verbose=args.verbose
        )
        
        # 运行 AR benchmark（如果启用）
        ar_throughput = None
        if args.run_ar_benchmark:
            _, _, _, ar_throughput = run_ar_benchmark(
                engine, prompts, sampling_params, verbose=args.verbose
            )
            if ar_throughput > 0:
                speedup = pearl_throughput / ar_throughput
                logger.info(f"PEARL Speedup: {speedup:.2f}x")
        
        # 导出功耗数据
        if power_monitor:
            power_monitor.stop()
            power_monitor.print_summary()
            if args.export_power_csv:
                power_monitor.export_to_csv(args.export_power_csv)
    
    finally:
        # 清理资源
        if power_monitor:
            power_monitor.stop()
        engine.exit()

