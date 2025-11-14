"""
Prompt Loader Utility
从 JSONL 文件中加载和采样 prompts
"""
import json
import random
import os
from nano_pearl.utils.pearl_logger import logger


def load_prompts_from_jsonl(jsonl_files, max_prompts=None, random_seed=None):
    """
    从 JSONL 文件中加载 prompts
    
    Args:
        jsonl_files: JSONL 文件路径列表
        max_prompts: 最大加载的 prompts 数量，如果为 None 则加载所有
        random_seed: 随机种子，用于随机采样 prompts。如果为 None 则按顺序加载
    
    Returns:
        prompts 列表
    """
    prompts = []
    
    for jsonl_file in jsonl_files:
        if not os.path.exists(jsonl_file):
            logger.warning(f"文件不存在，跳过: {jsonl_file}")
            continue
            
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 从 conversations 中提取第一条 human 的消息
                    if 'conversations' in data and len(data['conversations']) > 0:
                        for conv in data['conversations']:
                            if conv.get('from') == 'human':
                                prompt = conv.get('value', '').strip()
                                if prompt:
                                    prompts.append(prompt)
                                break  # 只取第一条 human 消息
                except json.JSONDecodeError as e:
                    logger.warning(f"解析 JSON 失败，跳过此行: {e}")
                    continue
    
    total_loaded = len(prompts)
    
    # 如果指定了随机种子，进行随机采样
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(prompts)
        logger.info(f"使用随机种子 {random_seed} 对 prompts 进行打乱")
    
    # 如果指定了最大数量，进行截取
    if max_prompts is not None and max_prompts > 0:
        prompts = prompts[:max_prompts]
        logger.info(f"从 {total_loaded} 条 prompts 中采样了 {len(prompts)} 条")
    else:
        logger.info(f"从 {len(jsonl_files)} 个 JSONL 文件中加载了 {len(prompts)} 条 prompts")
    
    return prompts


def get_default_prompts(max_prompts=None, random_seed=None):
    """
    从默认的 JSONL 文件中加载 prompts
    
    Args:
        max_prompts: 最大加载的 prompts 数量，如果为 None 则加载所有
        random_seed: 随机种子，用于随机采样 prompts。如果为 None 则按顺序加载
    
    Returns:
        prompts 列表
    """
    # 从 nano_pearl/utils/prompt_loader.py 找到项目根目录
    # 文件路径: project_root/nano_pearl/utils/prompt_loader.py
    # 需要找到: project_root/static/
    current_file = os.path.abspath(__file__)
    # 向上三级: prompt_loader.py -> utils -> nano_pearl -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    # 定义三个 JSONL 文件路径
    jsonl_files = [
        os.path.join(project_root, 'static', 'sharegpt_gpt4.jsonl'),
        os.path.join(project_root, 'static', 'sharegpt_V3_format.jsonl'),
        os.path.join(project_root, 'static', 'sharegpt_zh_38K_format.jsonl')
    ]
    
    return load_prompts_from_jsonl(jsonl_files, max_prompts=max_prompts, random_seed=random_seed)


def estimate_prompts_needed(throughput, target_duration, tokens_per_prompt):
    """
    根据吞吐量和目标时长估算需要的 prompts 数量
    
    Args:
        throughput: 吞吐量 (tokens/s)
        target_duration: 目标时长 (秒)
        tokens_per_prompt: 每个 prompt 平均的 token 数
    
    Returns:
        估算的 prompts 数量
    """
    if throughput <= 0 or tokens_per_prompt <= 0:
        return 0
    
    estimated = int(throughput * target_duration / tokens_per_prompt)
    return max(1, estimated)  # 至少返回 1

