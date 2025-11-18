#!/bin/bash
# Benchmark Runner Script for nano-PEARL
# 使用方法: bash run_benchmark.sh

# ==================== 配置区域 ====================
# 请修改以下参数以适配你的环境

# 模型路径（必填）
DRAFT_MODEL="/home/llm/model_hub/Qwen3-0.6B"      # 例如: "/data/models/llama-68m"
TARGET_MODEL="/home/llm/model_hub/Qwen3-8B"    # 例如: "/data/models/llama-7b"

# 并行度设置
DRAFT_TP=2          # Draft 模型的 tensor parallel size
TARGET_TP=6         # Target 模型的 tensor parallel size

# GPU 内存利用率
GPU_MEMORY_UTIL=0.9

# 生成参数
TEMPERATURE=0.0     # 采样温度，0.0 表示贪婪采样
MAX_TOKENS=200      # 最大生成 token 数
IGNORE_EOS=false    # 是否忽略 EOS token

# 实验控制参数
MAX_PROMPTS=100              # 最大 prompts 数量，0 表示加载所有
TARGET_DURATION=300          # 目标实验时长（秒），默认 300 秒（5分钟）
REQUEST_RATE=1.0             # 请求发送速率（QPS）
BATCH_SIZE=10                # 批次大小

# 功耗监控（可选）
MONITOR_POWER=true           # 是否监控 GPU 功耗
POWER_SAMPLE_INTERVAL=0.1    # 功耗采样间隔（秒）
EXPORT_POWER_CSV="power_log.csv"  # 功耗数据导出文件，留空则不导出

# 其他选项
RUN_AR_BENCHMARK=false       # 是否运行 AR baseline 对比实验
VERBOSE=false                # 是否输出详细日志
SEED=0                       # 随机种子

# ==================== 脚本主体 ====================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}nano-PEARL Benchmark Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查必需参数
if [[ "$DRAFT_MODEL" == "/path/to/draft/model" ]] || [[ "$TARGET_MODEL" == "/path/to/target/model" ]]; then
    echo -e "${RED}错误: 请先在脚本中配置 DRAFT_MODEL 和 TARGET_MODEL 路径！${NC}"
    echo "编辑此脚本，修改第 6-7 行的模型路径"
    exit 1
fi

# 检查模型路径是否存在
if [[ ! -d "$DRAFT_MODEL" ]]; then
    echo -e "${RED}错误: Draft 模型路径不存在: $DRAFT_MODEL${NC}"
    exit 1
fi

if [[ ! -d "$TARGET_MODEL" ]]; then
    echo -e "${RED}错误: Target 模型路径不存在: $TARGET_MODEL${NC}"
    exit 1
fi

# 构建命令
CMD="python bench.py \
    --draft-model \"$DRAFT_MODEL\" \
    --target-model \"$TARGET_MODEL\" \
    --draft-tp $DRAFT_TP \
    --target-tp $TARGET_TP \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --max-prompts $MAX_PROMPTS \
    --target-duration $TARGET_DURATION \
    --request-rate $REQUEST_RATE \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --power-sample-interval $POWER_SAMPLE_INTERVAL"

# 添加布尔标志
if [[ "$IGNORE_EOS" == "true" ]]; then
    CMD="$CMD --ignore-eos"
fi

if [[ "$MONITOR_POWER" == "true" ]]; then
    CMD="$CMD --monitor-power"
fi

if [[ "$RUN_AR_BENCHMARK" == "true" ]]; then
    CMD="$CMD --run-ar-benchmark"
fi

if [[ "$VERBOSE" == "true" ]]; then
    CMD="$CMD --verbose"
fi

if [[ -n "$EXPORT_POWER_CSV" ]]; then
    CMD="$CMD --export-power-csv \"$EXPORT_POWER_CSV\""
fi

# 显示配置
echo -e "${YELLOW}实验配置:${NC}"
echo "  Draft Model:      $DRAFT_MODEL"
echo "  Target Model:     $TARGET_MODEL"
echo "  Draft TP:         $DRAFT_TP"
echo "  Target TP:        $TARGET_TP"
echo "  GPU Memory:       ${GPU_MEMORY_UTIL}"
echo "  Max Tokens:       $MAX_TOKENS"
echo "  Max Prompts:      $MAX_PROMPTS"
echo "  Target Duration:  ${TARGET_DURATION}s"
echo "  Request Rate:     ${REQUEST_RATE} QPS"
echo "  Batch Size:       $BATCH_SIZE"
echo "  Monitor Power:    $MONITOR_POWER"
echo "  Run AR Baseline:  $RUN_AR_BENCHMARK"
echo ""

# 确认执行
echo -e "${YELLOW}即将执行的命令:${NC}"
echo "$CMD"
echo ""
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 执行命令
echo -e "${GREEN}开始运行 benchmark...${NC}"
echo ""
eval $CMD

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Benchmark 完成!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    if [[ -n "$EXPORT_POWER_CSV" ]] && [[ -f "$EXPORT_POWER_CSV" ]]; then
        echo "功耗数据已保存到: $EXPORT_POWER_CSV"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Benchmark 执行失败!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

