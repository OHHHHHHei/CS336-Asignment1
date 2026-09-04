#!/usr/bin/env bash

set -u

cd "/home/leejt/Language Modeling from Scratch"

RUN_ROOT="/data/leejt/cs336_assignment1/runs"
LOG_ROOT="${RUN_ROOT}/remaining_experiment_logs"
mkdir -p "${LOG_ROOT}"

run_experiment() {
    local run_name="$1"
    shift
    local log_path="${LOG_ROOT}/${run_name}.log"

    echo
    echo "===== 开始 ${run_name} $(date '+%Y-%m-%d %H:%M:%S %z') ====="
    echo "===== 命令：CUDA_VISIBLE_DEVICES=6 uv run python -u cs336_basics/run_train_tinystories.py $* --run-name ${run_name} --run-root ${RUN_ROOT} ====="

    CUDA_VISIBLE_DEVICES=6 uv run python -u cs336_basics/run_train_tinystories.py "$@" \
        --run-name "${run_name}" \
        --run-root "${RUN_ROOT}" 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}

    if [[ ${status} -eq 0 ]]; then
        echo "===== 完成 ${run_name} $(date '+%Y-%m-%d %H:%M:%S %z') ====="
    else
        echo "===== 失败 ${run_name} status=${status} $(date '+%Y-%m-%d %H:%M:%S %z') ====="
    fi
}

# 按教案补充归一化实验。
run_experiment tinystories_no_rmsnorm_lr1e-4 \
    --experiment no_rmsnorm \
    --max-learning-rate 1e-4 \
    --min-learning-rate 1e-5

# Peak learning-rate sweep。最后一个运行使用更激进的设置，尝试覆盖稳定性分析所需的发散区间。
run_experiment tinystories_lr1e-4 \
    --experiment baseline \
    --max-learning-rate 1e-4 \
    --min-learning-rate 1e-5
run_experiment tinystories_lr2e-4 \
    --experiment baseline \
    --max-learning-rate 2e-4 \
    --min-learning-rate 2e-5
run_experiment tinystories_lr3e-4 \
    --experiment baseline \
    --max-learning-rate 3e-4 \
    --min-learning-rate 3e-5
run_experiment tinystories_lr6e-4 \
    --experiment baseline \
    --max-learning-rate 6e-4 \
    --min-learning-rate 6e-5
run_experiment tinystories_lr1e-3 \
    --experiment baseline \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-4
run_experiment tinystories_lr3e-3 \
    --experiment baseline \
    --max-learning-rate 3e-3 \
    --min-learning-rate 3e-4

# Batch-size sweep 从单样本开始，直到 24 GB RTX 3090 上找到的最大安全 batch。
# 所有运行使用相同数量的 optimizer steps。
run_experiment tinystories_batch1 \
    --experiment baseline \
    --batch-size 1
run_experiment tinystories_batch32 \
    --experiment baseline \
    --batch-size 32
run_experiment tinystories_batch64 \
    --experiment baseline \
    --batch-size 64
run_experiment tinystories_batch128 \
    --experiment baseline \
    --batch-size 128
run_experiment tinystories_batch192 \
    --experiment baseline \
    --batch-size 192

echo
echo "===== 队列完成 $(date '+%Y-%m-%d %H:%M:%S %z') ====="
