#!/bin/bash
# ============================================================
#  1-shot × 3 连续训练脚本
# ============================================================
#  用途：用相同超参连续跑 3 次 1-shot 训练，验证统计稳定性
#  产物：每次训练自动归档到 experiments/history/<run_name>/
#  汇总：跑完打印 3 次的 best test acc + Mean/Std/Min/Max
#
#  用法：
#    bash run_1shot_x3.sh
#
#  后台跑（关掉终端不影响）：
#    nohup bash run_1shot_x3.sh > run_1shot_x3.log 2>&1 &
#    tail -f run_1shot_x3.log    # 看进度
# ============================================================

set -e
cd "$(dirname "$0")"

# ---- 配置 ----
CONDA_PY="/home/sanguin1us/miniconda3/envs/myenv/bin/python"
TASK="action"
TEST_LIST=0
K=1
N=1
P=0.5
D=32
LR=0.0005
BATCH_SIZE=256
NUM_RUNS=3

# 记录脚本启动时间（用于过滤"本次新跑的" run）
START_TS=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

echo "================================================================"
echo "  1-shot × ${NUM_RUNS} 连续训练"
echo "  开始时间: ${START_HUMAN}"
echo "  超参: task=${TASK}, test_list=${TEST_LIST}"
echo "        k=${K}, n=${N}, p=${P}, d=${D}, lr=${LR}, batch_size=${BATCH_SIZE}"
echo "================================================================"

OVERALL_START=$(date +%s)

for i in $(seq 1 ${NUM_RUNS}); do
    RUN_START=$(date +%s)
    echo ""
    echo "▶▶▶ Run ${i}/${NUM_RUNS} 开始  $(date '+%H:%M:%S')"
    echo ""

    "${CONDA_PY}" train_fall.py \
        --task ${TASK} \
        --test_list ${TEST_LIST} \
        --k ${K} \
        --n ${N} \
        --p ${P} \
        --d ${D} \
        --mode 1 \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --cuda 0

    RUN_END=$(date +%s)
    RUN_DUR=$((RUN_END - RUN_START))
    echo ""
    echo "▶▶▶ Run ${i}/${NUM_RUNS} 完成  耗时 $((RUN_DUR / 60))分$((RUN_DUR % 60))秒"
done

OVERALL_END=$(date +%s)
OVERALL_DUR=$((OVERALL_END - OVERALL_START))

echo ""
echo "================================================================"
echo "  全部 ${NUM_RUNS} 次训练完成"
echo "  总耗时: $((OVERALL_DUR / 60))分$((OVERALL_DUR % 60))秒"
echo "================================================================"

# 汇总统计
"${CONDA_PY}" <<PYEOF
import json, glob, os, statistics
from datetime import datetime

START_TS = ${START_TS}

# 过滤匹配的 1-shot run
runs_new = []
runs_all_legit = []  # 历史所有相同参数的合法 run（排除 NaN 崩溃和 rebalanced 那两次）

for d in sorted(glob.glob('experiments/history/*')):
    meta_path = os.path.join(d, 'meta.json')
    if not os.path.exists(meta_path):
        continue
    with open(meta_path) as f:
        m = json.load(f)
    args = m.get('args', {})
    if args.get('k') != ${K} or args.get('n') != ${N}:
        continue
    if args.get('test_list') != [${TEST_LIST}]:
        continue
    if args.get('task') != '${TASK}':
        continue
    if args.get('lr') != ${LR}:
        continue
    # 排除 rebalanced 那次
    if 'data_note' in args:
        continue

    ts_str = m.get('timestamp', '')
    try:
        ts = int(datetime.fromisoformat(ts_str).timestamp())
    except Exception:
        ts = 0
    rec = (ts, m['run_name'], m['best_test_acc'], m['total_epochs'])
    runs_all_legit.append(rec)
    if ts >= START_TS:
        runs_new.append(rec)

def print_table(runs, title):
    print(f"\n=== {title} ===")
    if not runs:
        print("  (空)")
        return
    print(f"{'Run name':<55} {'best_test_acc':>15} {'epochs':>8}")
    print("-" * 85)
    for _, name, acc, ep in sorted(runs):
        print(f"{name:<55} {acc:>15.4f} {ep:>8}")
    if len(runs) >= 2:
        accs = [r[2] for r in runs]
        print()
        print(f"  Mean:  {statistics.mean(accs):.4f}")
        print(f"  Std:   {statistics.stdev(accs):.4f}")
        print(f"  Min:   {min(accs):.4f}")
        print(f"  Max:   {max(accs):.4f}")
        print(f"  Range: [{min(accs):.4f}, {max(accs):.4f}]  (跨度 {max(accs)-min(accs):.4f})")

print_table(runs_new, f"本次新跑的 {len(runs_new)} 次 1-shot")
print_table(runs_all_legit, f"累计合法 1-shot Person0 共 {len(runs_all_legit)} 次")

# 看 best/ 现在是哪个
if os.path.exists('experiments/best/meta.json'):
    with open('experiments/best/meta.json') as f:
        bm = json.load(f)
    print(f"\n=== 当前 experiments/best/ ===")
    print(f"  run:           {bm['run_name']}")
    print(f"  best_test_acc: {bm['best_test_acc']:.4f}")
PYEOF

echo ""
echo "================================================================"
echo "  打开 TensorBoard 对比："
echo "    tensorboard --logdir ./runs --port 6006"
echo "================================================================"
