"""
一次性完成 KNN-MMD WiFall 训练集类均衡 + 验证
对标论文: "randomly drop 34 of the samples from the 'fall' class"
仅修改 People 1-9 (源域), Person 0 (目标域) 保持不变
"""
import numpy as np
import sys, os

DATA_DIR = "./data/fall"
BACKUP_DIR = "./data/fall_before_rebalance"

# ── Step 1: 加载数据 ──────────────────────────────────
mag  = np.load(os.path.join(DATA_DIR, "magnitude_linear.npy"))
pha  = np.load(os.path.join(DATA_DIR, "phase_linear.npy"))
act  = np.load(os.path.join(DATA_DIR, "action.npy"))
ppl  = np.load(os.path.join(DATA_DIR, "people.npy"))

print("=" * 60)
print("Step 1: 当前数据全貌")
print("=" * 60)
print(f"总样本: {len(act)}  magnitude: {mag.shape}  phase: {pha.shape}")
print(f"Person 分布: {np.bincount(ppl)}")
print(f"Action 分布: {np.bincount(act)}")

# ── Step 2: 分析训练集(People 1-9)各类样本数 ──────────
train_mask = ppl != 0    # Person 1-9 = 源域
test_mask  = ppl == 0    # Person 0   = 目标域

train_act = act[train_mask]
class_ids = np.unique(train_act)
class_counts = {c: (train_act == c).sum() for c in class_ids}

print(f"\nStep 2: 训练集(People 1-9)各类分布")
print("-" * 40)
for c in sorted(class_ids):
    print(f"  Class {c}: {class_counts[c]} 样本")
fall_class = max(class_counts, key=class_counts.get)
print(f"  → fall 类 = Class {fall_class} ({class_counts[fall_class]} 样本，最多)")

# ── Step 3: 类均衡 ────────────────────────────────────
# 论文: 每类 108, 从 fall 随机丢弃 34
# 我们的数据: 取非 fall 类的最小样本数作为目标, 所有类统一到此数量
non_fall_min = min(v for k, v in class_counts.items() if k != fall_class)
target_per_class = non_fall_min
print(f"\nStep 3: 类均衡 (目标: {target_per_class}/类)")

# 构建训练集均衡索引
balanced_train_indices = []
for c in sorted(class_ids):
    c_mask = train_mask.copy()
    # 找出属于 People 1-9 且 action==c 的全局索引
    c_global_indices = np.where((ppl != 0) & (act == c))[0]
    n_available = len(c_global_indices)
    n_select = min(n_available, target_per_class)
    selected = np.random.choice(c_global_indices, size=n_select, replace=False)
    balanced_train_indices.append(selected)
    print(f"  Class {c}: {n_available} → {n_select}")

balanced_train_indices = np.sort(np.concatenate(balanced_train_indices))
# 目标域 (Person 0) 保持不动
test_indices = np.where(test_mask)[0]
new_indices = np.sort(np.concatenate([balanced_train_indices, test_indices]))

print(f"\n训练集: {len(balanced_train_indices)} 样本 (均衡后)")
print(f"目标域: {len(test_indices)} 样本 (未改动)")
print(f"总计:   {len(new_indices)} 样本")

# ── Step 4: 备份旧数据 + 写新 npy ─────────────────────
os.makedirs(BACKUP_DIR, exist_ok=True)
for f in ["magnitude_linear.npy", "phase_linear.npy", "action.npy", "people.npy"]:
    src = os.path.join(DATA_DIR, f)
    dst = os.path.join(BACKUP_DIR, f)
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)

np.save(os.path.join(DATA_DIR, "magnitude_linear.npy"), mag[new_indices])
np.save(os.path.join(DATA_DIR, "phase_linear.npy"),     pha[new_indices])
np.save(os.path.join(DATA_DIR, "action.npy"),            act[new_indices])
np.save(os.path.join(DATA_DIR, "people.npy"),            ppl[new_indices])
print(f"\nStep 4: 旧数据已备份到 {BACKUP_DIR}/")
print(f"新数据已写入 {DATA_DIR}/")

# ── Step 5: 验证 ──────────────────────────────────────
print(f"\n{'=' * 60}")
print("Step 5: 验证")

mag2 = np.load(os.path.join(DATA_DIR, "magnitude_linear.npy"))
act2 = np.load(os.path.join(DATA_DIR, "action.npy"))
ppl2 = np.load(os.path.join(DATA_DIR, "people.npy"))

print(f"新数据总量: {len(act2)}")
print(f"Person 0 样本数: {(ppl2==0).sum()} (应不变)")
print(f"Person 0 各类: {np.bincount(act2[ppl2==0])}")
print(f"\n训练集(People 1-9)各类分布:")
balanced_act = act2[ppl2 != 0]
for c in np.unique(balanced_act):
    print(f"  Class {c}: {(balanced_act==c).sum()} 样本")
print(f"\n各类差: {max(np.bincount(balanced_act)) - min(np.bincount(balanced_act))} (应为0)")
assert (ppl2 == 0).sum() == (ppl == 0).sum(), "ERROR: Person 0 样本数变了!"
print("\n✓ 验证通过 — 训练集类均衡完成, 目标域未改动")
