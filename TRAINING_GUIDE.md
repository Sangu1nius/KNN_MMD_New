# KNN-MMD 训练操作指南

> 基于本目录三份 README 的整理 + 源码细节核对，给出从「数据预处理 → 模型训练 → 结果评估」的一站式操作手册。
>
> 三份 README 对应内容：
> - [README.md](README.md) ── 项目主入口，说明如何运行 `train.py` / `train_fall.py`
> - [WiFall/README.md](WiFall/README.md) ── WiFall 数据集格式与采集配置
> - [WiFall/data_process_example/README.md](WiFall/data_process_example/README.md) ── CSV → numpy 预处理流程

---

## 0. 总览：5 种训练任务

代码支持两个数据集 × 两类标签 = 共 5 种典型训练任务：

| # | 数据集 | 任务 | 入口脚本 | `--task` | 类别数 | 训练目的 |
|---|---|---|---|---|---|---|
| ① | WiGesture | **手势识别**（跨人） | `train.py` | `action` | 6 | 训练能在新用户身上识别手势的模型 |
| ② | WiGesture | **人员识别**（跨动作） | `train.py` | `people` | 8 | 训练能在新动作下识别用户身份的模型 |
| ③ | WiFall | **动作识别**（跨人） | `train_fall.py` | `action` | 5 | 训练能在新用户身上识别动作的模型 |
| ④ | WiFall | **人员识别**（跨动作） | `train_fall.py` | `people` | 10 | 训练能在新动作下识别用户身份的模型 |
| ⑤ | WiFall | **跌倒检测**（跨人，二分类） | `train_fall.py`（需改 class_num=2 或数据预处理时把非 fall 合并）| `action` | 2 | 训练跌倒报警模型，把 fall 视作一类、其余 4 类合为一类 |

> ⚠️ 任务 ⑤ 在当前代码里**没有直接接口**——`train_fall.py` 默认是 5 类动作识别。要做"跌倒 vs 非跌倒"二分类，需要在预处理阶段把 `action.npy` 里 0/1/2/3/4 重映射为 0(fall)/1(other)，并把代码里的 `class_num=5` 改成 `class_num=2`。论文 Table III 提到这点，但代码没有自带这个开关。

---

## 1. 训练前准备

### 1.1 激活 Python 环境

```bash
# 进入项目目录
cd "/home/sanguin1us/ScientificResearch/Documents/WIFI_CSI_HAR/KNN_MMD/codes/KNN-MMD-main(1)/KNN-MMD-main"

# 激活 conda 环境
source /home/sanguin1us/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# 验证
python -c "import torch, umap, sklearn; print('ok')"
```

### 1.2 GPU / CPU 选择

**硬件**：RTX 2060（支持 CUDA）。

**当前状态**：`nvidia-smi` 报 "Driver/library version mismatch"（NVML 库 535.309 vs 内核驱动 535.288.01），这是**驱动升级后没重启**导致的常见问题。修复办法：

```bash
# 推荐：直接重启
sudo reboot

# 重启后验证
nvidia-smi                         # 应该正常显示 RTX 2060
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

**正确训练命令**：使用 GPU（重启后）：
```bash
python train.py --cuda 0 ...       # 指定使用 GPU 0
```

如果暂时无法重启，临时用 CPU（速度慢约 20-30 倍）：
```bash
python train.py --cpu ...
```

### 1.3 数据预处理（关键步骤，严格按顺序执行）

代码加载的 4 个 npy 文件：

```
data/
├── magnitude_linear.npy   # shape: (N, 100, 52)  幅度
├── phase_linear.npy       # shape: (N, 100, 52)  相位
├── action.npy             # shape: (N,)          动作 ID
└── people.npy             # shape: (N,)          人物 ID
```

但 [WiFall/data_process_example/](WiFall/data_process_example/) 里的脚本只产生 `magnitude.npy` / `phase.npy` 等**没带 `_linear` 后缀**的版本——`_linear` 后缀来自 [CSI-BERT](https://github.com/RS2002/CSI-BERT) 的丢包恢复后处理。如果不想跑 CSI-BERT，**最简单的处理是手动改名或加一步线性插值**（见下方"快速方案"）。

#### 1.3.1 准备 WiFall 数据（必须按 STEP 1 → STEP 2 → STEP 3 顺序）

[WiFall/data_process_example/README.md](WiFall/data_process_example/README.md) 明确要求**先 `process1.py`，再 `process2*.py`**——前者把 CSV 解析成 pkl，后者基于 pkl 切样本。**跳过 process1.py 不能直接跑 process2**，因为 process2 系列脚本第一行就是 `with open("./csi_data.pkl", 'rb')` 读 process1 的输出。

```bash
cd "/home/sanguin1us/ScientificResearch/Documents/WIFI_CSI_HAR/KNN_MMD/codes/KNN-MMD-main(1)/KNN-MMD-main"
cd WiFall/data_process_example/

# ──────────────────────────────────────────────────────
# STEP 1 (必跑且最先跑): CSV → pkl
# ──────────────────────────────────────────────────────
# process1.py 里硬编码 root="./data"，所以必须先建一个软链接
# 让 ./data 指向真正的 CSV 根目录 ../WiFall (即 ID0~ID9 所在目录)
ln -s ../WiFall ./data
ls ./data    # 应看到 ID0 ID1 ... ID9

# 运行：会遍历所有 IDx/{fall,Jump,sit,stand,walk}/*.csv
# 提取 CSI 复数信号 → magnitude/phase → 存进 list of dict
python process1.py
# 产物：./csi_data.pkl    （几十~几百 MB，看数据规模）

# ──────────────────────────────────────────────────────
# STEP 2 (必跑): pkl → npy （切成 100 帧定长样本）
# ──────────────────────────────────────────────────────
# 论文用的就是 process2-split.py（按时间窗 1 秒/100 帧切片）
python process2-split.py
# 产物：./magnitude.npy, phase.npy, action.npy, people.npy, timestamp.npy
# 形状: (N, 100, 52), (N, 100, 52), (N,), (N,), (N, 100)

# ──────────────────────────────────────────────────────
# STEP 3 (必跑): 丢包线性插值 + 改名 + 放到 data/fall/
# ──────────────────────────────────────────────────────
# 见 §1.3.3 的脚本（推荐）或 §1.3.2 的快速搬运方案（不推荐）
```

> **三个 process2 脚本怎么选**？
> - `process2-split.py` ⭐ **训练用这个**（论文用法）：按时间窗切样本，丢包帧填 -1000
> - `process2.py`：保留长序列、不切片——做序列模型时才用
> - `process2-squeeze-split.py`：切样本但**直接跳过丢包帧**——会改变时间结构，不推荐

#### 1.3.2 创建 `data/fall` 目录（train_fall.py 默认路径）

```bash
# 回到项目根目录
cd "/home/sanguin1us/ScientificResearch/Documents/WIFI_CSI_HAR/KNN_MMD/codes/KNN-MMD-main(1)/KNN-MMD-main"

mkdir -p data/fall

# 快速方案：把没带 _linear 后缀的 npy 改名搬过去（粗糙但能跑）
# 注：跳过 CSI-BERT 丢包恢复，丢包位置仍是 -1000 哨兵值，
# 模型会在含 -1000 的样本上看到异常输入，精度会比论文低
cp WiFall/data_process_example/magnitude.npy  data/fall/magnitude_linear.npy
cp WiFall/data_process_example/phase.npy      data/fall/phase_linear.npy
cp WiFall/data_process_example/action.npy     data/fall/action.npy
cp WiFall/data_process_example/people.npy     data/fall/people.npy
```

#### 1.3.3 严谨方案（推荐）：在改名前做线性插值

把 `-1000` 占位的丢包位置改为相邻有效值的线性插值。简单脚本：

```bash
# 在 KNN-MMD-main 目录运行
python <<'EOF'
import numpy as np
mag = np.load('WiFall/data_process_example/magnitude.npy')   # (N, 100, 52)
pha = np.load('WiFall/data_process_example/phase.npy')

def linear_fill(arr):
    """逐 (sample, subcarrier) 沿时间维做线性插值，把 -1000 当 NaN 处理"""
    arr = arr.astype(np.float32).copy()
    arr[arr == -1000] = np.nan
    N, T, S = arr.shape
    for n in range(N):
        for s in range(S):
            col = arr[n, :, s]
            nan_mask = np.isnan(col)
            if nan_mask.all():
                col[:] = 0.0
            elif nan_mask.any():
                idx = np.arange(T)
                col[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], col[~nan_mask])
            arr[n, :, s] = col
    return arr

mag = linear_fill(mag)
pha = linear_fill(pha)

import os
os.makedirs('data/fall', exist_ok=True)
np.save('data/fall/magnitude_linear.npy', mag)
np.save('data/fall/phase_linear.npy', pha)

# 标签直接复制
np.save('data/fall/action.npy', np.load('WiFall/data_process_example/action.npy'))
np.save('data/fall/people.npy', np.load('WiFall/data_process_example/people.npy'))
print('done; saved to data/fall/')
print('magnitude shape:', mag.shape)
EOF
```

#### 1.3.4 WiGesture 数据（如果你要跑任务 ①②）

`train.py` 默认 `--data_path ./data`。WiGesture 不在本仓库，需要：
- 从 [paperswithcode WiGesture](https://paperswithcode.com/dataset/wigesture) 或 [Hugging Face WiFall](https://huggingface.co/datasets/RS2002/WiFall) 下载
- 按同样格式预处理得到 `magnitude_linear.npy / phase_linear.npy / action.npy / people.npy`
- 放到 `./data/`

> **若没拿到 WiGesture**：直接跳过任务 ①②，先跑 ③④。

---

## 2. 训练任务详细指令

### 2.1 公共参数速查

| 参数 | 默认 | 含义 | 建议值 |
|---|---|---|---|
| `--task` | `action` | 任务类型 | `action` 或 `people` |
| `--test_list` | `0` | 目标域 ID（要预测的对象/动作）| 多次跑 0~9 做交叉验证 |
| `--k` | `5` | **n-shot 数**（每类支撑集大小）| 1（极少样本）或 5（多样本）|
| `--n` | `0` | KNN 邻居数（0 自动 = k//2+1）| 论文设 1 |
| `--p` | `0.5` | Help Set 比例（取置信度 top p%）| 0.5（论文最优）|
| `--d` | `128` | UMAP 输出维度 | 128（论文设）|
| `--mode` | `1` | top-p% 选择策略 | 1（论文用，全局取 top p%）|
| `--lr` | `0.0005` | 学习率 | action=5e-4, people=1e-3 |
| `--batch_size` | `256` | 批大小 | 256（CPU 训练可降到 32~64）|
| `--epoch` | `30` | 早停 patience | 30（默认即可）|
| `--cpu` / `--cuda 0` | 默认尝试 cuda:0 | CPU 还是 GPU | 重启修复驱动后用 `--cuda 0` |
| `--data_path` | `./data` (train) / `./data/fall` (train_fall) | npy 目录 | 默认即可 |
| `--norm` | True | 是否做样本级 z-score | 默认开（**别加 --norm，加了反而关闭**）|

> ⚠️ **`--norm` 坑**：源码用 `action="store_false", default=True`，意思是"加上 `--norm` 反而关闭归一化"。**保持默认（不加这个参数）就是开归一化**。

### 2.2 任务 ③：WiFall 动作识别（推荐先跑这个）

**训练目的**：让模型学会识别 fall/Jump/sit/stand/walk 5 种动作，且能泛化到训练时**没见过的人**（ID=0）。

**1-shot 设定**（最难，论文主测）：
```bash
cd "/home/sanguin1us/ScientificResearch/Documents/WIFI_CSI_HAR/KNN_MMD/codes/KNN-MMD-main(1)/KNN-MMD-main"

python train_fall.py \
    --task action \
    --test_list 0 \
    --k 1 \
    --n 1 \
    --p 0.5 \
    --d 128 \
    --mode 1 \
    --lr 0.001 \
    --batch_size 256 \
    --cuda 0
```
**预期结果（论文 Table V）**：测试集精度约 **75.3%**。

**5-shot 设定**（每类 5 个标签样本，更容易）：
```bash
python train_fall.py \
    --task action \
    --test_list 0 \
    --k 5 \
    --n 3 \
    --p 0.5 \
    --lr 0.001 \
    --batch_size 256 \
    --cuda 0
```
**预期**：精度比 1-shot 高 5~10 个点。

**结果保存**：
- 日志 → `./action.txt`（追加模式，每个 epoch 一行）
- 权重 → `./action.pth`（最佳 ResNet）+ `./action_cls.pth`（最佳分类头）

### 2.3 任务 ④：WiFall 人员识别

**训练目的**：让模型学会识别 10 个用户身份，且能泛化到训练时**没见过的动作**（ID=0，即 "fall" 动作）。

**1-shot**：
```bash
python train_fall.py \
    --task people \
    --test_list 0 \
    --k 1 \
    --n 1 \
    --p 0.5 \
    --lr 0.001 \
    --cuda 0
```

**5-shot**：
```bash
python train_fall.py \
    --task people \
    --test_list 0 \
    --k 5 \
    --n 3 \
    --p 0.5 \
    --lr 0.001 \
    --cuda 0
```

> 注意：跑 `--task people` 时 `--test_list 0` 表示**目标域是动作 ID=0**（不是人 ID=0）。
```

**结果保存**：日志 `./people.txt`，权重 `./people.pth` + `./people_cls.pth`。

### 2.4 任务 ①：WiGesture 手势识别（需先备好 WiGesture 数据）

**训练目的**：跨人员的 6 类手势识别。

**1-shot**（论文最佳 93.26%）：
```bash
python train.py \
    --task action \
    --test_list 0 \
    --k 1 \
    --n 1 \
    --p 0.5 \
    --lr 0.0005 \
    --cuda 0
```

**5-shot**：
```bash
python train.py \
    --task action \
    --test_list 0 \
    --k 5 \
    --n 3 \
    --p 0.5 \
    --lr 0.0005 \
    --cuda 0
```

### 2.5 任务 ②：WiGesture 人员识别

**训练目的**：跨动作的 8 类用户身份识别。

```bash
python train.py \
    --task people \
    --test_list 0 \
    --k 1 \
    --n 1 \
    --p 0.5 \
    --lr 0.001 \
    --cuda 0
```

### 2.6 任务 ⑤：WiFall 跌倒检测（二分类，需改代码）

**训练目的**：实战场景的跌倒报警——只关心"是否摔倒"，不区分摔倒姿态。

**改动方案**：
```bash
# 1) 重新生成二值标签
python <<'EOF'
import numpy as np
act = np.load('data/fall/action.npy')
# 假设原 action.npy 里 fall=0 (或 fall 对应的 id), 其余=other
# 先查看一下:
print('原 action ID:', np.unique(act))
print('每类样本数:', np.bincount(act))
# 论文里 fall 被采集了 48 次/人，其它每个 12 次/人，所以样本数最多的应该是 fall
fall_id = np.argmax(np.bincount(act))
print('fall_id =', fall_id)
# 把 fall 标为 0，其余标为 1
new_act = (act != fall_id).astype(np.int64)
np.save('data/fall/action_binary.npy', new_act)
print('done; new label dist:', np.bincount(new_act))
EOF

# 2) 复制一份 train_fall.py 改成二分类版本
cp train_fall.py train_fall_binary.py
sed -i 's/class_num=5/class_num=2/g; s/class_num = 5/class_num = 2/g' train_fall_binary.py
# 改数据加载里的 action 文件名
sed -i 's/action.npy/action_binary.npy/g' dataset.py    # 临时改，跑完记得改回
# 注意: 改 dataset.py 是粗糙做法，更干净的做法是给 load_zero_shot 加一个参数

# 3) 训练
python train_fall_binary.py \
    --task action \
    --test_list 0 \
    --k 1 \
    --n 1 \
    --p 0.5 \
    --lr 0.001 \
    --cuda 0
```

**预期结果（论文 Table V）**：1-shot 约 **77.62%**。

---

## 3. 训练监控与调试

### 3.1 监控日志输出

训练时每个 epoch 控制台会输出三行：
```
Epoch 1 | Train Loss 1.234567,  Train Acc 0.876543 |
Valid Loss 1.234567, Valid Acc 0.876543 |
Test Loss 1.234567, Test Acc 0.876543
Acc Epoch 5, Loss Epoch 3, Same Epoch 0
```

**关键指标解读**：
- **Train Acc** ↑：源域上的拟合情况。一般几个 epoch 就接近 100%。
- **Valid Acc**：**支撑集精度**（目标域，少量标签）。这是早停看的指标。
- **Test Acc**：**真正测试集精度**（目标域剩余样本）。这是最终关心的指标。
- **Acc Epoch / Loss Epoch**：连续多少个 epoch 没改善。任意一个 > `--epoch` 即触发早停（且 `j > 200`）。

### 3.2 早停规则（参考 [knowledge.md §2.6](knowledge.md)）

代码里早停逻辑（[train.py:422-429](train.py#L422-L429)）：
1. `j ≤ 200`：**永不早停**，强制至少 200 epoch（这是 `e_min`）。
2. `j == 200`：触发"松弛"，把 `best_acc *= 0.8, best_loss *= 1.2`，给后续训练空间。
3. `j > 200`：若 `acc_epoch ≥ 30 AND loss_epoch ≥ 30`（或 `same_epoch ≥ 30`） → 早停。
4. `j > 350`：强制停止。

**想快速试跑**（不等 200 epoch）：临时改 [train.py:422](train.py#L422) 的 `j>200` 为 `j>20`、[train.py:424](train.py#L424) 的 `j==200` 为 `j==20`。

### 3.3 后台运行 + 日志重定向

CPU 训练较慢（200~350 epoch × 每 epoch 几分钟），建议后台跑：

```bash
# 后台跑 + 日志记录
nohup python train_fall.py --task action --test_list 0 --k 1 --p 0.5 --lr 0.001 --cpu \
    > logs_fall_action_id0_k1.log 2>&1 &

# 查看 PID
echo $!

# 实时看日志
tail -f logs_fall_action_id0_k1.log

# 查看进度（解析 .txt 日志）
tail -5 action.txt
```

### 3.4 调小 batch_size 节约内存

CPU 训练时 256 的 batch 可能很慢/占内存。可降到 32 或 64：
```bash
python train_fall.py ... --batch_size 32 ...
```
**注意**：batch 太小会影响 MMD 估计的稳定性（MMD 需要足够样本估计分布），建议不低于 32。

---

## 4. 系统性实验：跨域留一交叉验证

论文 Table V 的 "average" 精度是把目标域 ID 从 0 到 N-1 全部各跑一遍，取平均。完整跨域评估流程：

### 4.1 WiFall 动作识别全员交叉

```bash
# 用 shell 循环跑 10 次（ID 0~9 各做一次目标域）
for tid in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== Target Person ID = $tid ==="
    python train_fall.py \
        --task action \
        --test_list $tid \
        --k 1 --n 1 --p 0.5 \
        --lr 0.001 --cpu \
        > logs_fall_action_id${tid}.log 2>&1
    # 取这次的最佳测试精度
    grep "Test Acc" action.txt | tail -1
    # 把 action.txt 改名保存，避免下次被追加污染
    mv action.txt action_id${tid}.txt
done
```

### 4.2 汇总精度

```bash
python <<'EOF'
import re
import numpy as np
accs = []
for tid in range(10):
    with open(f'action_id{tid}.txt') as f:
        text = f.read()
    test_accs = [float(m.group(1)) for m in re.finditer(r'Test Acc (\d+\.\d+)', text)]
    if test_accs:
        best = max(test_accs)
        print(f'ID {tid}: best test acc = {best:.4f}')
        accs.append(best)
print(f'Average: {np.mean(accs):.4f}')
EOF
```

---

## 5. 超参数调优建议

按论文 Sensitivity Analysis 与代码默认值整理：

| 超参 | 调优方向 | 备注 |
|---|---|---|
| `--k` (n-shot) | 1 → 5 → 10 | 增大 k 通常涨点，但 5 之后边际效益递减 |
| `--n` (KNN 邻居数) | 1 ~ k | k=1 时只能 n=1；k≥3 时建议 n=1~3 |
| `--p` (Help Set 比例) | 0.3 ~ 0.7 | 论文 sensitivity 显示 0.5 附近最稳，太小样本不够、太大噪声多 |
| `--d` (UMAP 维度) | 32, 64, 128 | 论文表 VI 显示 d 对 KNN-MMD 不敏感，默认 128 即可 |
| `--mode` | 0, 1, 2 | 默认 1（全局取 top p%）。类别不平衡严重时可试 0 或 2 |
| `--lr` | 5e-4 ~ 1e-3 | action 用 5e-4，people 用 1e-3（论文经验）|
| `mmd_weight` | 1, 2, 5 | 在 [train.py:17](train.py#L17) 硬编码=2。loss 收不下来可适当调小 |

---

## 6. 训练结果分析与可视化

### 6.1 画训练曲线
```bash
python <<'EOF'
import re
import matplotlib.pyplot as plt
with open('action.txt') as f:
    text = f.read()
train_acc = [float(m.group(1)) for m in re.finditer(r'Train Acc (\d+\.\d+)', text)]
valid_acc = [float(m.group(1)) for m in re.finditer(r'Valid Acc (\d+\.\d+)', text)]
test_acc  = [float(m.group(1)) for m in re.finditer(r'Test Acc (\d+\.\d+)', text)]
plt.figure(figsize=(10,5))
plt.plot(train_acc, label='Train')
plt.plot(valid_acc, label='Valid (support set)')
plt.plot(test_acc,  label='Test')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid()
plt.savefig('training_curve.png', dpi=100)
print('saved training_curve.png')
EOF
```

### 6.2 加载训练好的模型推理
```bash
python <<'EOF'
import torch
from model import Resnet, Linear
from dataset import load_zero_shot
from torch.utils.data import DataLoader

# 加载权重
model = Resnet(output_dims=64, channel=1, pretrained=False, norm=True)
classifier = Linear(input_dims=64, output_dims=5)
model.load_state_dict(torch.load('action.pth', map_location='cpu'))
classifier.load_state_dict(torch.load('action_cls.pth', map_location='cpu'))
model.eval(); classifier.eval()

# 加载目标域测试集
_, test_data = load_zero_shot(test_people_list=[0], data_path='./data/fall', task='action')
loader = DataLoader(test_data, batch_size=64, shuffle=False)

correct, total = 0, 0
with torch.no_grad():
    for x, y in loader:
        pred = classifier(model(x)).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f'Test Accuracy: {correct/total:.4f}')
EOF
```

---

## 7. 常见问题排查

### Q1: `ModuleNotFoundError: No module named 'umap'`
A: 进错环境了。用全路径 `/home/sanguin1us/miniconda3/envs/myenv/bin/python` 或先 `conda activate myenv`。

### Q2: `FileNotFoundError: ./data/fall/magnitude_linear.npy`
A: 数据预处理没跑或路径不对。回到 §1.3 重新生成。

### Q3: `RuntimeError: CUDA error: forward compatibility ...` 或 `Failed to initialize NVML: Driver/library version mismatch`
A: NVIDIA 用户空间库（NVML 535.309）与内核驱动模块（535.288.01）版本不匹配，通常是系统升级了 nvidia 包但**还没重启**。修复方法：
```bash
sudo reboot                         # 推荐：重启后内核加载新驱动
# 重启后验证
nvidia-smi                          # 应正常显示 RTX 2060
python -c "import torch; print(torch.cuda.is_available())"  # 应为 True
```
临时备选：训练命令加 `--cpu` 用 CPU 跑。

### Q4: 训练第一个 epoch 极慢
A: 检查实际用的设备。看打印里有没有 `cuda:0`，或在脚本里临时加：
```python
print('Device:', device)
```
- 若打印 `cuda:0` 还慢 → 可能是 batch_size 太大或 IO 瓶颈
- 若打印 `cpu` → GPU 没启用，回去查 Q3

### Q5: Test Acc 一直在 20% 左右（随机猜测水平）
A: 检查：
- 数据是否预处理完整、有无大量 -1000 哨兵值（看 [§1.3.3 严谨方案](§1.3.3)）
- `--test_list` 的 ID 是否真的存在于数据里（`np.unique(np.load('data/fall/people.npy'))`）
- 是否误加了 `--norm` 关闭了归一化

### Q6: `RuntimeError: Found dtype Long but expected Float` 之类的张量类型错误
A: 通常是 npy 文件 dtype 不对。`magnitude_linear.npy` 必须是 `float32`，`action.npy/people.npy` 必须是 `int64`。预处理时已经 `.astype()` 过，正常不会出错。

### Q7: 想换个目标域 ID 跑
A: 改 `--test_list <id>` 即可。注意每次跑会**追加**写到 `action.txt`，需要先备份/删除旧的：
```bash
mv action.txt action_id0.txt
python train_fall.py ... --test_list 1 ...
```

### Q8: 想加速调试，不想等 200 epoch 才早停
A: 临时编辑 [train.py:422-424](train.py#L422) / [train_fall.py:420-422](train_fall.py#L420)：
```python
if (((acc_epoch >= args.epoch and loss_epoch >= args.epoch) or same_epoch >= args.epoch) and j>20) or j>50:
    break
if j==20:
    ...
```
把 `200` 改 `20`、`350` 改 `50` 即可快速 smoke test。**正式实验记得改回**。

---

## 8. 推荐训练流程（首次跑）

如果你刚拿到这个项目，建议按下面顺序走：

```bash
# === 阶段 0: 修复 GPU 驱动（如果 nvidia-smi 报版本不匹配）===
sudo reboot
# 重启后：
nvidia-smi                                    # 确认 RTX 2060 正常显示

# === 阶段 1: 环境就绪 ===
conda activate myenv
cd "/home/sanguin1us/ScientificResearch/Documents/WIFI_CSI_HAR/KNN_MMD/codes/KNN-MMD-main(1)/KNN-MMD-main"
python -c "import torch; print('cuda:', torch.cuda.is_available())"  # 应为 True

# === 阶段 2: 数据预处理（严格按顺序！）===
cd WiFall/data_process_example/
ln -s ../WiFall ./data            # 让 process1.py 能找到 CSV 根目录
python process1.py                # ⭐ STEP 1: CSV → csi_data.pkl
python process2-split.py          # ⭐ STEP 2: pkl → magnitude.npy 等
cd ../..

# ⭐ STEP 3: 用 §1.3.3 的脚本做线性插值并搬到 data/fall/
# 生成 data/fall/{magnitude_linear,phase_linear,action,people}.npy

# === 阶段 3: Smoke Test (临时改小 e_min 后跑) ===
python train_fall.py --task action --test_list 0 --k 1 --p 0.5 --lr 0.001 --cuda 0 --batch_size 32

# === 阶段 4: 正式训练（恢复原 e_min=200）===
# 单 ID 跑通后再做留一交叉验证
for tid in 0 1 2 3 4 5 6 7 8 9; do
    python train_fall.py --task action --test_list $tid \
        --k 1 --p 0.5 --lr 0.001 --cuda 0 > logs_id${tid}.log 2>&1
    mv action.txt action_id${tid}.txt
done

# === 阶段 5: 汇总 =
# (用 §4.2 的脚本汇总平均精度)
```

---

## 9. 三份 README 内容速查

### [README.md](README.md)（项目主入口）
**核心信息**：
- 论文链接（TMC 2025 + ICCC 2025）
- 自建数据集 WiFall 已上传 Hugging Face
- 运行命令模板：`python train.py --k <shot> --n <KNN-k> --p <top-p%> --task <action|people> --lr <lr>`
- WiGesture → 用 `train.py`，WiFall → 用 `train_fall.py`

### [WiFall/README.md](WiFall/README.md)（WiFall 数据集说明）
**关键参数**：
- 采集设备：ESP32-S3
- 频段：2.4 GHz，20 MHz 带宽
- 子载波数：**52**
- 采样率：**100 Hz**
- 协议：802.11n OFDM
- 数据组织：`ID0~ID9/{fall,Jump,sit,stand,walk}/*.csv`
- CSI 数据列：`data` 列含 104 个数，配对成 52 个复数子载波
  - 复数 = `data[2i] + data[2i+1]·j`
  - 实际代码 [process1.py:14-16](WiFall/data_process_example/process1.py#L14-L16) 顺序略不同，按代码为准

### [WiFall/data_process_example/README.md](WiFall/data_process_example/README.md)（预处理流程）
**两步法**：
1. **process1.py**：CSV → pkl（提取 magnitude / phase）
2. **process2 系列**（三选一）：
   - `process2.py`：保留完整长序列（丢包用 -1000 填）
   - `process2-split.py` ⭐ **训练用**：切成 100 帧固定长度样本
   - `process2-squeeze-split.py`：切固定长度但丢包帧直接跳过
- 建议结合 **CSI-BERT** 做丢包恢复，生成 `_linear` 版本（代码里加载的是 `magnitude_linear.npy`）

---

## 10. 训练时间估算

WiFall 数据集 ~5000 样本/epoch 的参考时长：

| 硬件 | batch | 每 epoch | 完整训练 (200~350 epoch) | 10-fold 留一交叉 |
|---|---|---|---|---|
| **RTX 2060 (GPU)** ⭐ | 256 | ~5-10 秒 | ~30-60 分钟 | ~5-10 小时 |
| RTX 2060 (GPU) | 32 | ~3-6 秒 | ~20-40 分钟 | ~3-6 小时 |
| CPU | 256 | ~2-5 分钟 | ~10-15 小时 | 5-7 天 |
| CPU | 32 | ~30-60 秒 | ~3-5 小时 | 1-2 天 |

**建议**：重启电脑修好 CUDA 驱动后再开始大规模训练。CPU 跑只适合做 smoke test。
