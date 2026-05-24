# KNN-MMD 项目知识手册

> **本文档用途**：完整解释 `KNN-MMD-main/` 目录下的所有代码在做什么、为什么这么做，以便任何切换进来的模型/协作者能在不重新读论文的情况下立即理解工程全貌。
>
> **基于**：论文《KNN-MMD: Cross Domain Wireless Sensing Via Local Distribution Alignment》(IEEE TMC 2025, Zhao et al.) 与本目录下源码的逐行对照。

---

## 0. TL;DR（30 秒速通）

- **任务**：基于 WiFi CSI 信号做**人体活动识别（HAR）**——但训练数据（源域）和测试数据（目标域）来自**不同人**或**不同动作**，存在严重 **domain shift**。
- **设定**：**Few-Shot Cross-Domain**——目标域每类只有 n 个（1~5）标签样本（"支撑集"），其余无标签。
- **方法**：双阶段流程——
  1. **Step 1 伪标签生成**：把目标域无标签样本经 UMAP 降维后送进 KNN（参考支撑集），生成伪标签 + 置信度，只取 top-p% 高置信度样本做"**帮助集（Help Set）**"。
  2. **Step 2 局部分布对齐训练**：ResNet18+MLP，损失 = 交叉熵 + **类内 MK-MMD**（核心创新）+ 全局 MK-MMD，用支撑集做早停验证。
- **核心创新**：传统 MMD 做"全局对齐"会把不同类样本拉混，本文按类别做"**局部对齐（Local Distribution Alignment）**"——这是论文唯一最重要的 idea。
- **数据集**：
  - **WiGesture**（公开，6 手势 × 8 人）→ 用 [train.py](train.py)
  - **WiFall**（作者自建，5 动作 × 10 人）→ 用 [train_fall.py](train_fall.py)
- **代码量**：~5 个核心 Python 文件，总共不到 800 行。模型结构其实非常简单。

---

## 1. 项目背景与问题定义

### 1.1 WiFi CSI 是什么
- **CSI (Channel State Information，信道状态信息)**：WiFi 信号经过环境传播后，在接收端测得的"信道指纹"。物理上等价于公式 `Y = HX + N` 中的信道矩阵 `H`。
- 复数表示 `H = |H| · e^(j∠H)`，可拆为**幅度（magnitude）**与**相位（phase）**。
- 论文设备：**ESP32-S3**，2.4 GHz，20 MHz 带宽 → **52 个有效子载波**，约 100 Hz 采样。
- 一个样本 = 100 帧 × 52 子载波 → 形如 `(1, 100, 52)` 的 2D "图像"喂给卷积网络。

### 1.2 跨域问题
- 人体活动会扰动 WiFi 信号多径传播 → 不同动作产生不同 CSI 模式 → ML 模型可识别。
- 但 CSI **极度敏感**于"动作以外的因素"：人体身材、家具位置、设备型号、房间几何……
- 结果：A 房间训出的模型在 B 房间精度暴跌。论文 Fig.2-3 用 UMAP 可视化证实了这一点。
- 这就是 **domain shift / cross-domain problem**——源域 `P_s(x) ≠ P_t(x)`，且常常 `P_s(y|x) ≠ P_t(y|x)`。

### 1.3 Few-Shot 设定
- 目标域不完全是 zero-shot：每类有 **n 个**有标签样本，组成 **support set（支撑集）**。
- 论文测了 **1-shot 和 5-shot** 两种。
- 这种 "少量目标域标签 + 大量目标域无标签"的设定是本文的**主战场**。

### 1.4 论文相比传统 DAL 的三点贡献
1. **局部（类内）分布对齐**——而非全局对齐，避免跨类混淆。
2. **支撑集早停**——支撑集**不参与**训练损失，因而可作"目标域代理验证集"判定何时收敛。
3. **多任务统一框架**——手势识别、人员识别、跌倒检测、动作识别四类任务统一处理。

---

## 2. 方法论核心（论文公式与算法）

### 2.1 整体架构图（来自论文 Fig.5）

```
            ┌──────────────  Target Domain ───────────────┐
            │  Testing Set (无标签)        Support Set (少标签)
            │                ↓                 ↓
            │           ┌─────────────┐
            │           │ UMAP 降维   │
            │           └─────┬───────┘
            │                 ↓
            │           ┌─────────────┐
            │           │  KNN 分类    │ ← 用 support 当参考
            │           └─────┬───────┘
            │                 ↓
            │       Pseudo-Labels + Confidence
            │                 ↓
            │       取 Top-p% → Help Set (伪标签)         │
            └──────────────────┬───────────────────────────┘
                               │
─────────  Step 1: Preliminary Classification 完  ─────────
                               │
            ┌──────  Source Domain ──────┐  + Help Set
            │ Training Set (大量标签)     │
            │            ↓
            │     BatchNorm → ResNet18 → 64维嵌入 → MLP → 类别
            │            │
            │            ↓ 三项损失同时优化
            │   L_cls (训练集)
            │ + α₁ · L_MMD^global  (训练集 ↔ 帮助集，全部样本)
            │ + α₂ · L_MMD^local   (训练集 ↔ 帮助集，按类别条件)
            │
            │     用 Support Set 监控 → 自适应早停
            └─────────────────────────────────────────────────┘
```

### 2.2 数学依据：为什么需要"局部对齐"

源域→目标域的条件分布关系（贝叶斯展开，论文公式 6）：

```
P_s(y|g(x)) = P_t(y|g(x)) · [P_s(x)/P_t(x)] · [P_s(y)/P_t(y)]
```

- 传统 DAL 只做 `P_s(g(x)) ≈ P_t(g(x))`（**边际**对齐）。
- 这只保证 `P_s(x) ≈ P_t(x)`，**不**保证条件分布对齐。
- KNN-MMD 额外要求 `P_t(g(x)|y) ≈ P_s(g(x)|y)`——即**每个类别内**对齐——从而推出 `P_s(y|g(x)) ≈ P_t(y|g(x))`，分类边界才能真正迁移过去。

### 2.3 MMD / MK-MMD

**MMD 定义（公式 7）**：找 RKHS 中函数 f 使两分布期望差最大。
```
MMD[F, p, q] = sup_{f ∈ F} | E_p[f(x)] − E_q[f(x)] |
```

**核 MMD 可计算形式（公式 8）**：
```
MMD²(p, q) = (1/n²) Σ k(xᵢᵖ, xⱼᵖ)       源域内部
           + (1/m²) Σ k(xᵢᵠ, xⱼᵠ)       目标域内部
           − (2/nm) Σ k(xᵢᵖ, xⱼᵠ)       跨域
```

**MK-MMD（公式 9）**：用 H 个核加权求和，避免单核选择困难。论文实测发现"主要靠一个 Gaussian 核 σ=0.5 就够了"。

### 2.4 Algorithm 1：Help Set 构造（论文 P9）

```
输入：(X_support, y_support)、X_test、k、p%
1. for each x_i in X_test:
2.   N ← KNN(x_i, X_support, k)              # 在支撑集中找 k 个最近邻
3.   ŷ_i ← majorityVote(N 的标签)             # 伪标签 = 邻居众数
4.   x* ← N 中与 x_i 同类且距离最小的样本
5.   d_i ← dist(x_i, x*)
6.   c_i ← 1 / (d_i + ε)                     # 置信度 = 距离倒数
7. 按 c_i 降序排
8. 取前 n = ⌊p% × |X_test|⌋ 个 → Help Set
```

### 2.5 损失函数（公式 11）

```
L = L_cls + α₁ · L_MMD^global + α₂ · L_MMD^local
```

- **L_cls** = 交叉熵（仅用源域训练集）。
- **L_MMD^global** = MK-MMD(训练集嵌入, 帮助集嵌入)，不分类别（兜底，处理类不平衡）。
- **L_MMD^local** = (1/M) Σ_{i=1}^M MK-MMD(训练集嵌入_i, 帮助集嵌入_i)，**按类别条件**。

> ⚠️ **论文写的权重范围与代码不一致**：论文 Table IV 给的 `[e_min, e_max, e_threshold, α, β] = [200, 350, 30, 1.2, 0.8]`，但这里的 α、β 是早停的"松弛因子"，不是损失权重 α₁、α₂。代码里实际只有一个标量 `mmd_weight=2`（见 [train.py:17](train.py#L17)），同时乘到全局和局部 MMD 上。

### 2.6 Algorithm 2：自适应早停（论文 P10）

- 每个 epoch 在**支撑集**上算 loss 和 acc（支撑集**不参与**训练，所以可当验证集）。
- 维护 best_loss / best_acc 与各自的"连续未改善 epoch 计数"。
- 到达 `e_min` 之前不允许早停；之后若 loss 和 acc 都连续 `e_threshold` 个 epoch 未改善 → 停。
- 到达 `e_min` 时做一次"松弛"：`best_loss *= α (α≥1)`、`best_acc *= β (β≤1)`，重置计数。

---

## 3. 代码结构总览

```
KNN-MMD-main/
├── README.md                      # 项目说明（论文链接 + 运行示例）
├── model.py                       # ⭐ 模型定义（ResNet + MLP）
├── dataset.py                     # ⭐ 数据加载、跨域划分
├── func.py                        # ⭐ MMD/核函数/距离计算工具
├── train.py                       # ⭐ WiGesture 训练入口（6 手势 / 8 人）
├── train_fall.py                  # ⭐ WiFall 训练入口（5 动作 / 10 人）
├── img/                           # 论文示意图
│   ├── model.png, network.png, layout.png, sketch.png
├── WiFall/                        # 自建数据集与预处理
│   ├── README.md                  # 数据集说明
│   ├── WiFall.zip                 # 压缩包
│   ├── WiFall/                    # 解压后的原始 CSV（在 .gitignore）
│   │   └── ID0~ID9/{fall,Jump,sit,stand,walk}/*.csv
│   └── data_process_example/      # CSV → numpy 预处理脚本
│       ├── process1.py            # CSV → pkl（提取 magnitude/phase）
│       ├── process2.py            # pkl → 长序列 npy（保留丢包）
│       ├── process2-split.py      # pkl → 固定长度 npy（按时间切）
│       └── process2-squeeze-split.py  # pkl → 固定长度 npy（去丢包）
└── knowledge.md                   # ← 本文档
```

**主要 import 依赖**：`torch`, `torchvision`, `numpy`, `tqdm`, `umap-learn`, `scikit-learn`, `scipy`, `pandas`。

---

## 4. 逐文件详解

### 4.1 [model.py](model.py) — 网络结构

对应论文公式 (10): `E = ResNet(BatchNorm(X)); Ŷ = MLP(E)`。

```python
class Resnet(nn.Module):
    def __init__(self, output_dims=64, channel=2, pretrained=True, norm=False):
        self.model = models.resnet18(pretrained)
        # 改写第一层 conv 接受任意通道数（默认 2 = magnitude+phase；
        # 但 load_zero_shot 只用 magnitude 单通道 → 实际 channel=1）
        self.model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 输出维度改为 output_dims（默认 64，论文 Table IV 也是 64）
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dims)
        self.batch_norm = nn.BatchNorm2d(1)
        self.norm = norm
    def forward(self, x):
        if self.norm:                          # 沿子载波维度做样本级归一化
            mean = torch.mean(x, dim=-2, keepdim=True)
            std  = torch.std(x, dim=-2, keepdim=True)
            y = (x - mean) / (std + 1e-8)
        else:
            y = x
        y = self.batch_norm(y)                 # 再过一层 BatchNorm2d
        return self.model(y)                   # ResNet18 → 64 维嵌入

class Linear(nn.Module):
    def __init__(self, input_dims=64, output_dims=6):
        self.model = nn.Sequential(
            nn.Linear(input_dims, 64), nn.ReLU(),
            nn.Linear(64, 64),         nn.ReLU(),
            nn.Linear(64, output_dims),        # 输出 = 类别数 (action=6/5, people=8/10)
        )
```

**几个易忽略的细节**：
- ImageNet 预训练 ResNet18 接受 3 通道，这里**改写 conv1 为 channel=1**，**预训练权重在 conv1 自动丢弃**（其余层保留）。
- `BatchNorm2d(1)` 期望输入是 `(N, 1, H, W)`，所以输入张量必须是 `(B, 1, 100, 52)`。
- 论文说 64 维嵌入接 3 层 MLP——和这里完全一致。
- `output_dims=64`（嵌入维度）≠ `hidden_dim=64`（MLP 中间层），命名容易混淆但其实数值相同。

### 4.2 [dataset.py](dataset.py) — 数据加载与跨域划分

**核心类 `CSI_dataset`**：标准 PyTorch `Dataset` 包装，支持 `task="action"` 或 `task="people"` 切换标签。

**三个加载函数**：

1. **`load_data(...)`**：加载 magnitude + phase **2 通道**数据，按训练/验证/测试比例顺序划分。**train.py 不使用此函数。**

2. **`load_zero_people(test_people_list, ...)`**：按人 ID 划分——`test_people_list` 中的人作为目标域。**train.py 也不使用。**

3. **`load_zero_shot(test_people_list, test_action_list, ...)`** ⭐ **训练实际使用的函数**：
   - 注意 [dataset.py:94-95](dataset.py#L94-L95)：**只加载 magnitude，单通道**`np.expand_dims(magnitude, axis=1)`。Phase 被注释掉了。
   - `test_people_list`：这些 ID 的人作为目标域，其余作为源域。
   - `test_action_list`：这些 ID 的动作作为目标域，其余作为源域。
   - 二者同时给可以走 `func="and"/"or"` 组合（一般只给一个）。
   - 返回 `(源域 Dataset, 目标域 Dataset)`。

> ⚠️ **关键陷阱**：`load_zero_shot` **没有打乱**——按 npy 文件顺序划分，所以 npy 必须按 (person, action) 分组连续存放，否则划分会乱。`process2-split.py` 默认就是这样输出的。

### 4.3 [func.py](func.py) — MMD / 核函数工具

- **`compute_kernel(x, y, kernel_type, kernel_param)`** ([func.py:8-31](func.py#L8-L31))：
  - `kernel_type='gaussian'`：高斯核 `exp(-||x-y||²/(2σ²))`，`kernel_param=σ`。
  - `kernel_type='linear'`：内积 `x·yᵀ`。

- **`mk_mmd_loss(x, y, kernel_types, kernel_params)`** ([func.py:33-66](func.py#L33-L66)) ⭐ 训练损失中调用：
  - 默认 5 个高斯核 σ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}（但训练代码里实际只传 `[0.5, 1.0]` 两个，见 [train.py:256](train.py#L256)）。
  - 计算 `MMD = mean(K_xx) - 2·mean(K_xy) + mean(K_yy)`，开根号后跨核取平均。
  - **注意**：`torch.sqrt(torch.max(0, mmd))`——夹紧到非负，避免负值开根号 NaN。

- **`mk_dis(x, y, ...)`** ([func.py:68-98](func.py#L68-L98))：基于核的样本间距离，专给 `KernelKMeans` 用。Gaussian 核对角为 1，所以 `x²=y²=1`。

- **`KernelKMeans`** ([func.py:100-145](func.py#L100-L145))：核 K-means 聚类。**当前训练流程不使用**——可能是早期实验残留或扩展接口。

- **`find_min_distance(A, B)`** ([func.py:152-166](func.py#L152-L166))：用 `scipy.optimize.linear_sum_assignment` 做匈牙利匹配。**当前训练流程不使用**。

### 4.4 [train.py](train.py) — WiGesture 训练入口

**全局开关**（[train.py:15-17](train.py#L15-L17)）：
```python
support_mmd = False    # 是否额外算"训练集 vs 真实支撑集"的 MMD（默认关）
global_mmd  = True     # 是否算"训练集 vs 测试集全部"的全局 MMD（默认开）
mmd_weight  = 2        # MMD 损失权重（action=1, people=2，注释说的，代码默认 2）
```

#### 4.4.1 命令行参数（[train.py:19-39](train.py#L19-L39)）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--batch_size` | 256 | 同论文 |
| `--hidden_dim` | 64 | 嵌入维度（论文也是 64）|
| `--data_path` | `./data` | npy 数据目录 |
| `--lr` | 0.0005 | action 用 5e-4，people 用 1e-3（注释说的）|
| `--test_list` | `[0]` | **作为目标域的人/动作 ID 列表** |
| `--epoch` | 30 | 早停 patience（连续 30 epoch 无改善才停）|
| `--task` | "action" | "action" 或 "people" |
| `--k` | 5 | **每类支撑集大小**（n-shot 的 n）|
| `--n` | 0 | KNN 邻居数（0 → 自动设为 k//2+1）|
| `--p` | 0.5 | **Help Set 比例**（top p%） |
| `--d` | 128 | UMAP 输出维度（论文 Table IV 也是 128）|
| `--mode` | 1 | top-p 选择策略，见 4.4.4 |
| `--norm` | True | 是否做样本级标准化 |

#### 4.4.2 `k_shot(...)` ([train.py:41-74](train.py#L41-L74))

从目标域的 `embedding`（UMAP 降维后）里**每类取前 k 个**作为支撑集模板（`template`），其余作为待分类的测试集。返回模板+剩余样本。

注意：是按"出现顺序"取前 k 个，**不随机**。

#### 4.4.3 `dimension_reducation(...)` ([train.py:76-92](train.py#L76-L92))

把目标域 DataLoader 的全部样本展平 (1×100×52=5200 维)，用 `umap.UMAP(n_components=d)` 降到 `d=128` 维。返回 `(原始数据, UMAP嵌入, 标签)`。**这是 Step 1 的"前置降维"**。

#### 4.4.4 `top_knn(...)` ([train.py:94-189](train.py#L94-L189)) ⭐ 对应 Algorithm 1

1. 调 `k_shot` 切出支撑集模板。
2. `sklearn.neighbors.KNeighborsClassifier(n_neighbors=n)` 在模板上拟合。
3. 对所有剩余样本预测 KNN 邻居，用 `scipy.stats.mode` 投票得伪标签 `y_pred`。
4. 计算每个样本到"同类最近邻"的距离 `y_dis`——**距离越小置信度越高**（代码里直接用距离，越小越好）。
5. 按 `mode` 选 top-p% 高置信度样本：
   - `mode=0`：每类取固定数 `m = total*p%/class_num`（保证类均衡）。
   - `mode=1` **默认/论文使用**：全局直接取 top p%（**可能类不平衡**，所以才需要 global_mmd 兜底）。
   - `mode=2`：每类取本类内的 top-p%。
6. 返回 `(template, template_label, x_support, y_support, origin_data, label)`。
   - `template` = 真实支撑集模板（k 个/类）
   - `x_support` = top-p% 帮助集（伪标签）
   - `origin_data` = 剩余测试样本（用真实标签做最终评估）

> 注意命名混乱：变量 `x_support` 在论文里其实是 **Help Set**，而 `template` 才是 **Support Set**。**别被名字误导！**

#### 4.4.5 `iteration(...)` ([train.py:198-321](train.py#L198-L321)) ⭐ 训练核心

参数：`(model, classifier, optim, train_loader, test_loader, support_loader, global_loader, ...)`

- 训练阶段 `train=True`：遍历 `train_loader`，每个 batch：
  1. 计算源域分类损失 `L_cls`。
  2. **类内 MMD**（[train.py:239-257](train.py#L239-L257)）：取一个 `test_loader` batch（即"我的帮助集"`my_support_loader`），对每个类别 i 计算 `mk_mmd_loss(源域类i嵌入, 帮助集类i嵌入)`，求平均后 ×`mmd_weight` 加入 loss。这就是论文公式 (12) 的 `L_MMD^local`。
  3. （可选）**真实支撑集类内 MMD**（[train.py:260-279](train.py#L260-L279)）：`support_mmd=True` 时启用，类似上面但用真实支撑集模板。
  4. **全局 MMD**（[train.py:282-294](train.py#L282-L294)）：`global_mmd=True` 时启用，从 `global_loader`（指向真正测试集）取 batch，整体算一次 MMD，对应公式 (13) 的 `L_MMD^global`。
  5. 反向传播 + 梯度裁剪 `clip_grad_norm_(..., 3.0)`。

- 验证/测试阶段 `train=False`：只前向，关梯度，遍历 `test_loader`（可以传入支撑集或测试集），返回 loss/acc。

#### 4.4.6 `main(...)` ([train.py:323-433](train.py#L323-L433)) — 编排逻辑

```python
# 1. 加载源/目标域
if task == "action":
    train_data, _ = load_zero_shot(test_people_list=args.test_list+['2'], ...)
    _, test_data  = load_zero_shot(test_people_list=args.test_list, ...)
elif task == "people":
    train_data, _ = load_zero_shot(test_action_list=args.test_list+['1'], ...)
    _, test_data  = load_zero_shot(test_action_list=args.test_list, ...)
```

> ⚠️ **细节陷阱**：`args.test_list+['2']` 这里 `test_list` 是 int 列表，加了字符串 `'2'`——但 `dataset.py` 里比较 `people[i] not in test_people_list`，`people[i]` 是 int64，与字符串 `'2'` 永不相等，所以 `'2'` 这个额外项**实际是 no-op**。猜测是历史遗留或调试残留。当前等价于 `test_people_list=args.test_list`。

```python
# 2. Step 1: UMAP 降维 + KNN 伪标签 + Top-p% 帮助集
origin_data, embedding, data_label = dimension_reducation(test_loader, ...)
template, template_label, x_support, y_support, origin_data, label = top_knn(...)
x_support = torch.cat([x_support, template], dim=0)  # 帮助集 ⊕ 模板（充实样本）
y_support = torch.cat([y_support, template_label], dim=0)

# 3. 包装三个 DataLoader
support_loader    = DataLoader(template)             # 真实支撑集 (k×classes 样本)
my_support_loader = DataLoader(x_support)            # 帮助集 + 真实模板
test_loader       = DataLoader(origin_data)          # 真正的目标域测试集

# 4. 初始化模型
model      = Resnet(output_dims=64, channel=1, pretrained=True, norm=True)
classifier = Linear(input_dims=64, output_dims=class_num)
optim      = Adam(lr=5e-4, weight_decay=0.01)

# 5. 训练主循环
while True:
    j += 1
    iteration(..., train=True)   # 训练一个 epoch
    iteration(..., train=False, test_loader=support_loader)  # 在支撑集上验证
    iteration(..., train=False, test_loader=test_loader)     # 在测试集上看效果

    # 自适应早停（对应 Algorithm 2）
    if acc 创新高 or loss 创新低: 保存 ckpt
    if acc/loss 连续 args.epoch 个 epoch 没改进 and j > 200: break
    if j == 200: 触发松弛——best_acc *= 0.8, best_loss *= 1.2
    if j > 350: break  # 强制停
```

### 4.5 [train_fall.py](train_fall.py) — WiFall 训练入口

**与 [train.py](train.py) 99% 相同**，差异仅在：
- `class_num`：action 任务 5 类（fall/Jump/sit/stand/walk）、people 任务 10 人。
- `--data_path` 默认 `./data/fall`。
- 默认 `--lr 0.001`。
- 不再在 `test_list` 后面加 `'2'`/`'1'`（即没有那个 no-op）。
- 其余流程、损失、早停完全一致。

### 4.6 [WiFall/](WiFall/) — 数据集与预处理

**目录组织**：`WiFall/WiFall/IDx/{fall,Jump,sit,stand,walk}/*.csv`，每个 ID 5 个动作子目录，每个动作下若干 CSV。

**CSV 列含义**（来自 [WiFall/README.md](WiFall/README.md)）：
- `seq`: 行号
- `timestamp` (UTC+8) / `local_timestamp` (ESP local time)
- `rssi`: RSSI 信号
- `data`: CSI 复数信号 = 104 个数 = 52 子载波 × (real, imag)
  - `complex_csi[i] = data[2i] + data[2i+1] · j`（实际代码里是 `data[2i+1] + data[2i]·j`，见 [process1.py:14-16](WiFall/data_process_example/process1.py#L14-L16) 注意顺序）
- 其他列：MAC、MCS 等设备信息

**预处理流程**（见 [WiFall/data_process_example/README.md](WiFall/data_process_example/README.md)）：

1. **`process1.py`**：CSV → pkl
   - 遍历 `./data/IDx/{action}/*.csv`，提取 CSI 复数序列。
   - 拆出 `magnitude = |csi|` 和 `phase = ∠csi (deg)`。
   - 保存到 `csi_data.pkl`，含 `volunteer_id, action_id, magnitude, phase, timestamps...`。

2. **`process2.py`**：pkl → 长序列 npy
   - 按时间戳差识别丢包，丢包处填 `-1000`（哨兵值）。
   - 输出 `data_sequence.pkl`。

3. **`process2-split.py`** ⭐ **训练实际使用**：pkl → 固定长度 npy
   - 以 `gap=1` 秒为窗口，每窗口取 `length=100` 帧。
   - 丢包填 `-1000`，不足 100 帧的窗口用最后帧外推填充。
   - 输出 `magnitude.npy / phase.npy / action.npy / people.npy / timestamp.npy`，形状 `(N, 100, 52)` / `(N,)`。

4. **`process2-squeeze-split.py`**：pkl → 固定长度 npy（去丢包）
   - 直接按 100 帧切，**不处理丢包**——丢包帧被压缩掉。
   - 输出到 `./squeeze_data/`。

> **提示**：训练前需准备的 npy 文件——`magnitude_linear.npy`, `phase_linear.npy`, `people.npy`, `action.npy`（[dataset.py:30-34](dataset.py#L30-L34)）。命名里的 `_linear` 后缀提示这是经过 CSI-BERT 之类丢包恢复后的线性插值版本（详见 [WiFall/data_process_example/README.md](WiFall/data_process_example/README.md)）。

---

## 5. 训练流程完整时序

```
┌────────────────────────────────────────────────────────────────────┐
│ A. 准备数据 (一次性)                                                  │
│   1. 解压 WiFall.zip → WiFall/WiFall/IDx/{action}/*.csv               │
│   2. 跑 process1.py → csi_data.pkl                                   │
│   3. 跑 process2-split.py → magnitude.npy 等 → 用 CSI-BERT 等         │
│      恢复丢包 → magnitude_linear.npy, phase_linear.npy               │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ B. 运行 train.py (或 train_fall.py)                                   │
│                                                                     │
│ ┌── main() ──────────────────────────────────────────────────────┐ │
│ │                                                                │ │
│ │ 1. load_zero_shot()  ── 按人/动作切源/目标域                     │ │
│ │ 2. DataLoader(train_data) ── 源域，batch_size=256               │ │
│ │ 3. dimension_reducation(test_loader) ── UMAP → 128 维           │ │
│ │ 4. top_knn() ── KNN 伪标签 + Top-p% Help Set                     │ │
│ │                                                                │ │
│ │ ── Step 1 完，得到三个集合：                                     │ │
│ │   - support_loader (k 个/类 真实支撑集，做早停验证)               │ │
│ │   - my_support_loader (Help Set + 支撑集，参与 MMD 对齐)          │ │
│ │   - test_loader (剩余目标域样本，最终评估)                        │ │
│ │                                                                │ │
│ │ 5. Resnet + Linear + Adam                                      │ │
│ │ 6. while True:                                                 │ │
│ │      iteration(train=True)                                     │ │
│ │      ├── 源域分类 L_cls                                          │ │
│ │      ├── 类内 MK-MMD(源域 batch, 帮助集 batch)                     │ │
│ │      ├── 全局 MK-MMD(源域 batch, 测试集 batch)                     │ │
│ │      └── backward + clip_grad                                  │ │
│ │      iteration(train=False, support_loader) → 支撑集验证          │ │
│ │      iteration(train=False, test_loader) → 测试评估              │ │
│ │      早停检查 + 模型保存                                          │ │
│ │                                                                │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## 6. 关键超参与论文 Table IV 对照

| 参数 | 代码默认 | 论文 Table IV | 作用 |
|---|---|---|---|
| CSI 长度 | 100 | 100 | 每样本帧数 |
| 子载波数 | 52 | 52 | OFDM 子载波 |
| batch_size | 256 | 256 | |
| 输入形状 | (256, 1, 100, 52) | (256, 1, 100, 52) | |
| UMAP dim | 128 | 128 | KNN 前的降维 |
| ResNet 隐藏维 | 64 | 64 | 嵌入维度 |
| KNN k | 0 → k//2+1 | 1 (one-shot) | 邻居数 |
| support k (shot 数) | 5 | n (1 或 5) | 每类支撑集大小 |
| Help Set 比例 p% | 0.5 | 50% | top-p% 伪标签筛选 |
| 优化器 | Adam | Adam | |
| 学习率 | 5e-4 | 5e-4 | |
| 早停 [e_min, e_max, e_threshold, α, β] | [200, 350, 30, 1.2, 0.8] | 同 | 软触发松弛 |
| MMD 核 | Gaussian σ∈{0.5, 1.0} | Gaussian σ∈{0.5, 1.0} | |
| 总参数 | ~11M | 11.21M | ResNet18+MLP |

---

## 7. 跨域评测协议

四种任务的源/目标域定义（对应论文 Table III）：

| 任务 | 数据集 | 源域 | 目标域 | 类数 |
|---|---|---|---|---|
| 手势识别 | WiGesture | People 1-7 | People 0 | 6 |
| 人员识别 | WiGesture | Action 1-5 | Action 0 | 8 |
| 跌倒检测 | WiFall | People 1-9 | People 0 | 2 (fall vs other) |
| 动作识别 | WiFall | People 1-9 | People 0 | 5 |

在命令行里通过 `--test_list 0` 指定目标域 ID 为 0，其余作源域。

**论文最佳结果（one-shot, Table V）**：
- 手势识别 **93.26%** | 人员识别 **81.84%** | 跌倒检测 **77.62%** | 动作识别 **75.30%** | 平均 **82.01%**
- 对比基线：
  - 上限（ResNet18 同域训练同域测试）：82.47% 平均
  - 下限（ResNet18 zero-shot 直接跨域）：49.30% 平均
  - 传统 MK-MMD：55.70% 平均

---

## 8. 怎么运行

### 8.1 WiGesture（手势识别 6 类，目标域 ID=0）
```bash
python train.py --k 5 --n 1 --p 0.5 --task action --lr 0.0005 --test_list 0
```

### 8.2 WiGesture（人员识别 8 类，目标域 Action=0）
```bash
python train.py --k 5 --n 1 --p 0.5 --task people --lr 0.001 --test_list 0
```

### 8.3 WiFall（动作识别 5 类，目标域 ID=0）
```bash
python train_fall.py --k 5 --n 1 --p 0.5 --task action --lr 0.001 --test_list 0
```

### 8.4 1-shot 设定
把 `--k 1` 即可。

### 8.5 输出
- 控制台 + `{task}.txt` 日志文件（每行：Epoch、Train Loss/Acc、Valid Loss/Acc、Test Loss/Acc）
- `{task}.pth`（模型权重）、`{task}_cls.pth`（分类头权重）

---

## 9. 易踩坑 / 常见误解

1. **`x_support` 在代码里是"帮助集（Help Set）"，而 `template/support_data` 才是"支撑集（Support Set）"**。变量命名与论文术语不对齐。

2. **代码只用 magnitude 单通道，不用 phase**。虽然 `model.py` 写了 `channel=2` 的接口，但 `load_zero_shot` 只 expand magnitude，实际 channel=1。

3. **`load_zero_shot` 没有 shuffle**，依赖 npy 按 (person, action) 连续排布。如果你自己重做 npy 时打乱了，划分会变成乱码。

4. **`test_list+['2']` / `['1']` 是 no-op**——int 列表加字符串元素后，与 int64 标签比较永不相等。可能是历史遗留代码。

5. **`mmd_weight=2` 是单一标量**——同时乘到全局 MMD 和局部 MMD 上，不是论文公式 (11) 的 α₁、α₂ 分开调。如果想分别调，需要修改 [train.py:257](train.py#L257) 和 [train.py:294](train.py#L294)。

6. **`mk_mmd_loss` 默认 5 核但训练时只传 2 核**。函数签名 `kernel_params=[0.1, 0.5, 1.0, 2.0, 5.0]` 默认值在训练中被覆盖为 `[0.5, 1.0]`。

7. **`umap` 包名是 `umap-learn`**，import 为 `import umap`，新手容易 `pip install umap` 装错包。

8. **`func.py` 里的 `KernelKMeans` 和 `find_min_distance` 当前训练流程不使用**——是早期实验或扩展接口，无须深究。

9. **早停在 `j > 200` 之前不会真停**——`e_min=200`。如果调试想快速验证，记得把 [train.py:422](train.py#L422) 的 `j>200` 改小，否则即使 patience 满了也不停。

10. **ResNet18 用了 ImageNet 预训练权重**——`pretrained=True`。第一层 conv 被改写后会丢弃预训练权重，但后续层保留，能加速收敛。如果离线训练拉不到预训练权重，可以改为 `pretrained=False`。

---

## 10. 如果要扩展/修改

- **想加 phase 通道**：把 [dataset.py:94-95](dataset.py#L94-L95) 改回 magnitude+phase 拼接，并把 [train.py:367](train.py#L367) 的 `channel=1` 改为 `channel=2`。

- **想换骨干网络**：改 [model.py](model.py) 的 `Resnet` 类，替换 `self.model = models.resnet18(...)` 即可。注意维持 (B, C, H, W) 输入接口。

- **想分别调 α₁、α₂**：修改 [train.py:17](train.py#L17) 为 `mmd_weight_local=...; mmd_weight_global=...`，然后分别用在 [train.py:257](train.py#L257) 和 [train.py:294](train.py#L294)。

- **想换核函数**：修改 [train.py:256, 278, 293](train.py#L256) 的 `kernel_types=['gaussian','gaussian'], kernel_params=[0.5, 1.0]` → 比如换成 Laplacian。但 `func.py` 当前只实现了 gaussian/linear，需要先扩展 `compute_kernel`。

- **想换数据集**：仿照 `load_zero_shot` 写一个新的加载函数，确保返回 `(magnitude, action, people)` 三个 np.array，shape 分别为 `(N, 100, 52)`、`(N,)`、`(N,)`。

- **想加新任务（如回归）**：当前是分类框架，损失 = CE + MMD。回归需要把 CE 换成 MSE，并思考"类别条件 MMD"如何定义（论文也提到这是 future work）。

---

## 11. 与论文对应关系一览表（精确到行号）

| 论文概念 / 公式 / 算法 | 代码实现 |
|---|---|
| 公式 (10) `E = ResNet(BatchNorm(X))` | [model.py:5-28](model.py#L5-L28) |
| 公式 (10) `Ŷ = MLP(E)` | [model.py:30-48](model.py#L30-L48) |
| 公式 (8) 核 MMD 可计算形式 | [func.py:33-66](func.py#L33-L66) `mk_mmd_loss` |
| 公式 (9) MK-MMD 多核加权 | [func.py:54-64](func.py#L54-L64) 多个核循环 |
| 公式 (14) Gaussian 核 | [func.py:20-25](func.py#L20-L25) `compute_kernel('gaussian')` |
| Algorithm 1: Help Set 构造 | [train.py:94-189](train.py#L94-L189) `top_knn` |
| Step 1 UMAP 降维 | [train.py:76-92](train.py#L76-L92) `dimension_reducation` |
| Step 1 KNN 投票 | [train.py:112-119](train.py#L112-L119) `KNeighborsClassifier` + `stats.mode` |
| Step 1 置信度 = 1/(d+ε) | [train.py:120-124](train.py#L120-L124) 但代码直接用 d，越小越好（等价）|
| Step 1 Top-p% 筛选 | [train.py:130-185](train.py#L130-L185) 三种 mode |
| 公式 (12) 局部 MMD（类内）| [train.py:245-257](train.py#L245-L257) 按 label 分类后调 `mk_mmd_loss` |
| 公式 (13) 全局 MMD | [train.py:282-294](train.py#L282-L294) `global_mmd=True` |
| 公式 (11) 总损失 | [train.py:233, 257, 279, 294](train.py#L233) `loss = L_cls + ...` |
| Algorithm 2: 自适应早停 | [train.py:382-429](train.py#L382-L429) main 循环里的状态机 |
| 跨域协议（leave-one-out）| [dataset.py:90-124](dataset.py#L90-L124) `load_zero_shot` |
| Table III 训练/支撑/测试集划分 | [train.py:328-365](train.py#L328-L365) main 编排 |

---

## 12. 论文亮点速记（讲给其他人听用）

1. **问题**：WiFi CSI 跨域时精度暴跌（49% → 同域 82%），需要 few-shot 跨域。

2. **核心 idea**：传统 MMD 做全局对齐 → 跨类样本拉混 → 改为**按类别条件做局部 MMD**。

3. **实现关键**：目标域没有标签 → 用 KNN+置信度筛选生成高质量伪标签子集（Help Set） → 在 Help Set 上做局部对齐。

4. **数学依据**：贝叶斯展开 `P_s(y|x) = P_t(y|x) · 比率项` → 要让条件分布对齐，必须**按类别**对齐边际分布。

5. **工程亮点**：支撑集**不参与训练**，可当目标域代理验证集 → 解决 few-shot 跨域"何时停训练"的难题。

6. **结果**：one-shot 设定下平均 82% 精度，逼近同域上限，显著高于传统 MK-MMD（55.7%）。

---

**文档维护提示**：这份 knowledge.md 基于代码当前快照与论文 v5 (arXiv:2412.04783)。如果代码后续大改（如换骨干网络、加多模态），需要同步更新对应章节，特别是 §4（逐文件详解）和 §11（公式↔代码对照）。
