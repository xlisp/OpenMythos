# OpenMythos 中文指南

> 本文档面向想从 0 跑起来的中文读者，覆盖：项目简介、架构与 MoE 设计、在 A800 × 8 上的训练部署、数据准备、SFT / RL 后训练方案、以及可预期的效果。
>
> 代码入口：`open_mythos/main.py`；训练脚本：`training/3b_fine_web_edu.py`；模型变体：`open_mythos/variants.py`。

---

## 1. 项目是什么

OpenMythos 是对 Claude Mythos 架构的开源**理论重建**（与 Anthropic 无官方关联）。核心是一个 **Recurrent-Depth Transformer（RDT / Looped Transformer）**：同一组权重在前向中被循环多次，用**深度**换**推理能力**，而不是一味堆参数。

一次前向分三段：

```
Input
  ↓
[Prelude P]           ← 普通 Transformer 层，跑一次
  ↓
[Recurrent Block R]   ← 循环 T 次（默认 max_loop_iters）
  ↑______↓            每一步都注入原始输入 e
  ↓
[Coda C]              ← 普通 Transformer 层，跑一次
  ↓
Output
```

循环块的递推公式（`open_mythos/main.py` 中实现）：

```
h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
```

其中 `A` 被参数化为 `-exp(log_A)` 的对角矩阵并经 ZOH 离散化，保证谱半径 `ρ(A) < 1`，从而**由构造保证稳定性**，避免 looped 模型常见的残差爆炸与 loss spike。

关键特性（均在代码中对应实现）：

| 特性 | 作用 | 代码位置 |
|---|---|---|
| Prelude + Recurrent + Coda | 分段架构，循环块共享权重 | `main.py: OpenMythos` |
| LTI 稳定注入 | `ρ(A)<1` 保证收敛 | `main.py: InputInjection` |
| ACT 自适应停止 | 每 token 学一个 halting 概率，难则多循环 | `main.py: ACT` |
| MLA / GQA 可切换 | 注意力机制可选 | `attn_type="mla" or "gqa"` |
| 细粒度 MoE + 共享专家 | 每个 token 激活 top-K + 共享专家 | `main.py: MoE` |
| 深度向 LoRA | 每一次 loop 用小秩 LoRA 做差异化 | `lora_rank` |
| RoPE | 旋转位置编码 | `precompute_rope_freqs` |

## 2. MoE 设计详解

OpenMythos 在**循环块的 FFN** 位置换成 MoE，而不是在所有层。这样：
- **深度（loop 次数）** 负责推理链条长度
- **宽度（MoE 多专家）** 负责跨领域知识（代码 / 数学 / 文学 / 科学…）

核心设计（见 `mythos_3b()` 为例）：

```python
n_experts=64              # 64 个路由专家
n_shared_experts=2        # 2 个永远激活的共享专家（吸收通用知识）
n_experts_per_tok=4       # 每个 token 路由到 top-4 专家
expert_dim=4096           # 单个细粒度专家的隐层维度
```

设计要点：

1. **细粒度切分（Fine-grained segmentation）**
   每个专家比传统 MoE 小，总专家数更多，路由组合更丰富。参见 DeepSeek-MoE 论文（arXiv:2401.06066）。

2. **共享专家（Shared experts）**
   少量专家对每个 token 都激活，用来装载"所有领域都用得到"的公共能力（语法、基础推理、通用语境），避免每个路由专家都把这些再学一遍，减少冗余。

3. **无辅助损失的负载均衡**
   路由器 logits 上加一个**动态偏置**来防止专家坍塌，不直接在 loss 里加 aux-loss，避免干扰主训练信号。相关修复见最近一次 commit：
   ```
   [bugf][moe-router-bias][stop load balance bias gradient leak]
   ```

4. **循环中每一步专家分布可以不同**
   `h_t` 在 loop 中不断演化，路由器会在不同深度选不同专家——形式上是"同一组权重"，实际每次迭代都在做不同的计算。

5. **激活比 ≈ (K + shared) / n_experts**
   以 3B 为例：`(4+2)/64 ≈ 9.4%`。即总参数 3B，**每 token 实际参与计算的约 9–10%**，所以"显存装得下 3B"不代表"每步都计算 3B"。

## 3. 模型规模概览（`open_mythos/variants.py`）

| 变体 | dim | experts | expert_dim | loop iters | ctx | 说明 |
|---|---|---|---|---|---|---|
| `mythos_1b` | 2048 | 64 | 2048 | 16 | 4k | 小型研究/微调 |
| `mythos_3b` | 3072 | 64 | 4096 | 16 | 4k | **推荐入门** |
| `mythos_10b` | 4096 | 128 | 5632 | 24 | 8k | 中等通用 |
| `mythos_50b` | 6144 | 256 | 9728 | 32 | 8k | 推理主力 |
| `mythos_100b` | 8192 | 256 | 13568 | 32 | 1M | 前沿级 |
| `mythos_500b` | 12288 | 512 | 23040 | 48 | 1M | 超大规模 |
| `mythos_1t` | 16384 | 512 | 34560 | 64 | 1M | 极限 |

---

## 4. A800 × 8 跑起来

**硬件前提**：8 × A800 80GB（总 640GB HBM），NVLink/NVSwitch 节点内互联。

**宿主软件前提（已由管理员安装，不需 root）**：
- Python 3.12.3（系统）
- torch 2.10.0+cu130（系统 site-packages）
- CUDA 13（系统驱动与 runtime）

> **不要用 `pip install torch`**：系统 torch 已适配 CUDA 13；我们只需把 `datasets / transformers / loguru / tensorboard` 叠加到用户态 venv。

### 4.1 环境准备（用户态 venv，继承系统 torch）

项目提供了 `runs/setup_env.sh`，一行搞定：

```bash
cd OpenMythos
bash runs/setup_env.sh
source .venv/bin/activate
```

它做的事：
1. `python -m venv --system-site-packages .venv` —— 创建 venv 时**继承系统 site-packages**，直接复用系统的 `torch 2.10.0+cu130`；
2. 打印 torch 版本、CUDA 可用性、GPU 数量做 sanity check；
3. `pip install -U datasets transformers loguru tensorboard huggingface_hub` —— 只装业务依赖；
4. 再次 import 验证各包版本。

可选覆盖（非默认 python 路径时）：

```bash
PYTHON=/opt/python3.12/bin/python3 VENV=/data/.venv bash runs/setup_env.sh
```

登录 HuggingFace（首次拉取 FineWeb-Edu 等数据集时需要）：

```bash
huggingface-cli login
```

### 4.2 一键端到端（pretrain → SFT → DPO → REPL）

项目在 `runs/` 下提供了完整编排脚本：

```
runs/
├── setup_env.sh      # 上一步用到
├── pretrain_3b.py    # FineWeb-Edu 预训练，FSDP + TensorBoard
├── sft_3b.py         # Alpaca 指令微调，prompt-masked loss
├── dpo_3b.py         # policy + 冻结 ref 的 DPO，偏好对
├── eval_repl.py      # 交互式 REPL，调 n_loops / temperature / top_k
└── run_all.sh        # 自动 nohup 后台编排器（本节主角）
```

默认启动（自动 nohup 到后台 + 自动起 TensorBoard server）：

```bash
bash runs/run_all.sh
```

它会立即返回：

```
[run_all] PID=12345
[run_all] log=logs/run_all_20260420_231512.log
[run_all] tail:  tail -f logs/run_all_20260420_231512.log
[run_all] stop:  kill 12345
[run_all] tb:    tensorboard --logdir logs/tb --port 6006 --bind_all
[run_all] repl:  python runs/eval_repl.py   (after stages finish)
```

内部逻辑：
- 首次调用时 `exec` 一次自身到 `nohup ... &`，父进程立刻退出，不会因为终端断开被 kill；
- 激活 `.venv`，按 `nvidia-smi` 自动选 `NPROC=$(torch.cuda.device_count())`；
- 顺序跑 `pretrain → sft → dpo`，每阶段单独写日志 `logs/{stage}_<时间戳>.log`；
- `MYTHOS_TB_DIR` 按阶段分桶：`logs/tb/{pretrain,sft,dpo}`，TensorBoard server 后台拉起在 `:6006`；
- REPL 是交互式的，后台模式下会**跳过**并提示手动起。

**常用环境变量覆盖**（都可以 `VAR=... bash runs/run_all.sh` 传入）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `STAGES` | `pretrain,sft,dpo,repl` | 只跑子集，例如 `STAGES=sft,dpo` |
| `FOREGROUND` | `0` | `=1` 不 nohup，在当前 shell 跑（调试用） |
| `NPROC` | 自动 | 覆盖 GPU 数 |
| `VENV` | `./.venv` | venv 路径 |
| `TB_PORT` | `6006` | TensorBoard 端口 |
| `START_TB` | `1` | `=0` 不自动起 TB server |
| `MYTHOS_TARGET_TOKENS` | `30000000000` | 预训练目标 token 数；冒烟测试可设 `50000000` |
| `MYTHOS_SFT_EPOCHS` | `2` | SFT 轮数 |
| `MYTHOS_DPO_BETA` | `0.1` | DPO 温度 |

### 4.3 TensorBoard 监控

三个阶段都在 master rank 写 `SummaryWriter` 事件：

| 阶段 | 记录的标量 |
|---|---|
| `pretrain/*` | `loss`, `grad_norm`, `lr`, `tokens_per_sec`, `tokens_seen_B` |
| `sft/*` | `loss`, `grad_norm`, `lr` |
| `dpo/*` | `loss`, `accuracy`（policy 偏好 chosen 的比例）, `grad_norm`, `lr` |

`run_all.sh` 默认已经起了 TB server，浏览器打开 `http://<节点 IP>:6006/` 直接看。若想手动起：

```bash
source .venv/bin/activate
tensorboard --logdir logs/tb --port 6006 --bind_all
```

想只看某个阶段：`--logdir logs/tb/pretrain`。

### 4.4 日志与后台运维

```bash
# 看总日志
tail -f logs/run_all_*.log

# 看某阶段详细日志
tail -f logs/pretrain_*.log
tail -f logs/sft_*.log
tail -f logs/dpo_*.log

# TensorBoard 自己的 server 日志
tail -f logs/tensorboard.log

# 查后台进程
ps -p $(cat /tmp/openmythos.pid 2>/dev/null) 2>/dev/null || pgrep -f 'runs/run_all.sh'

# 停止
kill <PID>
```

断点：所有三个阶段都实现了 `_list_ckpts(...)` 自动发现最新 ckpt 续跑；杀掉进程再 `bash runs/run_all.sh` 即可从断点继续。

### 4.5 在 A800 × 8 上能跑多大

以下为经验估计（FSDP FULL_SHARD + bf16 + 激活检查点，大致可行性）：

| 变体 | 8×A800 可行性 | 备注 |
|---|---|---|
| 1B / 3B | 充裕 | 默认脚本直接跑，吞吐最高 |
| 10B | 可跑 | 需要适当降 `micro_batch` 或开 activation checkpointing |
| 50B | 边缘 | 需打开 CPU offload、更大 grad accum、seq_len 减到 2k |
| 100B+ | 单机 8×A800 不建议 | 需要多节点或更多 GPU |

推荐从 **`mythos_3b`** 起步把流程跑通，再换 `mythos_10b`。

### 4.6 超参数调整（通过环境变量，不用改代码）

`runs/pretrain_3b.py` 已把常用超参暴露为环境变量，`run_all.sh` 透传：

```bash
MYTHOS_SUBSET=sample-100BT \
MYTHOS_TARGET_TOKENS=100000000000 \
MYTHOS_SEQ_LEN=2048 \
MYTHOS_MICRO_BATCH=4 \
MYTHOS_WARMUP=2000 \
MYTHOS_LR=3e-4 \
MYTHOS_CKPT_EVERY=1000 \
    bash runs/run_all.sh
```

SFT / DPO 同理：`MYTHOS_SFT_LR`, `MYTHOS_SFT_EPOCHS`, `MYTHOS_DPO_LR`, `MYTHOS_DPO_BETA`, `MYTHOS_DPO_DATASET` …

有效全局 batch（tokens）= `world_size × micro_batch × grad_accum × seq_len`，默认 `8×4×8×2048 ≈ 524k tokens/步`。

### 4.7 常见坑位

- **checkpoint 恢复**：FSDP 下模型和优化器必须在**同一个** `FULL_STATE_DICT` 上下文内 gather，否则会 silent 错位。代码里已经处理好，但若你改写这一段请保留该模式。
- **数据流式不可 seek**：断点重启时 dataset 会从头重拉，预训练规模下可接受。
- **梯度裁剪**：FSDP 下必须用 `model.clip_grad_norm_(1.0)`，`nn.utils.clip_grad_norm_` 只看本地分片，不对。

---

## 5. 数据准备

### 5.1 预训练数据（已由脚本内置）

| 数据 | HuggingFace | 规模 | 用途 |
|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 1.3T tokens | 主力预训练 |
| OpenHermes 2.5 | `teknium/OpenHermes-2.5` | ~1M 条 | 混 5% 提升指令跟随 |
| OpenWebMath | `open-web-math/open-web-math` | 14.7B | 数学/推理增强（≥10B 模型推荐） |

按模型规模建议的 token 预算（循环模型比普通 Transformer 更省 token）：

| 变体 | Chinchilla 最优 | 循环模型推荐 |
|---|---|---|
| 1B | ~20B | ~10–15B |
| 3B | ~60B | ~30–40B |
| 10B | ~200B | ~100–150B |
| 50B+ | ~1T+ | ~500B+ |

Tokenizer 默认是 `openai/gpt-oss-20b`，封装在 `open_mythos/tokenizer.py: MythosTokenizer`。

### 5.2 SFT 数据（推荐组合）

| 数据 | 用途 |
|---|---|
| OpenHermes-2.5 | 通用指令 |
| UltraChat 200k | 多轮对话 |
| MetaMathQA / OpenMathInstruct | 数学推理 |
| CodeAlpaca / Magicoder-Evol-Instruct | 代码 |
| Anthropic HH-RLHF 的 chosen 分支 | 有害-无害对话范式 |

### 5.3 偏好 / RL 数据

| 数据 | 用途 |
|---|---|
| UltraFeedback | DPO / PPO 偏好数据 |
| Anthropic HH-RLHF | 安全偏好 |
| PRM800K | 过程奖励（Process Reward Model），适合 looped 推理 |
| GSM8K / MATH | 数学可验证奖励 |

---

## 6. SFT 阶段

预训练收敛后，进行指令微调：

1. **数据格式化**：统一成 chat template（system / user / assistant），`MythosTokenizer` 不带对话模板时可按 ChatML 自行拼接。
2. **loss mask**：只在 assistant token 上计算 loss。
3. **超参建议**（3B 规模）：
   - 学习率：`2e-5 ~ 5e-5`（远小于预训练的 3e-4）
   - 轮数：1–3 epoch，避免过拟合损失泛化
   - batch：每 step ~1–2M tokens
   - seq_len：4096（与预训练保持一致）
4. **循环深度**：训练阶段可把 `n_loops` 固定在 `max_loop_iters`（例如 16），推理再按需降低以节省计算。

最小改法：复制 `training/3b_fine_web_edu.py`，将 `FineWebEduDataset` 换成 SFT 数据集，把 `cross_entropy` 的 target 加入 prompt mask 即可。

---

## 7. RL / 偏好优化阶段

looped 架构天然适合"可验证任务 + 深度推理"，推荐顺序：

### 7.1 DPO（推荐起点）

- 最省算力，不需要 reward model；
- 数据：UltraFeedback / HH-RLHF 的 (chosen, rejected) 对；
- 实践：可接入 `trl` 的 `DPOTrainer`，先把 OpenMythos 封成 `AutoModelForCausalLM` 兼容接口（需写 adapter）。
- A800×8 训 3B DPO 轻松，3–5k 步即可看到指令跟随和风格提升。

### 7.2 RLHF / PPO（更强但更贵）

- 需要 reward model（可用 OpenAssistant RM 或自训）；
- `trl` 的 `PPOTrainer`；
- 小心 KL 爆炸，建议用参考模型冻结的 SFT 权重。

### 7.3 RLVR / 过程奖励（最契合 looped 模型）

looped 架构的优势是**隐式 CoT**：循环块相当于 T 步连续潜空间推理。可直接对最终答案做**可验证奖励**（RLVR），例如：

- GSM8K / MATH：答案匹配 → 奖励；
- 代码任务：单元测试通过 → 奖励；
- PRM800K：对中间步骤打分（PRM）。

策略：把 loop 次数当做"思考时间"——奖励正确答案的同时，对 loop 数加一个小惩罚（鼓励在解对的前提下更少循环）。这也是 ACT 机制的自然延伸。

---

## 8. 可预期的效果（参考系）

由于 OpenMythos 是**理论重建**，下面给出的是基于论文与相似架构的预期，而非官方 benchmark：

| 模型规模 | 预训练 tokens | 预期定位 |
|---|---|---|
| 3B（30–40B token，8×A800 数周） | ~30B | 约等于传统 6B dense 水平；在推理类任务（GSM8K、简单代码）上由于 loop 深度有显著超越 |
| 10B（100–150B token，需更多卡或更长时间） | ~120B | 达到 ~20B dense 水平；数学 / 多跳推理接近前沿 20B 开源模型 |
| 50B+ | 500B+ | 进入前沿开源竞争区间 |

核心收益（来自 Parcae / Saunshi 等论文结论，以及 README 中的理论分析）：

1. **参数效率**：770M looped ≈ 1.3B dense（同数据同 loss）；
2. **深度外推**：训 5-hop 推理可在推理期跑 10-hop；
3. **推理期可调**：推理时 `n_loops` 越大推理越深，且呈**可预测的指数衰减**收敛；
4. **批内不等深**：Continuous Depth-wise Batching 理论上 2–3× 吞吐提升。

**局限**：
- 循环结构天然偏向**组合推理**而非**事实记忆**，纯知识问答不一定比同参 dense 模型强；
- 训练稳定性虽已通过 LTI 约束解决，但 MoE 负载均衡、ACT halting 阈值、每一步专家选择仍需精心调参。

---

## 9. 一条可跑的上手路径

```bash
# 1. 环境
pip install -r requirements.txt -r training/requirements.txt
huggingface-cli login

# 2. 跑通 3B 预训练（10BT 子集）
torchrun --standalone --nproc_per_node=8 training/3b_fine_web_edu.py

# 3. 切换到 sample-100BT 做正式预训练
# 编辑 training/3b_fine_web_edu.py 中 dataset_subset = "sample-100BT"

# 4. SFT：复制脚本，换数据集 + loss mask
# 5. DPO：trl 接入，(chosen, rejected) 数据对
# 6. 可验证 RL：GSM8K / 代码单测任务
```

---

## 10. 参考

- Parcae — Scaling Laws for Stable Looped Language Models: https://arxiv.org/abs/2604.12946
- Reasoning with Latent Thoughts — On the Power of Looped Transformers: https://arxiv.org/abs/2502.17416
- Universal Transformers：https://arxiv.org/pdf/1807.03819
- DeepSeek-MoE（共享 + 细粒度专家）：https://arxiv.org/abs/2401.06066
- Relaxed Recursive Transformers（深度向 LoRA）：https://arxiv.org/pdf/2410.20672

> 代码对应关系请查 `open_mythos/main.py`（核心架构）、`open_mythos/variants.py`（规模预设）、`training/3b_fine_web_edu.py`（端到端训练示例）。
