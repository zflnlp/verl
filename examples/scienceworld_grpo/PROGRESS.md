# ScienceWorld GRPO 项目进度

## 项目概述
基于 verl 框架为 ScienceWorld benchmark 构建 GRPO 训练管线，训练 Qwen3-1.7B 模型完成科学实验任务。

## 当前状态
**阶段**: SFT cold start + GRPO 训练进行中（vllm 引擎，CUDA 13.1 新机器）

### 最新进展
- ✅ verl 升级到 main 分支（agent_loop 新架构）
- ✅ 引擎从 sglang 切换到 vllm（依赖更简单）
- ✅ CUDA 13.1 兼容性问题全部解决
- ✅ SFT 训练完成 v2（loss 0.0249，eval loss 0.0040）
- ✅ SFT + GRPO 训练已启动，初始效果极好
- 🎉 **GRPO Step 1**: 平均分数 0.459，最大分数 1.000（无 SFT 时仅 0.028/0.250）
- ⏳ GRPO 训练运行中（56 步，~5.6 分钟/步）

### SFT 版本
| 分支 | verl 版本 | 引擎 | CUDA | 机器 |
|------|----------|------|------|------|
| `webshop-grpo-v0.4.1` | v0.4.1 | sglang | 12.4 | hgx18 (旧) |
| **`webshop-grpo-main`** | main (0.9.0.dev) | **vllm** | **13.1** | 容器 (新) |

### SFT Cold Start 效果
| Step 1 指标 | 无 SFT (旧) | 有 SFT (新) | 提升 |
|------------|------------|------------|------|
| 平均分数 | 0.028 | **0.459** | 16x |
| 最大分数 | 0.250 | **1.000** | 4x |
| 奖励计算 | 342s | **3.4e-5s** | 1e7x |

## 源码参考
- ScienceWorld 源码: `benchmarks/ScienceWorld/`
- llama-factory 源码: `llama-factory/`（git submodule, v0.9.3）
- 数据目录: `data/`

---

## 训练流程

### SFT + RL 两阶段训练
```
1. Gold Trajectories → SFT (llama-factory) → Qwen3-1.7B-SFT
2. Qwen3-1.7B-SFT → Multi-turn GRPO (verl+sglang) → 最终模型
3. 最终模型 → eval_all_tasks.py → 论文结果
```

---

## SFT Cold Start

### SFT 的价值
SFT cold start 的核心作用是**让模型学会格式和基本动作**，而不直接提升任务完成率：

| 阶段 | 作用 | 分数贡献 |
|------|------|----------|
| Base | 不会输出 `<action>` 标签 | 0%（无法执行） |
| **SFT** | **学会 `<action>` 标签 + 有效动作 + 任务流程** | **0%（会动作但不会适应环境）** |
| GRPO | 学会根据环境反馈调整策略 | **10-18%（最终分数）** |

**SFT 跳过 RL 的低效探索阶段，让 GRPO 从一开始就在做有意义的探索。**

### SFT 模型评估结果

**Base vs SFT 行为对比（boil 任务，50 steps）**：

| 维度 | Base 模型 | SFT 模型 | 提升 |
|------|-----------|----------|------|
| `<action>` 标签输出 | ❌ 不会 | ✅ 正确输出 | ∞ |
| 动作有效性 | ❌ 随机文本 | ✅ 环境可执行动作 | ∞ |
| 任务流程理解 | ❌ 无 | ✅ 知道 gold 轨迹步骤 | ∞ |
| 环境反馈适应 | ❌ 不会 | ❌ 不会（需 GRPO） | — |
| **Success Rate（所有任务）** | **0%** | **0%** | **格式学会，策略待 GRPO** |

**Base 模型输出**：
```
Step 1: look around → look around → go to greenhouse → look around → ...
        随机动作，没有 <action> 标签
```

**SFT 模型输出**：
```
Step 1: <action>open door to kitchen</action>  ← 来自 gold 轨迹第 2 步！
Step 2-10: <action>open door to kitchen</action>  ← 死循环（门已开还去开）
```

**关键发现**：
1. SFT 模型确实学会了 `<action>` 标签和有效动作（✅ 核心目标达成）
2. SFT 模型记住了 gold 轨迹的动作序列（"open door to kitchen" 是黄金轨迹第 2 步）
3. SFT 模型无法根据环境反馈动态调整策略（门已开了还重复去开）
4. 这验证了 **SFT + GRPO 两阶段训练**的必要性：
   - SFT：学会"说话"（格式）+ "知道干什么"（任务结构）
   - GRPO：学会"适应环境"（动态调整策略）

**评估配置**（与 TCOD 论文对齐）：
- `do_sample=True, temperature=0.4`
- `max_steps=50`（boil 需 40+ 步完成）
- 评估所有 30 个任务，官方 test split

### 数据来源
- ScienceWorld 官方 gold trajectories（`benchmarks/ScienceWorld/goldpaths/goldpaths-all.zip`）
- 29 个任务，6,907 个 variations，453,345 步
- 官方 train/dev/test 划分

| 集合 | 数量 |
|------|------|
| Train | 3,442 |
| Dev | 1,721 |
| Test | 1,744 |
| **Total** | **6,907** |

### SFT 训练配置
| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-1.7B → Qwen3-1.7B-SFT-v2 |
| 方法 | 全量微调（full fine-tuning） |
| GPU | 2×H100（FSDP 分片） |
| Epochs | 3 |
| Batch size | 1（per GPU） |
| Gradient accumulation | 4 |
| Learning rate | 2e-5 (cosine schedule) |
| Max seq length | 8,192 |
| 精度 | fp16 |

### SFT 训练结果
| 指标 | v1（4096 token） | **v2（8192 token）** |
|------|------------------|----------------------|
| Train loss | 0.0411 | **0.0249** |
| Eval loss | 0.0041 | **0.0040** |
| 训练时间 | 47 分钟 | 48 分钟 |
| 模型保存 | `/workspace/models/Qwen3-1.7B-SFT` | `/workspace/models/Qwen3-1.7B-SFT-v2` |

### 评估脚本修复
| 问题 | 修复 |
|------|------|
| 贪婪解码 → 模型死循环 | `do_sample=True, temperature=0.4`（TCOD 论文 Table 5） |
| 30 步无法完成任务 | `max_steps=50`（boil 需要 40 步） |

### SFT 模型路径
- v1: `/workspace/models/Qwen3-1.7B-SFT`（4096 token，已弃用）
- **v2: `/workspace/models/Qwen3-1.7B-SFT-v2`（8192 token，当前使用）** ⭐

---

## TCOD 论文参数对照表

### 参考论文
- **TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents**
- 算法：OPD（我们使用 GRPO，训练配置对齐）

### 关键词对比
| 参数 | 论文值 | 我们的值 | 状态 |
|------|--------|----------|------|
| Learning rate | 1e-6 | 1e-6 | ✅ |
| Gradient clipping | 1.0 | 1.0 | ✅ |
| Train batch size | 64 | 64 | ✅ |
| Max prompt tokens | 10,240 | 10,240 | ✅ |
| Max response tokens | 512 | 512 | ✅ |
| Temperature (train) | 1.0 | 1.0 | ✅ |
| Temperature (eval) | 0.4 | 0.4 | ✅ |
| ScienceWorld max steps | 30 | 30 | ✅ |
| Seed | 42 | 42 | ✅ |
| GPU memory util | 0.7 | 0.7 | ✅ |
| Data type | BFloat16 | BFloat16 | ✅ |

### 论文结果（ScienceWorld）
| 模型 | 方法 | Success Rate |
|------|------|-------------|
| Qwen3-30B (Teacher) | — | 18.42% |
| Qwen3-1.7B | Zero-Shot | 0.00% |
| Qwen3-1.7B | TCOD-B2F (η=4) | **11.34%** |
| Qwen3-4B | TCOD-F2B (η=2) | **17.85%** |

---

## 已完成工作

### 1. 环境搭建 ✅
- verl 0.4.1 + vllm 0.8.5 + sglang 0.4.6.post5
- 服务器: hgx18 (8x H100 80GB)
- llama-factory v0.9.3 (git submodule)

### 2. Prompt 格式对齐 ✅
**⚠️ 不要修改！** GRPO/SFT/Eval 三者格式完全一致：
```
Your ScienceWorld task is: {task}
Prior to this step, you have already taken {n} step(s).
Below are the most recent {h} observations and the corresponding actions you took:
{history}
You are now at step {s} and your current observation is:
{observation}
Your valid actions of the current situation are: [{actions}].
Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.
```

### 3. 数据生成 ✅
- GRPO 数据: 30 tasks, train=3,592 / dev=1,796 / test=1,819 (`/workspace/data/scienceworld_all/`)
- SFT 数据: 29 tasks, train=3,442 / dev=1,721 / test=1,744 (`/workspace/data/scienceworld_sft/`)

### 4. 单轮 GRPO 验证 ✅
- 3 steps, 14 samples
- 平均奖励 0.001, 最大 0.020（有学习信号）

### 5. 多轮 GRPO 配置 ✅
- sglang + multi_turn.enable=True
- 过程奖励（dense rewards）
- 修复：-100 分数问题
- 修复：多轮奖励传递（convert_multiturn_rewards_to_rm_scores）
- 修复：tokenization 不一致（use_inference_chat_template: True）

### 6. SFT Cold Start ✅
- llama-factory v0.9.3 全量微调
- 训练完成：train loss 0.0411, eval loss 0.0041
- 模型保存：`/workspace/models/Qwen3-1.7B-SFT`

### 7. 评估脚本 ✅
- `eval_all_tasks.py`：评估所有 30 个任务，输出论文级结果（SR, Avg Score, Avg Steps）
- `eval_zero_shot.py`：单任务评估
- `debug_eval.py` / `debug_sft_model.py`：调试脚本

---

## 待完成工作

### 12. GRPO 训练收敛 ⏳
- 当前 Step 1：平均分 0.459，最大分 1.000
- 观察后续步骤分数变化
- 已有满分样本，验证 SFT cold start 有效

### 13. GRPO 训练后评估
- 对比 baseline（1.7B: 0.0, SFT: 0%, GRPO: TBD）
- 与 TCOD 论文结果对比（Qwen3-1.7B TCOD: 11.34%）

### 14. 论文结果整理
- 三阶段训练完整对比：Base → SFT → GRPO
- 注意：TCOD 论文用 OPD，我们用 GRPO（不同算法）

## 架构升级记录

### verl main 分支关键变化
| 组件 | v0.4.1 (旧) | main (新) |
|------|-----------|----------|
| 多轮交互 | `BaseInteraction` | `AgentLoopBase` |
| 引擎 | sglang | **vllm** |
| 交互实现 | `ScienceWorldInteraction` | `ScienceWorldAgentLoop` |
| 奖励管理 | `convert_multiturn_rewards_to_rm_scores()` | AgentLoop 自动处理 |
| 配置格式 | `multi_turn.enable=true` | `agent.default_agent_loop` |

### 解决的关键问题
1. **flash_attn 不兼容** → `attention_utils.py` 添加 torch fallback
2. **multi_turn undefined** → `ray_trainer.py` 从 meta_info 获取
3. **Multimodal 错误** → AgentLoop 移除无效参数
4. **DeepGEMM 缺失** → `VLLM_USE_DEEP_GEMM=0` + `VLLM_SKIP_WARMUP=1`
5. **flashinfer 版本冲突** → `FLASHINFER_DISABLE_VERSION_CHECK=1`
6. **merge conflict** → 修复 `.gitmodules`, `reward.py`, `ray_trainer.py`, `version/version`
7. **CUDA 13.1** → verl main 原生支持

---

## 关键文件

### 训练
| 文件 | 说明 |
|------|------|
| `run_multiturn_training.sh` | **多轮 GRPO 训练（sglang）** ⭐ |
| `run_sft.sh` | **SFT 训练（llama-factory）** ⭐ |
| `run_sft_distributed.py` | SFT 分布式训练（FSDP） |
| `run_real.sh` | 单轮训练（vllm） |
| `run_mock.sh` | Mock 训练 |

### 数据
| 文件 | 说明 |
|------|------|
| `data_preprocess.py` | GRPO 数据生成（30 任务） |
| `prepare_sft_data.py` | **SFT 数据准备（gold trajectories）** ⭐ |
| `generate_all_data.sh` | 批量数据生成 |

### 评估
| 文件 | 说明 |
|------|------|
| `eval_all_tasks.py` | **全任务评估（论文级）** ⭐ |
| `eval_zero_shot.py` | 单任务评估 |
| `debug_eval.py` | 调试评估 |
| `debug_sft_model.py` | 调试模型输出 |

### 配置
| 文件 | 说明 |
|------|------|
| `config/scienceworld_multiturn_grpo.yaml` | 多轮 GRPO 配置 |
| `config/scienceworld_grpo.yaml` | 单轮配置 |

### 核心代码
| 文件 | 说明 |
|------|------|
| `verl/interactions/scienceworld_interaction.py` | 多轮交互 + 过程奖励 |
| `verl/tools/scienceworld_tool.py` | Action 工具 |
| `verl/trainer/ppo/reward.py` | 多轮奖励传递 |

---

## 技术决策
| 决策 | 原因 |
|------|------|
| SFT cold start | 原始模型不会用 `<action>` 标签 |
| 全量微调（非 LoRA） | 1.7B 显存足够，效果更好 |
| 多轮 + sglang | ScienceWorld 需 30 步交互 |
| 过程奖励 | 学习信号更密集，收敛更快 |
| llama-factory v0.9.3 | 支持 transformers 4.51.1 |

---

## 注意事项
- verl checkpoint 不是 HuggingFace 格式，需要转换
- llama-factory 是 git submodule：`git submodule update --init --recursive`
- SFT 数据路径：`/workspace/data/scienceworld_sft/`
- GRPO 数据路径：`/workspace/data/scienceworld_all/`
- SFT 模型路径：`/workspace/models/Qwen3-1.7B-SFT`

## 恢复指令
1. `git pull && git submodule update --init --recursive`
2. 生成数据: `bash examples/scienceworld_grpo/generate_all_data.sh`
3. 准备 SFT 数据: `python examples/scienceworld_grpo/prepare_sft_data.py`
4. 运行 SFT: `conda activate llamafactory && CUDA_VISIBLE_DEVICES=X bash examples/scienceworld_grpo/run_sft.sh`
5. 评估 SFT: `python examples/scienceworld_grpo/eval_all_tasks.py --model_path /workspace/models/Qwen3-1.7B-SFT`
6. 多轮 GRPO: `MODEL_PATH=/workspace/models/Qwen3-1.7B-SFT bash examples/scienceworld_grpo/run_multiturn_training.sh`

---

## 📋 进度日志

### 2026-06-29 — verl 升级 + SFT cold start GRPO 启动

**背景**：切换到 CUDA 13.1 新机器（8×H200），旧 verl v0.4.1 依赖全面不兼容。

**关键决策**：
- 升级 verl 到 main 分支（0.9.0.dev），使用新的 AgentLoop 架构
- 引擎从 sglang 切换到 vllm（依赖更简单）
- 创建新分支 `webshop-grpo-main`

**创建的核心文件**：
- `verl/experimental/agent_loop/scienceworld_agent_loop.py` — 新的 AgentLoop
- `examples/scienceworld_grpo/run_multiturn_v2.sh` — v2 训练脚本
- `examples/scienceworld_grpo/config/scienceworld_multiturn_v2.yaml` — v2 配置

**解决的依赖问题**：
1. flash_attn 不兼容 → `attention_utils.py` 添加 torch fallback
2. `multi_turn` undefined → `ray_trainer.py` 从 meta_info 获取
3. Multimodal 错误 → AgentLoop 移除无效参数
4. DeepGEMM 缺失 → `VLLM_USE_DEEP_GEMM=0` + `VLLM_SKIP_WARMUP=1`
5. flashinfer 版本冲突 → `FLASHINFER_DISABLE_VERSION_CHECK=1`
6. merge conflict → 修复 4 个文件的冲突标记

**SFT Cold Start GRPO 初始结果**：
| Step | 平均分 | 最大分 | 说明 |
|------|--------|--------|------|
| 1 | 0.459 | 1.000 | SFT cold start 效果显著（无 SFT 仅 0.028） |

### 2026-06-30 — Java fd 爆炸 + 训练稳定性修复

**问题 1：py4j select() fd 溢出**

ScienceWorld 通过 py4j 创建 Java 进程，8 个 AgentLoop workers 同时创建导致 fd 爆炸。

**修复历程**：
1. 尝试 `ulimit -n 65535` → 无效
2. 尝试 `agent.num_workers=2` → 无效
3. 尝试 `agent.num_workers=1` → 过于保守
4. ✅ **最终方案**：信号量限流 `SCIENCEWORLD_MAX_CONCURRENT_ENVS=4`

**问题 2：max_num_batched_tokens 不足**

prompt(10240) + response(8192) = 18432 > 默认 16384。

**修复**：`max_num_batched_tokens: 20480`

**问题 3：env.close() 不完整**

`del env` 不关闭 Java gateway，改为 `env.close()`。

**训练进展**：
| Step | 平均分 | 最大分 | 验证分 |
|------|--------|--------|--------|
| 1 | 0.459 | 1.000 | - |
| 4 | 0.750 | 1.000 | - |
| 5 | 0.745 | 1.000 | 0.750 |
| 6 | 0.750 | 1.000 | - |

**配置调整**：
- `save_freq: 250 → 10`（每 10 步保存 checkpoint）
- `test_freq: 5`（每 5 步验证）

**待做**：
- GRPO 训练完成 → 评估对比 Base/SFT/GRPO
