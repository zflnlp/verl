# ScienceWorld GRPO 项目进度

## 项目概述
基于 verl 框架为 ScienceWorld benchmark 构建 GRPO 训练管线，训练 Qwen3-1.7B 模型完成科学实验任务。

## 当前状态
**阶段**: SFT cold start 完成，验证了 SFT 的有效性，准备用 SFT 模型启动 GRPO 训练

### 最新进展
- ✅ 多轮 GRPO 训练流程已验证（过程奖励、奖励传递、tokenization）
- ✅ SFT 训练完成 v2（FSDP 2卡，loss 0.0249，eval loss 0.0040）
- ✅ SFT 模型评估完成：学会了 `<action>` 标签和有效动作
- ✅ 确认 SFT cold start 的价值：跳过格式学习阶段，加速 GRPO 收敛
- ✅ 评估脚本修复（do_sample=True, temperature=0.4, max_steps=50）
- ✅ SFT/GRPO/Eval 三方 prompt 格式完全一致
- ⏳ Python 3.12 不兼容问题已解决（建议用 Python 3.10）
- ⏳ 准备用 SFT 模型进行多轮 GRPO 训练

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

**Base vs SFT 行为对比**：
```
Base:  "I need to figure out the task..." → 没有 <action> 标签 → 0 分
SFT:   "<action>open door to kitchen</action>" → 有效动作标签 → 0 分（但格式正确！）
```

**关键发现**：
- SFT 模型确实学会了 `<action>` 标签和有效动作
- 但无法根据环境反馈调整（如门已开了还去开）
- 这验证了 SFT + GRPO 两阶段训练的必要性

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

### 8. 评估 SFT 模型 ⏳
```bash
conda activate scienceworld
python examples/scienceworld_grpo/eval_all_tasks.py \
    --model_path /workspace/models/Qwen3-1.7B-SFT \
    --max_steps 30 --simplifications_preset easy
```

### 9. 用 SFT 模型进行多轮 GRPO 训练 ⏳
```bash
MODEL_PATH=/workspace/models/Qwen3-1.7B-SFT \
CUDA_VISIBLE_DEVICES=0-7 NGPUS_PER_NODE=8 \
bash examples/scienceworld_grpo/run_multiturn_training.sh
```

### 10. 训练后评估
- 对比 baseline（1.7B: 0.0, SFT: TBD, GRPO: TBD）
- 与 TCOD 论文结果对比

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
