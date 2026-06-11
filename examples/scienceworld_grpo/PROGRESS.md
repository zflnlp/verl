# ScienceWorld GRPO 项目进度

## 项目概述
基于 verl 框架为 ScienceWorld benchmark 构建 GRPO 训练管线，训练 Qwen3-1.7B 模型完成科学实验任务。

## 当前状态
**阶段**: 扩展训练数据已完成，准备运行完整训练（对齐 TCOD 论文）

## 源码参考
- ScienceWorld 源码: `benchmarks/ScienceWorld/`
- WebShop 源码: `benchmarks/WebShop/`
- alfworld 源码: `benchmarks/alfworld/`
- 数据目录: `data/`

---

## TCOD 论文参数对照表

### 参考论文
- **标题**: TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents
- **作者**: Jiaqi Wang, Wenhao Zhang, Weijie Shi, Yaliang Li, James Cheng (Tongyi Lab, Alibaba Group)
- **论文使用**: Qwen3-1.7B 作为 student，Qwen3-30B-A3B-Instruct 作为 teacher
- **算法**: On-Policy Distillation (OPD)，我们使用 GRPO（不同算法，但训练配置对齐）

### 论文 Table 4: Training Hyperparameters 详细对照

#### 算法配置
| 参数 | 论文值 (OPD) | 我们的值 (GRPO) | 说明 |
|------|-------------|-----------------|------|
| Algorithm type | On-Policy Distillation | GRPO | 不同算法 |
| Advantage function | Multi-turn OPD | GRPO | 不同算法 |
| KL coefficient | 1.0 | 0.001 | OPD 的 distillation loss vs GRPO 的 KL penalty |
| Learning rate | 1×10⁻⁶ | 1×10⁻⁶ | ✅ 已对齐 |
| Gradient clipping | 1.0 | 1.0 | ✅ 已对齐 |
| Repeat times | 1 | 1 | ✅ 已对齐 |

#### 训练配置
| 参数 | 论文值 | 我们的值 | 状态 |
|------|--------|----------|------|
| Total training steps | 250 | 250 | ✅ 已对齐 |
| Batch size | 16 | 16 | ✅ 已对齐 |
| Train batch size | 64 | 64 | ✅ 已对齐 |
| Save interval | 250 | 250 | ✅ 已对齐 |
| Evaluation interval | 5 steps | 5 steps | ✅ 已对齐 |
| Seed | 42 | 42 | ✅ 已对齐 |

#### 模型配置
| 参数 | 论文值 | 我们的值 | 状态 |
|------|--------|----------|------|
| Max prompt tokens | 10,240 | 10,240 | ✅ 已对齐 |
| Max response tokens | 512 | 512 | ✅ 已对齐 |

#### 推理配置
| 参数 | 论文值 | 我们的值 | 状态 |
|------|--------|----------|------|
| Temperature (training) | 1.0 | 1.0 | ✅ 已对齐 |
| Temperature (evaluation) | 0.4 | 0.4 | ✅ 已对齐 |
| Logprobs | Enabled (all tokens) | Enabled | ✅ 已对齐 |

#### 环境配置
| 参数 | 论文值 | 我们的值 | 状态 |
|------|--------|----------|------|
| ScienceWorld max steps | 30 | 30 | ✅ 已对齐 |
| History length | 2 steps | 3 steps | ⚠️ 略有不同 |

#### 硬件配置（因硬件不同而异）
| 参数 | 论文值 | 我们的值 | 说明 |
|------|--------|----------|------|
| Number of nodes | 1 | 1 | ✅ |
| GPUs per node | 8×H20 (96GB) | 1×H100 (80GB) | 硬件不同 |
| Tensor parallel size | 2 | 1 | 单卡不需要 |
| Sequence parallel size | 2 (Ulysses) | 无 | 单卡不需要 |
| Max tokens per GPU | 16,384 | 16,384 | ✅ 已对齐 |
| GPU memory utilization | 0.7 | 0.7 | ✅ 已对齐 |
| Data type | BFloat16 | BFloat16 | ✅ 已对齐 |

### 论文结果（ScienceWorld）
| 模型 | 方法 | Success Rate |
|------|------|-------------|
| Qwen3-30B (Teacher) | — | 18.42% |
| Qwen3-1.7B | Zero-Shot | 0.00% |
| Qwen3-1.7B | SFT | 0.00% |
| Qwen3-1.7B | Vanilla OPD | 0.05% |
| Qwen3-1.7B | TCOD-B2F (η=2) | 10.82% |
| Qwen3-1.7B | TCOD-B2F (η=4) | **11.34%** |
| Qwen3-1.7B | TCOD-B2F (η=6) | 10.65% |
| Qwen3-1.7B | TCOD-F2B (η=2) | 10.45% |
| Qwen3-1.7B | TCOD-F2B (η=4) | 9.22% |
| Qwen3-1.7B | TCOD-F2B (η=6) | **11.08%** |
| Qwen3-4B | Vanilla OPD | 15.95% |
| Qwen3-4B | TCOD-F2B (η=2) | **17.85%** |

### 论文评估配置 (Table 5)
| 参数 | 论文值 | 我们的值 | 状态 |
|------|--------|----------|------|
| Maximum tokens | 4,096 | 4,096 | ✅ |
| Temperature | 0.4 | 0.4 | ✅ |
| Top-p | 1.0 | 1.0 | ✅ |
| Max environment steps | 30 | 30 | ✅ |
| History length | 2 steps | 3 steps | ⚠️ 略有不同 |
| Number of workers | 8 | 1 | 硬件不同 |

---

## 已完成工作

### 1. 环境搭建 ✅
- verl 0.4.1 + vllm 0.8.5
- 服务器: hgx18 (8x H100 80GB)
- 模型: `/workspace/models/Qwen3-1.7B`（1.7B）和 `/workspace/models/Qwen3-14B`（14B）
- ScienceWorld 包已安装

### 2. ScienceWorld API 适配 ✅
所有 camelCase 改为 snake_case：
- `ScienceWorldEnv()` 无参数构造
- `env.load()` 返回 None（不是元组）
- `taskDescription()` → `taskdescription()`
- `getObservation()` → `look()`
- `getPossibleActions()` → `get_possible_actions()`
- `getVariations()` → `get_max_variations()`
- `getScore()` → 从 `step()` 的 `infos['score']` 跟踪
- `simplificationStr` 是 `load()` 参数

### 3. Prompt 格式对齐参考论文 ✅
**⚠️ 重要：Prompt 格式已对齐参考论文，不要修改！**

参考论文格式（论文 Section E.2 ScienceWorld Prompts）：
```
Your ScienceWorld task is: {task description}
Prior to this step, you have already taken {step count} step(s). Below are the most recent {history length} observations and the corresponding actions you took: {action history}
You are now at step {current step} and your current observation is: {current observation}
Your valid actions of the current situation are: [{admissible actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags.
Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.
```

**注意**：
- 不要添加 example（保持 zero-shot）
- 不要修改指令文本（参考论文格式已验证）
- System message 可以保留（帮助模型遵循格式）

### 4. Zero-shot 评估 ✅
- 任务: boil
- 1.7B 分数: 0.0（预期，模型不输出 `<action>` 标签）
- 14B 分数: 1.0（variation 2 得 3 分，其他 0 分）
- 速度: ~85秒/episode（新 prompt 格式后提速 4 倍）

### 5. 训练数据生成 ✅
**单任务（boil）数据**：
```bash
python examples/scienceworld_grpo/data_preprocess.py \
    --local_save_dir /workspace/data/scienceworld_real \
    --task_name boil \
    --use_real_env
```

**全部 30 个任务数据**（对齐 TCOD 论文）：
```bash
bash examples/scienceworld_grpo/generate_all_data.sh
```

**数据统计（30 个任务）**：
| 集合 | 数量 |
|------|------|
| Train | 3,592 |
| Dev | 1,796 |
| Test | 1,819 |
| **Total** | **7,207** |

- 保存位置: `/workspace/data/scienceworld_all/`
- 元数据: `task_metadata.json`

### 5. GRPO 训练已跑通 ✅（小规模验证）
```bash
conda activate verl
CUDA_VISIBLE_DEVICES=0 bash examples/scienceworld_grpo/run_real.sh \
    data.train_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4
```

**训练结果（3 steps，14 个训练样本）**:
- 平均奖励: 0.001
- 最大奖励: 0.020
- 优势范围: [-0.5, +1.5]（有学习信号）
- 奖励函数正常工作（timing_s/reward: 19.4s）

**Checkpoint 位置**:
`checkpoints/verl_grpo_scienceworld_real/scienceworld_real_grpo_20260610_0940/global_step_3/`

### 6. 配置对齐 TCOD 论文 ✅
已更新所有配置文件，参数与论文 Table 4/5 对齐。

---

## 待完成工作

### 7. 运行完整训练 ⏳
```bash
# 在服务器上运行
conda activate verl
git pull
CUDA_VISIBLE_DEVICES=1 bash examples/scienceworld_grpo/run_full_training.sh
```

### 8. Checkpoint 格式转换
verl checkpoint 是 `.pt` 格式，需要转换为 HuggingFace 格式才能用 eval 脚本测试

### 9. 训练后评估
- 用训练后的模型在 test set 上评估
- 报告 Average Score (0-100) 和 Success Rate (%)
- 对比 baseline（1.7B: 0.0, 14B: 1.0）

---

## 关键文件
- `eval_zero_shot.py` — 零样本评估脚本
- `data_preprocess.py` — 数据预处理脚本（支持所有 30 个任务）
- `reward_function.py` — 奖励函数（直接调用 ScienceWorld 环境）
- `run_mock.sh` — Mock 训练脚本
- `run_real.sh` — 真实训练脚本
- `run_full_training.sh` — 完整训练脚本（对齐 TCOD 论文）
- `generate_all_data.sh` — 数据生成脚本（所有 30 个任务）
- `config/scienceworld_grpo.yaml` — 主训练配置
- `verl/interactions/scienceworld_interaction.py` — 核心交互类
- `verl/tools/scienceworld_tool.py` — action 工具
- `benchmarks/ScienceWorld/` — ScienceWorld 源码

## 奖励函数设计
**当前方案**: reward function 直接调用 ScienceWorld 环境
1. 从模型输出提取 `<action>` 标签
2. 在 ScienceWorld 环境中执行动作
3. 返回真实环境分数（0-100 归一化到 0-1）

**Fallback**: 如果没有 `<action>` 标签，返回 0 分

## 技术决策
| 决策 | 原因 |
|------|------|
| sync 模式 + reward function 直接调环境 | vllm async 模式不兼容，verl 需要 sglang |
| 单轮生成 + reward function 执行 | 比多轮交互更简单、可靠 |
| 使用 Qwen3-1.7B | 用户指定，显存友好 |
| 全部 30 个任务 | 对齐 TCOD 论文 |
| ScienceWorld 内置划分 | 官方标准，论文可直接引用 |
| GRPO 而非 OPD | verl 框架原生支持 GRPO |

## 注意事项
- ScienceWorld Java 服务器有时会变僵尸进程，需要 `kill -9` 清理
- 命令行参数: `--num_variations` 不是 `--num_variation`
- 用户网络需要 clash 代理访问 GitHub/PyPI
- verl checkpoint 不是 HuggingFace 格式，需要转换
- 多轮交互需要安装 sglang（当前未安装）
- 论文用 OPD，我们用 GRPO，KL coefficient 含义不同（OPD=1.0 vs GRPO=0.001）

## 恢复指令
1. 生成数据: `bash examples/scienceworld_grpo/generate_all_data.sh`
2. 运行训练: `bash examples/scienceworld_grpo/run_full_training.sh`
3. 评估模型: `python examples/scienceworld_grpo/eval_zero_shot.py --model_path <checkpoint_path> --task_name boil --num_variations 30 --max_steps 30`
4. 查看 ScienceWorld 源码: `ls benchmarks/ScienceWorld/`
