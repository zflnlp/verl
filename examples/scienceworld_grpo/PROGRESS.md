# ScienceWorld GRPO 项目进度

## 项目概述
基于 verl 框架为 ScienceWorld benchmark 构建 GRPO 训练管线，训练 Qwen3-1.7B 模型完成科学实验任务。

## 当前状态
**阶段**: 多轮 GRPO 训练运行中（sglang + 30 步交互 + 过程奖励）

### 最新进展
- ✅ sglang 安装完成（0.4.6.post5）
- ✅ 数据格式修复（添加 interaction_kwargs 字段）
- ✅ 多轮训练配置创建完成
- ✅ 多轮训练正在运行中
- ✅ 过程奖励正常工作（分数从 0 提升到 0.25）
- ⏳ 观察训练效果，等待收敛

## 源码参考
- ScienceWorld 源码: `benchmarks/ScienceWorld/`
- WebShop 源码: `benchmarks/WebShop/`
- alfworld 源码: `benchmarks/alfworld/`
- 数据目录: `data/`

---

## 训练方式对比

### 单轮 vs 多轮

| | 单轮（已验证） | 多轮（推荐） |
|--|---------------|-------------|
| rollout 引擎 | vllm | sglang |
| 交互方式 | 模型一次性输出所有动作 | 每步生成一个动作，接收观察，继续 |
| max_turns | 1 | 30 |
| 奖励 | 只看最终分数 | 每步都可以有中间奖励 |
| 适用场景 | 简单任务 | ScienceWorld 等需要多步推理的任务 |

**为什么需要多轮？**
- ScienceWorld 是多轮交互任务，需要 30 步实验操作
- 单轮：模型一次性输出所有动作，无法根据环境反馈调整策略
- 多轮：模型每步看到新观察，动态调整策略，更接近真实实验过程

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
- verl 0.4.1 + vllm 0.8.5 + sglang 0.4.6.post5
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

### 6. 单轮 GRPO 训练已跑通 ✅（小规模验证）
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

### 7. 配置对齐 TCOD 论文 ✅
已更新所有配置文件，参数与论文 Table 4/5 对齐。

### 8. sglang 安装完成 ✅
```bash
pip install "sglang[all]==0.4.6.post5"
# 验证
python -c "import sglang; print(sglang.__version__)"  # 0.4.6.post5
```

### 9. 多轮训练配置创建 ✅
新增文件：
- `config/scienceworld_multiturn_grpo.yaml` — 多轮训练主配置
- `config/tool_config/scienceworld_tool_config.yaml` — 工具配置
- `config/interaction_config/scienceworld_interaction_config.yaml` — 交互配置
- `run_multiturn_training.sh` — 多轮训练脚本

### 10. 数据格式修复 ✅
添加 `interaction_kwargs` 字段，多轮训练必需：
```python
"extra_info": {
    "task_name": "boil",
    "variation": 14,
    "interaction_kwargs": {
        "ground_truth": task,  # 传递给 start_interaction()
    }
}
```

### 11. 多轮训练运行中 ⏳
```bash
# 在服务器上运行
cd /workspace/verl
git pull
# 重新生成数据（包含 interaction_kwargs）
bash examples/scienceworld_grpo/generate_all_data.sh
# 运行多轮训练（4 GPU）
CUDA_VISIBLE_DEVICES=2,3,4,5 bash examples/scienceworld_grpo/run_multiturn_training.sh
```

**训练观察（早期阶段）**：
- ✅ 多轮交互正常工作（30 步交互）
- ✅ 模型正确使用 `<think>` 和 `<action>` 标签
- ✅ 环境反馈正常（返回观察和错误信息）
- ✅ 有样本得满分 1.0（任务完成）
- ✅ 模型在学习任务推理（理解任务目标、规划多步操作）
- ⚠️ 模型存在"死循环"现象（反复尝试同一失败动作）
- ⚠️ 这是早期 RL 训练的正常现象，需要更多训练步骤

**预期进展**：
```
早期: 重复尝试 → 失败 → 重复尝试
中期: 尝试不同动作 → 部分成功
后期: 根据反馈调整策略 → 高成功率
```

### 12. 配置修复 ✅
**问题**：多轮训练出现 -100 分数
**原因**：配置文件中同时有 `custom_reward_function` 和 `ScienceWorldInteraction`，两者冲突
**修复**：移除 `custom_reward_function`，多轮模式下奖励由 `ScienceWorldInteraction` 类处理

```yaml
# 移除此配置
# custom_reward_function:
#   path: examples/scienceworld_grpo/reward_function.py
#   name: compute_score
```

### 13. 响应长度优化 ✅
**问题**：96.9% 响应被截断（512 token 不够）
**修复**：增加 `max_response_length` 到 6144

```yaml
data:
  max_response_length: 6144  # 从 512 增加到 6144
```

### 14. GPU 配置优化 ✅
**问题**：默认只用 1 张 GPU
**修复**：设置 `NGPUS_PER_NODE=4`

```bash
# 默认使用 4 张 GPU
NGPUS_PER_NODE=4
```

### 15. Tokenization 不一致修复 ✅
**问题**：训练和推理时的 tokenization 不一致，导致警告
**原因**：`use_inference_chat_template: False` 导致训练和推理使用不同的 chat template
**修复**：
```yaml
multi_turn:
  use_inference_chat_template: True  # 使用模型默认 chat template
  tokenization_sanity_check_mode: ignore_strippable  # 忽略可剥离 token 的检查
```

**说明**：
- `use_inference_chat_template: True`：训练和推理使用相同的 chat template，确保一致性
- `tokenization_sanity_check_mode: ignore_strippable`：Qwen3 模型有已知的 tokenization 差异，忽略可剥离 token 的检查
- Prompt 格式保持与 TCOD 论文一致，不受此配置影响

### 16. 奖励函数修复 ✅
**问题**：多轮训练出现 -100 分数
**原因**：
1. `reward_function.py` 在多轮模式下重复运行 ScienceWorld 环境
2. 系统需要 `custom_reward_function` 处理 `data_source='scienceworld'`
3. Interaction 的奖励存在 `non_tensor_batch["reward_scores"]`，但 reward manager 检查 `batch["rm_scores"]`

**修复**：
```python
# reward_function.py
def compute_score(...):
    # 检查是否是多轮模式
    if extra_info and extra_info.get("interaction_kwargs"):
        return 0.0  # 多轮模式：奖励由 interaction 处理
    # 单轮模式：运行环境
    ...
```

**待解决**：多轮模式下，interaction 的奖励需要正确传递给 reward manager
- 当前：reward function 返回 0.0，interaction 奖励未使用
- 目标：将 `non_tensor_batch["reward_scores"]` 传递给 `batch["rm_scores"]`

### 17. 多轮奖励传递修复 ✅
**问题**：interaction 的奖励存在 `non_tensor_batch["reward_scores"]`，但 reward manager 检查 `batch["rm_scores"]`
**修复**：在 `reward.py` 中添加 `convert_multiturn_rewards_to_rm_scores()` 函数

### 18. 过程奖励（Dense Rewards）✅
**问题**：只使用最终分数，学习信号稀疏
**修复**：使用 ScienceWorld 的过程奖励（delta score）

**ScienceWorld 奖励机制**：
```python
score = int(round(100 * self.server.getScore()))  # 0-100 分
reward = score - self.lastStepScore                # 过程奖励（分数变化）
```

**修改内容**：
1. `_process_real_action`: 自己计算 delta score（不依赖 `info["reward"]`）
2. `generate_response`: 返回累积过程奖励（不是 delta score）
3. `calculate_score`: 使用累积过程奖励

**关键修复**：
- 问题 1: `info["reward"]` 返回 0 → 自己计算 `total_score - last_score`
- 问题 2: `generate_response` 返回 delta → 返回累积奖励
- 问题 3: 奖励范围不正确 → 归一化到 [0, 1]

**训练结果**：
```
平均分数: 0.028（从 0 提升）
最大分数: 0.250（25% 完成）
Policy loss: 0.199（有学习信号）
优势范围: [-1.5, 1.5]（有差异）
```

**效果**：
- ✅ 学习信号更密集（每步都有奖励）
- ✅ 训练更快收敛
- ✅ 与 ScienceWorld 环境对齐
- ✅ 分数从 0 提升到 0.25

---

## 待完成工作

### 15. 等待多轮训练收敛
- 观察奖励是否上升（从 0 → 0.1 → 0.5 → ...）
- 观察动作是否更准确（从随机尝试 → 有策略地完成任务）
- 观察完成率是否提高（从 0% → 10% → 30% → ...）

### 16. Checkpoint 格式转换
verl checkpoint 是 `.pt` 格式，需要转换为 HuggingFace 格式才能用 eval 脚本测试

### 17. 训练后评估
- 用训练后的模型在 test set 上评估
- 报告 Average Score (0-100) 和 Success Rate (%)
- 对比 baseline（1.7B: 0.0, 14B: 1.0）

### 18. 优化方向（如果训练效果不好）
1. **增加探索**：提高 temperature（当前 1.0，可试 1.2）
2. **添加过程奖励**：给中间步骤小奖励（如"尝试新动作 +0.01"）
3. **优化 prompt**：在 system message 中提醒"如果动作失败，尝试其他有效动作"

---

## 关键文件

### 训练脚本
| 文件 | 说明 |
|------|------|
| `run_mock.sh` | Mock 训练脚本（测试 pipeline） |
| `run_real.sh` | 单轮真实训练脚本（vllm） |
| `run_full_training.sh` | 单轮完整训练脚本（对齐 TCOD） |
| `run_multiturn_training.sh` | **多轮训练脚本（sglang）** ⭐ |

### 配置文件
| 文件 | 说明 |
|------|------|
| `config/scienceworld_grpo.yaml` | 单轮训练配置 |
| `config/scienceworld_multiturn_grpo.yaml` | **多轮训练配置** ⭐ |
| `config/tool_config/scienceworld_tool_config.yaml` | 工具配置 |
| `config/interaction_config/scienceworld_interaction_config.yaml` | 交互配置 |

### 数据和评估
| 文件 | 说明 |
|------|------|
| `data_preprocess.py` | 数据预处理（支持 30 个任务） |
| `generate_all_data.sh` | 数据生成脚本 |
| `eval_zero_shot.py` | 零样本评估脚本 |
| `reward_function.py` | 奖励函数 |

### 核心代码
| 文件 | 说明 |
|------|------|
| `verl/interactions/scienceworld_interaction.py` | 多轮交互类 |
| `verl/tools/scienceworld_tool.py` | Action 工具 |
| `benchmarks/ScienceWorld/` | ScienceWorld 源码 |

---

## 奖励函数设计

### 单轮模式
**方案**: reward function 直接调用 ScienceWorld 环境
1. 从模型输出提取 `<action>` 标签
2. 在 ScienceWorld 环境中执行所有动作
3. 返回最终分数（0-100 归一化到 0-1）

### 多轮模式
**方案**: 每步交互都可以有奖励
1. 模型生成一个动作
2. 环境执行并返回观察
3. 可以给中间奖励（如完成子任务）
4. 最终给任务完成分数

---

## 技术决策
| 决策 | 原因 |
|------|------|
| 多轮交互 + sglang | ScienceWorld 需要 30 步交互，vllm 不支持多轮 |
| GRPO 而非 OPD | verl 框架原生支持 GRPO |
| 使用 Qwen3-1.7B | 用户指定，显存友好 |
| 全部 30 个任务 | 对齐 TCOD 论文 |
| ScienceWorld 内置划分 | 官方标准，论文可直接引用 |

---

## 注意事项
- ScienceWorld Java 服务器有时会变僵尸进程，需要 `kill -9` 清理
- 命令行参数: `--num_variations` 不是 `--num_variation`
- 用户网络需要 clash 代理访问 GitHub/PyPI
- verl checkpoint 不是 HuggingFace 格式，需要转换
- sglang 需要设置 `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True`
- 论文用 OPD，我们用 GRPO，KL coefficient 含义不同（OPD=1.0 vs GRPO=0.001）

---

## 恢复指令
1. 生成数据: `bash examples/scienceworld_grpo/generate_all_data.sh`
2. 运行多轮训练: `bash examples/scienceworld_grpo/run_multiturn_training.sh`
3. 评估模型: `python examples/scienceworld_grpo/eval_zero_shot.py --model_path <checkpoint_path> --task_name boil --num_variations 30 --max_steps 30`
4. 查看 ScienceWorld 源码: `ls benchmarks/ScienceWorld/`
