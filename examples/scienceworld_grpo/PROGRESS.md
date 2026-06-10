# ScienceWorld GRPO 项目进度

## 项目概述
基于 verl 框架为 ScienceWorld benchmark 构建 GRPO 训练管线，训练 Qwen3-1.7B 模型完成科学实验任务。

## 当前状态
**阶段**: 数据划分完成，准备运行 GRPO 训练

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

参考论文格式（使用 Qwen3 系列模型，格式已验证有效）：
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
使用 ScienceWorld 内置 train/dev/test 划分：
```bash
python examples/scienceworld_grpo/data_preprocess.py \
    --local_save_dir /workspace/data/scienceworld_real \
    --task_name boil \
    --use_real_env
```

**数据划分（boil 任务 30 个 variations）**：
| 集合 | 数量 | 比例 | 用途 |
|------|------|------|------|
| Train | 14 | 46.7% | GRPO 训练 |
| Dev | 7 | 23.3% | 训练中验证 |
| Test | 9 | 30.0% | 最终评估 |

- 保存位置: `/workspace/data/scienceworld_real/`
- 元数据: `task_metadata.json`（包含每个集合的 variation 索引）

### 6. GRPO 训练已跑通 ✅
```bash
conda activate verl
CUDA_VISIBLE_DEVICES=0 bash examples/scienceworld_grpo/run_real.sh \
    data.train_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4
```

**训练结果（6 steps）**:
- 平均奖励: 0.002（很低，1.7B 模型学习能力有限）
- 最大奖励: 0.020
- 优势范围: [-0.866, +0.866]（有学习信号）
- 奖励函数正常工作（timing_s/reward: 19.7s）

**关键发现**:
- vllm 不支持多轮交互的 async 模式（verl 需要 sglang）
- 改为 sync 模式，reward function 直接调用 ScienceWorld 环境
- 这是主流做法（DeepSeek-R1、Search-R1 等都用这种方式）

**Checkpoint 位置**:
`checkpoints/verl_grpo_scienceworld_real/scienceworld_real_grpo_20260609_0941/global_step_6/`

## 待完成工作

### 7. 重新训练（使用正确的 train/dev/test 划分）⏳
```bash
conda activate verl
CUDA_VISIBLE_DEVICES=0 bash examples/scienceworld_grpo/run_real.sh \
    data.train_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4
```

### 8. Checkpoint 格式转换
verl checkpoint 是 `.pt` 格式，需要转换为 HuggingFace 格式才能用 eval 脚本测试

### 9. 训练后评估
- 用训练后的模型在 test set（9 个 variations）上评估
- 报告 Average Score (0-100) 和 Success Rate (%)
- 对比 baseline（1.7B: 0.0, 14B: 1.0）

## 关键文件
- `eval_zero_shot.py` — 零样本评估脚本
- `data_preprocess.py` — 数据预处理脚本（使用 ScienceWorld 内置划分）
- `reward_function.py` — 奖励函数（直接调用 ScienceWorld 环境）
- `run_mock.sh` — Mock 训练脚本
- `run_real.sh` — 真实训练脚本
- `verl/interactions/scienceworld_interaction.py` — 核心交互类
- `verl/tools/scienceworld_tool.py` — action 工具

## 奖励函数设计
**当前方案**: reward function 直接调用 ScienceWorld 环境
1. 从模型输出提取 `<action>` 标签
2. 在 ScienceWorld 环境中执行动作
3. 返回真实环境分数（0-100 归一化到 0-1）

**Fallback**: 如果没有 `<action>` 标签，返回 0 分

## 训练参数
| 参数 | 值 |
|------|-----|
| rollout_n | 4 |
| actor_lr | 1e-6 |
| kl_loss_coef | 0.001 |
| train_batch_size | 4 |
| gpu_memory_utilization | 0.6 |

## 技术决策
| 决策 | 原因 |
|------|------|
| sync 模式 + reward function 直接调环境 | vllm async 模式不兼容，verl 需要 sglang |
| 单轮生成 + reward function 执行 | 比多轮交互更简单、可靠 |
| 使用 Qwen3-1.7B | 用户指定，显存友好 |
| boil 任务 | 简单经典，适合验证管线 |
| ScienceWorld 内置划分 | 官方标准，论文可直接引用 |

## 注意事项
- ScienceWorld Java 服务器有时会变僵尸进程，需要 `kill -9` 清理
- 命令行参数: `--num_variations` 不是 `--num_variation`
- 用户网络需要 clash 代理访问 GitHub/PyPI
- verl checkpoint 不是 HuggingFace 格式，需要转换
- 多轮交互需要安装 sglang（当前未安装）

## 恢复指令
1. 运行训练: `CUDA_VISIBLE_DEVICES=0 bash examples/scienceworld_grpo/run_real.sh data.train_batch_size=4 actor_rollout_ref.actor.ppo_mini_batch_size=4`
2. 评估模型: `python examples/scienceworld_grpo/eval_zero_shot.py --model_path <checkpoint_path> --task_name boil --num_variations 9 --max_steps 10`
