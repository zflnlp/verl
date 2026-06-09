# ScienceWorld GRPO 项目进度

## 项目概述
基于 verl 框架为 ScienceWorld benchmark 构建 GRPO 训练管线，训练 Qwen3-1.7B 模型完成科学实验任务。

## 当前状态
**阶段**: 训练数据已生成，准备运行 GRPO 训练

## 已完成工作

### 1. 环境搭建 ✅
- verl 0.4.1 + vllm≤0.8.5
- 服务器: hgx18 (8x H100 80GB)
- 模型: `/workspace/models/Qwen3-1.7B`
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
参考论文格式：
```
Your ScienceWorld task is: {task description}
Prior to this step, you have already taken {step count} step(s).
Below are the most recent {history length} observations and the corresponding actions you took:
{action history}
You are now at step {current step} and your current observation is:
{current observation}
Your valid actions of the current situation are: [{admissible actions}].
Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.
```

### 4. Zero-shot 评估 ✅
- 任务: boil
- 分数: 0.0（预期，1.7B 未经训练）
- 速度: ~85秒/episode（新 prompt 格式后提速 4 倍）
- 问题: 模型不输出 `<action>` 标签，需要 GRPO 训练

### 5. 训练数据生成 ✅
```bash
python examples/scienceworld_grpo/data_preprocess.py \
    --local_save_dir /workspace/data/scienceworld_real \
    --task_name boil \
    --num_variations 30 \
    --use_real_env
```
- 训练集: 24 个任务
- 测试集: 6 个任务
- 保存位置: `/workspace/data/scienceworld_real/`

## 待完成工作

### 6. 运行 GRPO 训练 ⏳
```bash
conda activate verl
bash examples/scienceworld_grpo/run_real.sh
```

### 7. 训练后评估
- 用训练后的模型重新跑 zero-shot 评估
- 对比 baseline（0.0）和训练后的分数

## 关键文件
- `eval_zero_shot.py` — 零样本评估脚本
- `data_preprocess.py` — 数据预处理脚本
- `reward_function.py` — 奖励函数
- `run_mock.sh` — Mock 训练脚本
- `run_real.sh` — 真实训练脚本
- `verl/interactions/scienceworld_interaction.py` — 核心交互类
- `verl/tools/scienceworld_tool.py` — action 工具

## 奖励函数设计
- **主要**: 环境奖励（ScienceWorld 返回的 0-100 分，归一化到 0-1）
- **Fallback**: 启发式奖励（动作多样性、任务相关性、推理质量）

## 训练参数
| 参数 | 值 |
|------|-----|
| rollout_n | 4 |
| max_steps | 25 |
| actor_lr | 1e-6 |
| kl_loss_coef | 0.001 |
| train_batch_size | 32 |

## 注意事项
- ScienceWorld Java 服务器有时会变僵尸进程，需要 `kill -9` 清理
- 命令行参数: `--num_variations` 不是 `--num_variation`
- 用户网络需要 clash 代理访问 GitHub/PyPI

## 恢复指令
运行 `bash examples/scienceworld_grpo/run_real.sh` 开始 GRPO 训练。
