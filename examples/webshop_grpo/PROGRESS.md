# WebShop GRPO 项目进度

> 最后更新: 2026-06-09 20:30

## 项目概述

使用 GRPO（Group Relative Policy Optimization）算法训练 Qwen3-1.7B 模型成为 WebShop 购物 Agent。

- **分支**: `webshop-grpo-v0.4.1`（基于 verl v0.4.1，兼容 vllm 0.8.5）
- **服务器**: hgx18 (10.24.10.118) — 8x H100 80GB
- **模型**: `/workspace/models/Qwen3-1.7B`
- **数据**: `/workspace/data/webshop_test/`

---

## 完成进度

| 阶段 | 状态 | 日期 | 备注 |
|------|------|------|------|
| 代码框架搭建 | ✅ 完成 | 2026-06-07 | interaction/tool/reward/data_preprocess |
| Mock 训练跑通 | ✅ 完成 | 2026-06-07 | avg reward 0.436，单轮+多轮均验证 |
| WebShop 真实环境安装 | ✅ 完成 | 2026-06-08 | conda env `webshop`，Python 3.10 |
| 零样本评估流程跑通 | ✅ 完成 | 2026-06-08 | Qwen3-1.7B 上运行 50 episodes |
| Prompt & Action Format 对齐论文 | ✅ 完成 | 2026-06-09 | 统一为 search[query] + click[button] |
| 零样本评估 Qwen3-1.7B（更新后） | ✅ 完成 | 2026-06-09 | 50 episodes, avg reward = 0.000 |
| 零样本评估 Qwen3-14B（对比） | 🔄 运行中 | 2026-06-09 | 15 episodes, num_products=200, 排查脚本问题 |
| 真实 GRPO 训练 | ⏳ 待开始 | — | 需要 WebShop 服务器运行 |
| 训练后评估 & 对比 | ⏳ 待开始 | — | 需要训练完成 |

---

## 零样本评估结果

### Qwen3-1.7B（50 episodes, num_products=1000）

| 指标 | 值 |
|------|-----|
| Average reward | 0.000 |
| Max reward | 0.000 |
| Min reward | 0.000 |
| Non-zero reward rate | 0.0% |
| Success rate (reward > 0.5) | 0.0% |
| 耗时 | ~1h40m (~120s/episode) |
| 结果文件 | `results/webshop_zero_shot/zero_shot_results.json` |

**分析**: 1.7B 小模型完全无法完成购物任务。需要用更大模型（14B）验证是模型能力问题还是脚本问题。

### Qwen3-14B（进行中）

```bash
python examples/webshop_grpo/evaluate_zero_shot.py \
    --model_path /workspace/models/Qwen3-14B/ \
    --num_episodes 15 \
    --max_steps 15 \
    --num_products 200 \
    --output_dir results/webshop_zero_shot_14B
```

- 目的：排查零 reward 是模型能力不足还是脚本逻辑问题
- 减少 num_products (1000→200) 和 num_episodes (50→15) 加速测试
- 如果 14B 能拿到非零 reward → 脚本正常，1.7B 能力不够
- 如果 14B 也全零 → 需要检查脚本逻辑

---

## Prompt & Action Format（论文规范）

WebShop 只有两种 action type：

```
search[<query>]    — 搜索产品（仅在搜索栏存在时可用）
click[<button name>] — 点击交互元素（产品链接、筛选按钮、分页等）
```

购买通过 `click[buy]` 实现，不是独立 action。

Prompt 模板：
```
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task description}.

Prior to this step, you have already taken {step count} step(s).
Below are the most recent {history length} observations and the corresponding actions you took:
{action history}

You are now at step {current step} and your current observation is:
{current observation}.

Your admissible actions of the current situation are:
[  {available actions} ]

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation,
then think carefully which admissible action best advances the shopping goal.
This reasoning process MUST be enclosed within <thought> tags.
Once you've finished your reasoning, you should choose an admissible action
for current step and present it within <action> </action> tags.
```

---

## 训练配置

| 参数 | Mock 模式 | 真实模式 |
|------|----------|----------|
| batch_size | 32 | 64 |
| rollout_n | 4 | 8 |
| epochs | 1 | 3 |
| max_response_length | 2048 | 3072 |
| rollout engine | vllm | vllm |
| multi_turn | 最多 15 轮 | 最多 15 轮 |

---

## 文件结构

```
examples/webshop_grpo/
├── config/
│   ├── webshop_grpo.yaml                    # 主训练配置（继承 ppo_trainer）
│   ├── webshop_interaction_config.yaml      # Mock 交互配置
│   └── webshop_tool_config.yaml             # Mock 工具配置（search + click）
├── data_preprocess.py                       # 数据预处理（生成 parquet）
├── evaluate_zero_shot.py                    # 零样本评估脚本
├── reward_function.py                       # 奖励函数
├── test_webshop.py                          # 环境测试脚本
├── run_mock.sh                              # Mock 训练脚本
├── run_real.sh                              # 真实训练脚本
├── setup_webshop_py310.sh                   # 环境安装脚本
├── setup_mock_pyserini.sh                   # Mock pyserini 安装
├── PROGRESS.md                              # 本文件
└── README.md                                # 说明文档

verl/interactions/
└── webshop_interaction.py                   # WebShop 交互类

verl/tools/
└── webshop_tool.py                          # WebShop 工具类（search/click）
```

---

## 服务器操作指南

### 拉取最新代码
```bash
cd /workspace/verl && git pull origin webshop-grpo-v0.4.1
```

### 运行零样本评估
```bash
conda activate webshop

# Qwen3-1.7B（完整评估）
python examples/webshop_grpo/evaluate_zero_shot.py \
    --model_path /workspace/models/Qwen3-1.7B \
    --num_episodes 50 \
    --max_steps 15 \
    --output_dir results/webshop_zero_shot

# Qwen3-14B（快速验证）
python examples/webshop_grpo/evaluate_zero_shot.py \
    --model_path /workspace/models/Qwen3-14B/ \
    --num_episodes 15 \
    --max_steps 15 \
    --num_products 200 \
    --output_dir results/webshop_zero_shot_14B
```

### 运行真实 GRPO 训练
```bash
conda activate webshop
# 先启动 WebShop 服务器（另一个终端）
# cd /workspace/WebShop && python run_server.py

# 然后运行训练
bash examples/webshop_grpo/run_real.sh
```

---

## 已知问题

| 问题 | 状态 | 备注 |
|------|------|------|
| transformers 版本需 >=4.51.0 才支持 Qwen3 | 已解决 | webshop conda env 中已安装 |
| pyserini 与 PyTorch 版本冲突 | 已解决 | 使用 mock pyserini 替代 |
| werkzeug 与 Flask 不兼容 | 已解决 | 降级到 werkzeug==2.3.7 |
| Mock 模式奖励不确定性 | 已知 | `random.uniform` 未 seed，不影响真实训练 |

---

## 下一步

1. 等 Qwen3-14B 零样本评估完成，判断是脚本问题还是模型能力问题
2. 如果 14B 也全零 → 检查 evaluate_zero_shot.py 的 action 解析和 env.step() 逻辑
3. 如果 14B 有非零 reward → 脚本正常，继续 GRPO 训练
4. 启动 WebShop 服务器，运行真实 GRPO 训练
5. 训练后评估，对比零样本基线
