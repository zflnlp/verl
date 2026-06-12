# ALFWorld GRPO 项目进度

## 当前状态：配置已对齐 TCOD 论文，准备 Mock 训练

最后更新：2026-06-11

---

## SGLang 多轮训练配置

**问题**：vLLM 不支持真正的多轮交互

**解决方案**：使用 SGLang 0.4.6.post5 进行多轮 rollout

### 新增文件
1. `config/alfworld_sglang_multiturn.yaml` - SGLang 多轮训练配置
2. `run_sglang_multiturn.sh` - SGLang 多轮训练启动脚本

### 关键配置差异
| 配置项 | vLLM (单轮) | SGLang (多轮) |
|--------|------------|---------------|
| rollout.name | vllm | sglang |
| multi_turn.enable | True | True |
| max_assistant_turns | 30 | 30 |
| max_response_length | 512 | 6144 |
| tensor_model_parallel_size | 1 | 2 |
| NGPUS_PER_NODE | 1 | 2 |
| 响应长度 | 501 tokens (截断) | 预期多轮对话历史 |

### 使用方法
```bash
# SGLang 多轮训练 (Mock)
bash examples/alfworld_grpo/run_sglang_multiturn.sh

# SGLang 多轮训练 (Real)
bash examples/alfworld_grpo/run_sglang_real.sh

# vLLM 单轮训练（旧）
bash examples/alfworld_grpo/run_mock.sh
```

### 为什么 max_response_length 需要 6144？

在多轮交互中，`max_response_length` 需要容纳整个对话历史：
- 模型的 thought + action (~100-200 tokens)
- 环境的 observation (~100-200 tokens)
- 重复 30 轮 = 6000-12000 tokens

参考 GSM8K 示例：`max_response_length=$((1024 * 3))` = 3072 tokens

### 为什么需要 interaction_kwargs？

GSM8K 示例的数据格式包含 `interaction_kwargs`：
```python
"extra_info": {
    "interaction_kwargs": {
        "query": question,
        "ground_truth": solution,
    },
},
```

这是传递给 `interaction.start_interaction()` 的参数，用于初始化交互环境。

ALFWorld 需要：
```python
"extra_info": {
    "interaction_kwargs": {
        "ground_truth": task,  # 包含 task_type, goal, game_file
    },
},
```

---

## 参考文献配置（TCOD 论文）

**论文**：TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents (Wang et al., 2026)

### Prompt 格式（Section E.1）✅ 已匹配
```
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task description}
Prior to this step, you have already taken {step count} step(s). Below are the most recent {history length} observations and the corresponding actions you took:
{action history}
You are now at step {current step} and your current observation is: {current observation}
Your admissible actions of the current situation are: [{admissible actions}].
Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```

### 数据划分（Section D.1）
| 划分 | 说明 | 数量 |
|------|------|------|
| `valid_seen` | 训练见过的场景 | - |
| `valid_unseen` | 新房间布局和物体组合（OOD 评估） | - |
| `train_hard` | 教师在 pass@10 采样下失败的任务 | 121 个 |

### 训练超参数（Table 4）
| 参数 | 论文值 | 当前值 | 状态 |
|------|--------|--------|------|
| Algorithm | On-Policy Distillation | GRPO | 不同方法 |
| KL coefficient | 1.0 | 1.0 | ✅ 已对齐 |
| Learning rate | 1×10⁻⁶ | 1×10⁻⁶ | ✅ 一致 |
| Gradient clipping | 1.0 | 1.0 | ✅ 已对齐 |
| Total training steps | 250 | - | ⚠️ 待设置 |
| Batch size | 16 | 32 | ⚠️ 待对齐 |
| Train batch size | 64 | 32 | ⚠️ 待对齐 |
| **Max prompt tokens** | **10,240** | **10,240** | ✅ **已对齐** |
| **Max response tokens** | **512** | **512** | ✅ **已对齐** |
| Temperature (training) | 1.0 | - | ⚠️ 待设置 |
| Temperature (evaluation) | 0.4 | 0.4 | ✅ 已对齐 |
| ALFWorld max steps | 30 | 30 | ✅ 一致 |
| GPUs | 8× H20 (96GB) | 1× GPU | 不同规模 |
| Tensor parallel size | 2 | 1 | 不同规模 |
| GPU memory utilization | 0.7 | 0.7 | ✅ 已对齐 |
| Data type | BFloat16 | BFloat16 | ✅ 一致 |

### 评估超参数（Table 5）
| 参数 | 论文值 | 当前值 | 状态 |
|------|--------|--------|------|
| Maximum tokens | 4,096 | 512 | ⚠️ 待对齐 |
| Temperature | 0.4 | 0.4 | ✅ 已对齐 |
| Top-p | 1.0 | 1.0 | ✅ 一致 |
| Max environment steps | 30 | 30 | ✅ 一致 |
| **History length** | **2 steps** | **2 steps** | ✅ **已对齐** |
| Number of workers | 8 | 1 | 不同规模 |

### ALFWorld 结果（Table 2）
| 模型 | 方法 | Valid Seen | Valid Unseen | Hard |
|------|------|------------|--------------|------|
| Qwen2.5-7B | Teacher (GRPO) | 85.71% | 76.87% | 6.61% |
| Qwen2.5-3B | Zero-Shot | 7.86% | 2.24% | 0.83% |
| Qwen2.5-3B | SFT | 32.14% | 25.37% | 4.96% |
| Qwen2.5-3B | Vanilla OPD | 65.72% | 60.45% | 10.74% |
| Qwen2.5-3B | **TCOD B2F** | **77.86%** | **69.89%** | **13.22%** |
| Qwen2.5-3B | TCOD F2B | 81.43% | 70.90% | 9.92% |
| Qwen2.5-7B | Zero-Shot | 9.29% | 8.96% | 1.65% |
| Qwen2.5-7B | SFT | 54.29% | 48.73% | 8.26% |
| Qwen2.5-7B | Vanilla OPD | 75.37% | 72.14% | 13.22% |
| Qwen2.5-7B | **TCOD B2F** | **86.43%** | **77.61%** | **20.66%** |

### 跨模型结果（Table 3，Qwen3 系列）
| 模型 | 方法 | ALFWorld | WebShop | ScienceWorld |
|------|------|----------|---------|--------------|
| Qwen3-30B | Teacher | 39.57% | 32.84% | 18.42% |
| Qwen3-1.7B | Vanilla OPD | 0.32% | 0.14% | 0.05% |
| Qwen3-1.7B | TCOD B2F (η=2) | 24.55% | 20.54% | 10.82% |
| Qwen3-1.7B | TCOD F2B (η=6) | 23.65% | 21.78% | 11.08% |
| Qwen3-4B | Vanilla OPD | 36.85% | 30.12% | 15.95% |
| Qwen3-4B | TCOD B2F (η=6) | 39.35% | 29.05% | 15.88% |
| Qwen3-4B | TCOD F2B (η=2) | 38.95% | 31.81% | 17.85% |

### 对齐优先级
1. **P0（必须）**：Max prompt tokens (10,240) ✅、Max response tokens (512) ✅、History length (2) ✅
2. **P1（重要）**：评估温度 (0.4) ✅、KL coefficient (1.0) ✅、Gradient clipping (1.0) ✅
3. **P2（可选）**：Batch size、Train batch size、GPU memory utilization ✅

---

## 已完成

### ✅ 代码实现（12 个文件）
| 文件 | 状态 |
|------|------|
| `verl/interactions/alfworld_interaction.py` | ✅ API 修正 + dones tuple bug + 文献 prompt |
| `verl/tools/alfworld_tool.py` | ✅ |
| `examples/alfworld_grpo/config/alfworld_grpo.yaml` | ✅ |
| `examples/alfworld_grpo/config/alfworld_interaction_config.yaml` | ✅ |
| `examples/alfworld_grpo/config/alfworld_tool_config.yaml` | ✅ |
| `examples/alfworld_grpo/data_preprocess.py` | ✅ API 修正 + 文献 prompt |
| `examples/alfworld_grpo/reward_function.py` | ✅ |
| `examples/alfworld_grpo/run_mock.sh` | ✅ |
| `examples/alfworld_grpo/run_real.sh` | ✅ |
| `examples/alfworld_grpo/eval_zero_shot.py` | ✅ API 修正 + dones bug + 文献 prompt + 多任务评估 |
| `examples/alfworld_grpo/setup_alfworld.sh` | ✅ |
| `examples/alfworld_grpo/README.md` | ✅ |

### ✅ Bug 修复记录
1. **`training_method` 缺失**：`init_env()` 要求 `config['general']['training_method'] = 'dagger'`
2. **游戏文件目录结构**：扁平 `task_type-Obj-None-Loc-ID/trial_T*/game.tw-pddl`（递归搜索）
3. **`dones` 类型是 tuple**：`isinstance(dones, list)` 为 False → `bool((False,)) = True` → 1 步结束。修复：`isinstance(dones, (list, tuple))`
4. **tokenizer.json 损坏**（14B）：加了 fallback 到 slow tokenizer
5. **setuptools 82.x 移除 pkg_resources**：裸环境需 `pip install "setuptools<70"`
6. **Prompt 格式不一致**：`data_preprocess.py` 和 `alfworld_interaction.py` 的 prompt 格式不同（bullet points vs comma-separated brackets）。修复：统一为 comma-separated brackets 格式
7. **游戏文件搜索模式错误**：目录名是 `task_type-Obj-None-Loc-ID` 格式，需要使用通配符 `task_type-*` 匹配

### ✅ Prompt 更新（匹配文献）
- System: `"You are an expert agent operating in the ALFRED Embodied Environment."`
- Admissible actions: `[action1, action2, ...]` 逗号分隔 + 方括号
- 简化指令文本，去掉了 task_type

### ✅ ALFWorld 安装
- 源码安装：`cd /workspace/alfworld && pip install -e .` ✅
- 游戏数据：`alfworld-download` → `/workspace/data/alf_data/`（4027 个游戏文件）✅

### ✅ Mock 数据生成
- 100 个 mock 任务（80 train / 20 test）→ `/workspace/data/alfworld/` ✅

### ✅ Zero-shot Baseline（pick_and_place, valid_unseen）
| 模型 | Prompt | Win Rate | WON 步数 | LOST 步数 |
|------|--------|----------|---------|----------|
| Qwen3-1.7B | 旧 prompt | 20% | 12.0 | 30.0 |
| Qwen3-1.7B | 新 prompt（文献版） | **30%** | - | 30.0 |
| Qwen3-14B | 旧 prompt | 50% | 4.0 | 30.0 |
| Qwen3-14B | 新 prompt（文献版） | 40% | 6.8 | 30.0 |

- 单任务结果：`/workspace/data/alfworld_eval_zero_shot_1.7B/`、`alfworld_eval_zero_shot_14B_v2/`
- ⏳ 全任务评估（`--all_task_types`）进行中：`/workspace/data/alfworld_eval_1.7B_all/`

---

## 评估标准

### 文献标准评估协议
- **数据划分**：`valid_unseen`（OOD，134 个游戏，训练未见过的场景）
- **任务类型**：所有 6 种任务类型
- **指标**：每个任务类型 win rate + 总体 win rate
- **评估命令**：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python examples/alfworld_grpo/eval_zero_shot.py \
      --model_path /workspace/models/Qwen3-1.7B/ \
      --all_task_types \
      --num_games 10 --max_steps 30 \
      --alfworld_data_dir /workspace/data/alf_data \
      --output_dir /workspace/data/alfworld_eval_1.7B_all
  ```
- 输出：`overall_summary.json` + 每个任务类型的 `results.json`

### ALFWorld 数据划分
| 划分 | 路径 | 数量 | 说明 |
|------|------|------|------|
| `train` | `json_2.1.1/train` | 3,553 | 训练集（场景 7-30, 207-230, 307-330, 407-430） |
| `valid_seen` | `json_2.1.1/valid_seen` | 140 | 训练见过的场景（ID） |
| `valid_unseen` | `json_2.1.1/valid_unseen` | 134 | 训练未见过的场景（OOD） |

### 6 种任务类型
| ID | 内部名 | 短名 | 说明 |
|----|--------|------|------|
| 1 | `pick_and_place_simple` | `pick_and_place` | 拿起放到位 |
| 2 | `look_at_obj_in_light` | `look_at_obj_in_light` | 在灯光下检查物体 |
| 3 | `pick_clean_then_place_in_recep` | `pick_clean_then_place` | 拿→洗→放 |
| 4 | `pick_heat_then_place_in_recep` | `pick_heat_then_place` | 拿→加热→放 |
| 5 | `pick_cool_then_place_in_recep` | `pick_cool_then_place` | 拿→冷却→放 |
| 6 | `pick_two_obj_and_place` | `pick_two_obj` | 拿两个同类物体 |

---

## 进行中

### ⏳ 全任务 Zero-shot 评估（1.7B）
- 命令：见下方"待做"部分
- 状态：进行中（`/workspace/data/alfworld_eval_1.7B_all/`）

### ⏳ Mock 训练
- 命令：`bash examples/alfworld_grpo/run_mock.sh`
- 状态：GPU 显存不足（公用机器），等空闲后重跑
- 如显存紧张可加 `ROLLOUT_GPU_MEM_UTIL=0.85`

---

## 待做

### 🔲 全任务类型 Zero-shot 评估（1.7B）- 进行中
```bash
CUDA_VISIBLE_DEVICES=0 python examples/alfworld_grpo/eval_zero_shot.py \
    --model_path /workspace/models/Qwen3-1.7B/ \
    --all_task_types \
    --num_games 10 --max_steps 30 \
    --alfworld_data_dir /workspace/data/alf_data \
    --output_dir /workspace/data/alfworld_eval_1.7B_all
```

### 🔲 Mock 训练
```bash
bash examples/alfworld_grpo/run_mock.sh
```
- 重新生成 mock 数据（使用修复后的 prompt 格式）
- 运行 mock 训练

### 🔲 Mock 训练后全任务评估
GRPO 训练完成后，用同样的 `--all_task_types` 评估，对比 baseline win rate 提升。

### 🔲 Real 训练
1. 生成真实数据（需要 alfworld 环境）
2. 跑真实训练（需要 alfworld + verl 统一环境）

---

## 真实数据训练

### 数据生成

```bash
# 生成单任务数据（pick_and_place）
python examples/alfworld_grpo/data_preprocess.py \
    --local_save_dir /workspace/data/alfworld_real \
    --use_real_env \
    --num_games 100 \
    --game_files_dir /workspace/data/alf_data

# 生成全部 6 种任务类型数据（推荐）
bash examples/alfworld_grpo/generate_all_tasks.sh
```

### 训练步骤

```bash
# 1. 生成真实数据
python examples/alfworld_grpo/data_preprocess.py \
    --local_save_dir /workspace/data/alfworld_real \
    --use_real_env \
    --num_games 100 \
    --game_files_dir /workspace/data/alf_data

# 2. 运行真实训练（SGLang 多轮）
CUDA_VISIBLE_DEVICES=2,7 bash examples/alfworld_grpo/run_sglang_real.sh
```

### 论文配置对齐

| 配置项 | 论文值 | 我们的值 | 状态 |
|--------|--------|----------|------|
| 数据划分 | train/valid_seen/valid_unseen | train/valid_seen/valid_unseen | ✅ 一致 |
| 任务类型 | 6 种 | 6 种 | ✅ 一致 |
| Max steps | 30 | 30 | ✅ 一致 |
| Max prompt tokens | 10,240 | 10,240 | ✅ 已对齐 |
| Max response tokens | 512 | 6,144 | ✅ 已对齐（多轮） |
| KL coefficient | 1.0 | 1.0 | ✅ 已对齐 |
| Learning rate | 1e-6 | 1e-6 | ✅ 一致 |

---

## 开发工作流程

**重要**：实验在服务器 Docker 环境中运行，不在本地 Mac 上！

1. **本地 Mac**：修改代码、调试、提交到 GitHub
2. **GitHub**：代码仓库（`webshop-grpo-v0.4.1` 分支）
3. **服务器 Docker**：拉取代码、生成数据、运行训练

```bash
# 服务器上拉取最新代码
cd /workspace/verl
git pull origin webshop-grpo-v0.4.1

# 生成数据
python examples/alfworld_grpo/data_preprocess.py \
    --local_save_dir /workspace/data/alfworld \
    --num_tasks 100 --seed 42

# 运行训练
bash examples/alfworld_grpo/run_mock.sh
```

---

## 环境信息

| 项目 | 路径/版本 |
|------|----------|
| verl 源码 | `/workspace/verl` |
| ALFWorld 源码 | `/workspace/alfworld` |
| 游戏数据 | `/workspace/data/alf_data/` |
| Mock 数据 | `/workspace/data/alfworld/` |
| 模型 1.7B | `/workspace/models/Qwen3-1.7B` |
| 模型 14B | `/workspace/models/Qwen3-14B/` |
| conda 环境 | `alfworld`（Python 3.10.20） |
| 裸环境 | 含 vllm/pytorch/ray（verl 训练用） |
| GPU | 公用机器，需等空闲 |

---

## 已知问题 & 备注

1. **setuptools 82.x 移除 pkg_resources**：裸环境需 `pip install "setuptools<70"`
2. **Mock 训练在裸环境跑**（不需 alfworld），Real 训练需要统一环境
3. **ALFWORLD_DATA=/workspace/data/alf_data**
4. **评估应使用 `--all_task_types` 跑全部 6 种任务**，匹配文献标准
