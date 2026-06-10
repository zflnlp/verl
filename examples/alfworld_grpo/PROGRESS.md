# ALFWorld GRPO 项目进度

## 当前状态：Zero-shot baseline 已建立，等待 GPU 空闲跑 Mock 训练

最后更新：2026-06-09

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

- 结果路径：`/workspace/data/alfworld_eval_zero_shot_1.7B/`、`alfworld_eval_zero_shot_14B_v2/`

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

### ⏳ Mock 训练
- 命令：`bash examples/alfworld_grpo/run_mock.sh`
- 状态：GPU 显存不足（公用机器），等空闲后重跑
- 如显存紧张可加 `ROLLOUT_GPU_MEM_UTIL=0.85`

---

## 待做

### 🔲 全任务类型 Zero-shot 评估（1.7B）
```bash
CUDA_VISIBLE_DEVICES=0 python examples/alfworld_grpo/eval_zero_shot.py \
    --model_path /workspace/models/Qwen3-1.7B/ \
    --all_task_types \
    --num_games 10 --max_steps 30 \
    --alfworld_data_dir /workspace/data/alf_data \
    --output_dir /workspace/data/alfworld_eval_1.7B_all
```

### 🔲 Mock 训练后全任务评估
GRPO 训练完成后，用同样的 `--all_task_types` 评估，对比 baseline win rate 提升。

### 🔲 Real 训练
1. 生成真实数据（需要 alfworld 环境）
2. 跑真实训练（需要 alfworld + verl 统一环境）

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
