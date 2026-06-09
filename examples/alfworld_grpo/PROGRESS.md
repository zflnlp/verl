# ALFWorld GRPO 项目进度

## 当前状态：Mock 训练就绪，等待 GPU 空闲

最后更新：2026-06-09

---

## 已完成

### ✅ 代码实现（12 个文件）
| 文件 | 状态 |
|------|------|
| `verl/interactions/alfworld_interaction.py` | ✅ 已修正 API |
| `verl/tools/alfworld_tool.py` | ✅ |
| `examples/alfworld_grpo/config/alfworld_grpo.yaml` | ✅ |
| `examples/alfworld_grpo/config/alfworld_interaction_config.yaml` | ✅ |
| `examples/alfworld_grpo/config/alfworld_tool_config.yaml` | ✅ |
| `examples/alfworld_grpo/data_preprocess.py` | ✅ 已修正 API |
| `examples/alfworld_grpo/reward_function.py` | ✅ |
| `examples/alfworld_grpo/run_mock.sh` | ✅ |
| `examples/alfworld_grpo/run_real.sh` | ✅ |
| `examples/alfworld_grpo/eval_zero_shot.py` | ✅ 已修正 API |
| `examples/alfworld_grpo/setup_alfworld.sh` | ✅ |
| `examples/alfworld_grpo/README.md` | ✅ |

### ✅ API 验证（基于真实 ALFWorld 源码修正）
- 环境初始化：`get_environment(env_type)(config_dict)` ✅
- `init_env()` 需要 `config['general']['training_method'] = 'dagger'` ✅
- `reset()` 不接受参数，需通过修改 `alfred_env.game_files` 指定游戏 ✅
- `step()` 返回 `(obs, scores, dones, infos)` ✅
- `infos['admissible_commands'][0]` 访问正确 ✅
- 任务类型内部名映射（`pick_and_place` → `pick_and_place_simple`）✅

### ✅ ALFWorld 安装（服务器 Docker）
- 源码安装：`cd /workspace/alfworld && pip install -e .` ✅
- 游戏文件：`alfworld-download` → `/workspace/data/alf_data/`（4027 个游戏文件）✅
- API 测试通过：env creation, reset, step 全部正常 ✅

### ✅ Mock 数据生成
- 100 个 mock 任务（80 train / 20 test）→ `/workspace/data/alfworld/` ✅
- 需要 `pip install pyarrow` 才能写 parquet ✅

### ✅ GitHub 推送
- 分支：`webshop-grpo-v0.4.1`
- 最新 commit：`bedbbf4e`（Add missing training_method to ALFWorld config）

---

## 进行中

### ⏳ Mock 训练
- 命令：`bash examples/alfworld_grpo/run_mock.sh`
- 状态：**首次运行报 GPU 显存不足**（公用机器卡被占用）
- 等 GPU 空闲后重跑，如显存紧张可加 `ROLLOUT_GPU_MEM_UTIL=0.85`

### ⏳ Zero-shot 评估
- 命令：
  ```bash
  conda activate alfworld
  python examples/alfworld_grpo/eval_zero_shot.py \
      --model_path /workspace/models/Qwen3-1.7B \
      --task_type pick_and_place \
      --num_games 10 \
      --max_steps 30 \
      --alfworld_data_dir /workspace/data/alf_data \
      --output_dir /workspace/data/alfworld_eval_zero_shot
  ```
- 状态：未运行（需要 alfworld 环境 + GPU）

---

## 待做

### 🔲 Real 训练
1. 生成真实数据：
   ```bash
   conda activate alfworld
   python examples/alfworld_grpo/data_preprocess.py \
       --local_save_dir /workspace/data/alfworld_real \
       --use_real_env --task_type pick_and_place --num_games 10 \
       --game_files_dir /workspace/data/alf_data
   ```
2. 跑真实训练：
   ```bash
   bash examples/alfworld_grpo/run_real.sh
   ```
   注意：real 训练需要 alfworld 和 verl 在同一个 conda 环境中

---

## 环境信息

| 项目 | 路径/版本 |
|------|----------|
| verl 源码 | `/workspace/verl` |
| ALFWorld 源码 | `/workspace/alfworld` |
| 游戏数据 | `/workspace/data/alf_data/` |
| Mock 数据 | `/workspace/data/alfworld/` |
| 模型 | `/workspace/models/Qwen3-1.7B` |
| conda 环境 | `alfworld`（Python 3.10.20，含 alfworld 依赖） |
| 裸环境 | 含 vllm/pytorch/ray（verl 训练用） |
| GPU | 公用机器，需等空闲 |

---

## 已知问题 & 备注

1. **setuptools 82.x 移除了 `pkg_resources`**：verl 的 `__init__.py` 依赖它，裸环境需确保 `setuptools<70`
2. **Mock 训练在裸环境跑**（不需 alfworld），Real 训练需要统一环境
3. **ALFWorld 数据路径**：`ALFWORLD_DATA=/workspace/data/alf_data`
