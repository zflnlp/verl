# ALFWorld GRPO 项目进度

## 当前状态：Zero-shot 验证通过，Mock 训练等待 GPU 空闲

最后更新：2026-06-09

---

## 已完成

### ✅ 代码实现（12 个文件）
| 文件 | 状态 |
|------|------|
| `verl/interactions/alfworld_interaction.py` | ✅ 已修正 API + dones tuple bug |
| `verl/tools/alfworld_tool.py` | ✅ |
| `examples/alfworld_grpo/config/alfworld_grpo.yaml` | ✅ |
| `examples/alfworld_grpo/config/alfworld_interaction_config.yaml` | ✅ |
| `examples/alfworld_grpo/config/alfworld_tool_config.yaml` | ✅ |
| `examples/alfworld_grpo/data_preprocess.py` | ✅ 已修正 API |
| `examples/alfworld_grpo/reward_function.py` | ✅ |
| `examples/alfworld_grpo/run_mock.sh` | ✅ |
| `examples/alfworld_grpo/run_real.sh` | ✅ |
| `examples/alfworld_grpo/eval_zero_shot.py` | ✅ 已修正 API + dones tuple bug |
| `examples/alfworld_grpo/setup_alfworld.sh` | ✅ |
| `examples/alfworld_grpo/README.md` | ✅ |

### ✅ API 验证（基于真实 ALFWorld 源码修正）
- 环境初始化：`get_environment(env_type)(config_dict)` ✅
- `init_env()` 需要 `config['general']['training_method'] = 'dagger'` ✅
- `reset()` 不接受参数，需通过修改 `alfred_env.game_files` 指定游戏 ✅
- `step()` 返回 `(obs, scores, dones, infos)` — `dones` 是 tuple 不是 list ✅
- `infos['admissible_commands'][0]` 访问正确 ✅
- 任务类型内部名映射（`pick_and_place` → `pick_and_place_simple`）✅
- 游戏文件路径：`task_type-Obj-None-Loc-ID/trial_T*/game.tw-pddl`（递归搜索）✅

### ✅ Bug 修复记录
1. **`training_method` 缺失**：`init_env()` 要求 `config['general']['training_method'] = 'dagger'`
2. **游戏文件目录结构**：不是 `task_type/` 子目录，是扁平的 `task_type-Obj-None-Loc-ID/` + 嵌套 `trial_T*/`
3. **`dones` 类型是 tuple**：`isinstance(dones, list)` 为 False → `is_done = (False,)` → `bool((False,)) = True` → 所有 episode 1 步结束。修复：`isinstance(dones, (list, tuple))`
4. **tokenizer.json 损坏**（14B）：fast tokenizer 失败，加了 fallback 到 slow tokenizer
5. **setuptools 82.x 移除 pkg_resources**：裸环境需 `pip install "setuptools<70"`

### ✅ ALFWorld 安装（服务器 Docker）
- 源码安装：`cd /workspace/alfworld && pip install -e .` ✅
- 游戏文件：`alfworld-download` → `/workspace/data/alf_data/`（4027 个游戏文件）✅
- API 测试通过：env creation, reset, step 全部正常 ✅

### ✅ Mock 数据生成
- 100 个 mock 任务（80 train / 20 test）→ `/workspace/data/alfworld/` ✅
- 需要 `pip install pyarrow` 才能写 parquet ✅

### ✅ Zero-shot 评估完成
| 模型 | Win Rate | 备注 |
|------|----------|------|
| Qwen3-1.7B | **20%** | 2 WON（12步）/ 8 LOST（30步超时） |
| Qwen3-14B | **50%** | 5 WON（4步完成）/ 5 LOST（30步超时） |

- 结果路径：`/workspace/data/alfworld_eval_zero_shot_14B/results.json`
- 1.7B 结果：`/workspace/data/alfworld_eval_zero_shot_1.7B/results.json`

### ✅ GitHub 推送
- 分支：`webshop-grpo-v0.4.1`
- 最新 commit：`bb0b884a`（Update progress: zero-shot eval done, 14B 50% win rate）

---

## 进行中
### ⏳ Mock 训练
- 命令：`bash examples/alfworld_grpo/run_mock.sh`
- 状态：**首次运行报 GPU 显存不足**（公用机器卡被占用）
- 等 GPU 空闲后重跑，如显存紧张可加 `ROLLOUT_GPU_MEM_UTIL=0.85`

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
| 模型 1.7B | `/workspace/models/Qwen3-1.7B` |
| 模型 14B | `/workspace/models/Qwen3-14B/` |
| conda 环境 | `alfworld`（Python 3.10.20，含 alfworld 依赖） |
| 裸环境 | 含 vllm/pytorch/ray（verl 训练用） |
| GPU | 公用机器，需等空闲；`CUDA_VISIBLE_DEVICES=0` 可用 |

---

## 已知问题 & 备注

1. **setuptools 82.x 移除了 `pkg_resources`**：verl 的 `__init__.py` 依赖它，裸环境需确保 `setuptools<70`
2. **Mock 训练在裸环境跑**（不需 alfworld），Real 训练需要统一环境
3. **ALFWorld 数据路径**：`ALFWORLD_DATA=/workspace/data/alf_data`
4. **14B tokenizer.json 损坏**：已加 fallback，或重新下载 `huggingface-cli download Qwen/Qwen3-14B tokenizer.json`
5. **14B safetensors 损坏**：需重新下载完整模型
