# WebShop GRPO 项目进度

> 最后更新: 2026-06-10

## 项目概述

使用 GRPO 算法训练 Qwen3-1.7B 模型成为 WebShop 购物 Agent。

- **分支**: `webshop-grpo-v0.4.1`（基于 verl v0.4.1，兼容 vllm 0.8.5）
- **服务器**: hgx18 (10.24.10.118) — 8x H100 80GB
- **WebShop 源码**: `benchmarks/WebShop/`（本地仓库中）

---

## 完成进度

| 阶段 | 状态 | 日期 | 备注 |
|------|------|------|------|
| 代码框架搭建 | ✅ | 2026-06-07 | interaction/tool/reward/data_preprocess |
| Mock 训练跑通 | ✅ | 2026-06-07 | avg reward 0.436 |
| WebShop 真实环境安装 | ✅ | 2026-06-08 | conda env `webshop`，Python 3.10 |
| Prompt & Action Format 对齐论文 | ✅ | 2026-06-09 | search[query] + click[button] |
| 零样本评估 Qwen3-1.7B | ✅ | 2026-06-09 | 50 episodes, reward = 0.000 |
| 零样本评估 Qwen3-14B | ✅ | 2026-06-09 | 50 episodes, reward = 0.000 |
| 评估脚本修复 | ✅ | 2026-06-10 | 双重reset/指令提取/可用动作/引号清洗 |
| 搜索返回0结果分析 | 🔄 | 2026-06-10 | 需优化prompt搜索策略 |
| 参考文献分析（TCOD） | ⏳ | — | 需读论文了解训练数据和评估方法 |
| 真实 GRPO 训练 | ⏳ | — | 需要 WebShop 服务器运行 |

---

## 当前问题

**搜索返回0结果**: 模型生成复杂查询如 `search["black women's dress XL polyester-spandex under $50"]`，但 WebShop 搜索是简单关键词匹配。需在 prompt 中添加搜索策略指导。

---

## 已修复 Bug

1. buy 作为独立 action → 统一为 click[buy]
2. 双重 env.reset() → evaluate_episode 不再 reset
3. 任务指令提取失败 → regex 适配 `Instruction: [SEP] <text> [SEP]`
4. 可用动作是占位符 → 从 observation 提取真实元素
5. action 多余引号 → _clean_action() 清洗
6. num_products=200 不支持 → 改为 100

---

## 下一步

1. 分析 TCOD 论文 — 了解训练数据和 zero-shot 方法
2. 优化 prompt 搜索策略
3. 重新零样本评估
4. 真实 GRPO 训练
5. 训练后评估对比
