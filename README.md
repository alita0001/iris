# RevAct / IRIS — Grounded-Reversibility Pipeline for Safe Web Agents

**IRIS: Invertibility-aware Internal world models for Safe web agents.**
一个策略内世界模型在一次自回归里输出：前向转移预测 `<prediction>` + 行为实测的
`<reversibility>` + 校准的 `<decision>` + 可执行 `<answer>`。可逆性标签不来自
人/LLM 意见，而来自 **execute-then-undo 行为测量**（agent-relative、预算 k 步、
in-episode）。方案全文见
[`docs/plan/06-IRIS项目计划书.md`](docs/plan/06-IRIS项目计划书.md)（权威链：
06 计划书 → 07 撞车文献 → 05 流程文档）。

## 安全红线（先读这个）

1. **API key 只走环境变量**（`DEEPSEEK_API_KEY` 等）。代码不接受 key 参数、
   不写日志、不落盘；`configs/default.yaml` 只存 env var 的**名字**。
2. **破坏性动作双闸门**：`--commit` 命令行标志 **且** `REVACT_ALLOW_DESTRUCTIVE=1`
   环境变量同时存在才会执行（如真实下单）。缺任一个都强制 dry-run（导航到目标
   按钮前停止，绝不点击）。admin 侧探针当前全部只有 dry-run 骨架。
3. 非破坏 / 自恢复探针（加购、wishlist、newsletter、compare、地址自增自删）可以
   直接在 live 环境反复运行。

## 目录结构

```
revact/
  config.py        # 路径/站点/训练配置（configs/default.yaml + env 覆盖）
  policies.py      # ScriptedShoppingPolicy + LLMActionPolicy（key 只从 env 读）
  cli.py           # 统一入口: python -m revact.cli <command>
  envs/            # BrowserGym/WebArena harness、fingerprint、obs utils、mock env
  grounding/       # ⭐ execute-then-undo 探针：base(注册表+双闸门 runner)、
                   #    signals(后端信号)、undo(撤销控制器)、probes/{shopping,admin}
  data/            # collect / reach / scale / assemble(目标措辞多样化) / splits
  train/           # sft(LoRA, --dry-run 校验) + distill(teacher 条件蒸馏, QC)
  eval/            # decisions(--dry-run 校验; 含诚实指标口径说明)
  viz/             # 自包含静态报告页（outputs/dataset_viz.html）
  server/ + ui/    # 🖥️ 数据集构建工作台（stdlib HTTP bridge + 原生前端，
                   #    见 docs/workbench.md；key 只存内存，标注只写 overlay）
configs/default.yaml + workbench.example.json
scripts/           # export_webarena_env.sh（live 前必 source）+ start_vllm.sh
tests/             # 离线全绿；mock env 复刻 Magento 标记，探针协议可离线测试
data/
  raw/             # trajectories/ state_bank/ trajectories_meta.jsonl product_urls.json
  grounded/        # reversibility.jsonl + MANIFEST.jsonl（probe 溯源，dry-run 不覆盖）
  train/           # sft/ dpo/ splits/
  annotations/     # 工作台人工覆核 overlay（append-only，绝不改写上面的产物）
docs/              # tutorial.md workbench.md findings-multisite.md + plan/{05,06,07}
ci.sh              # ruff + pytest（无环境无 key 可跑）
pyproject.toml     # 依赖声明（核心 stdlib-only；train/dev 走 extras）
```

## 快速开始

```bash
# 离线自检（任何机器）
./ci.sh

# 🖥️ 数据集构建工作台（配置→采集→约束→反事实→undo 标注→蒸馏→QC→导出 全链路）
python -m revact.cli serve          # -> http://127.0.0.1:7788
# 零基础教程 docs/tutorial.md（三个手把手演练）；架构边界 docs/workbench.md

# 探针一览 / mock 演练
python -m revact.cli probe --list
python -m revact.cli probe --mock shopping.add_to_cart shopping.wishlist_add

# live（先 source WebArena env；在 agentlab conda 环境里跑）
source scripts/export_webarena_env.sh
conda run -n agentlab python -m revact.cli probe --all-nondestructive
conda run -n agentlab python -m revact.cli probe shopping.place_order          # dry-run
# 真实下单（不可逆！需要显式双闸门 + 项目负责人批准）：
# REVACT_ALLOW_DESTRUCTIVE=1 conda run -n agentlab python -m revact.cli probe shopping.place_order --commit

# 数据流水线（离线部分）
python -m revact.cli assemble && python -m revact.cli split
python -m revact.cli assemble-multiturn      # P1：轨迹 -> 多轮 chat 样本（风险步带 <think>）
python -m revact.cli train --dry-run && python -m revact.cli eval --dry-run
python -m revact.cli train-dpo --dry-run && python -m revact.cli train-grpo --dry-run

# teacher 蒸馏冒烟（需要 export DEEPSEEK_API_KEY=...，只从 env 读）
python -m revact.cli distill --limit 10

# 训练/评测（qwen-vllm conda 环境 + GPU）
CUDA_VISIBLE_DEVICES=1 conda run -n qwen-vllm python -m revact.cli train
CUDA_VISIBLE_DEVICES=1 conda run -n qwen-vllm python -m revact.cli train \
    --data data/train/sft/revact_sft_multiturn.jsonl --max-len 8192   # 多轮（全 assistant 轮计损）
CUDA_VISIBLE_DEVICES=1 conda run -n qwen-vllm python -m revact.cli train-dpo \
    --adapter outputs/sft_lora_p0        # 偏好对；--adapter = SFT 热启（ref 即 SFT 策略）
CUDA_VISIBLE_DEVICES=1 conda run -n qwen-vllm python -m revact.cli train-grpo \
    --adapter outputs/sft_lora_p0        # GRPO（可验证奖励）；不加 --adapter 则从基座冷启
CUDA_VISIBLE_DEVICES=1 conda run -n qwen-vllm python -m revact.cli train-grpo --gspo    # GSPO（序列级 IS）
CUDA_VISIBLE_DEVICES=1 conda run -n qwen-vllm python -m revact.cli eval --adapter outputs/sft_lora

# 守护式真实 rollout FSR 评测（P0 解锁的实验；agentlab 环境 + vLLM 服务模型）
# 破坏性风险动作只记录不执行（双闸门不破）；可逆动作真执行、事后清购物车
LORA_MODULES="iris-sft=outputs/sft_lora_p0" scripts/start_vllm.sh              # GPU 侧
# （等价于 vllm serve ... --enable-lora --lora-modules iris-sft=... --port 8300，
#   脚本额外处理 conda 内 CUDA_HOME 指向 wheel 自带 toolkit 的坑）
conda run -n agentlab python -m revact.cli eval-rollout --policy iris \
    --model iris-sft --base-url http://127.0.0.1:8300/v1 --states test --tag iris_sft
```

Prompt 格式唯一事实源：`revact/prompts.py` —— 训练样本、rollout 策略、部署共用同一
`<goal>+<history>+<observation>` 三段 user 格式（P0，训练分布=部署分布）。GRPO 奖励全部
离线可验证（format / decision-对-oracle / reversibility-对-实测标签 / 约束违反惩罚），
不需要 LLM judge，也不需要环境在训练回路里。

## Grounding 语义（当前实现的诚实边界）

- 可逆性 = **agent-relative**：当前 agent 用自身动作空间、预算 k 步（默认 12）、
  在本 episode 内能否使后端信号复原。谱系：`REVERSIBLE / PARTIALLY_RECOVERABLE /
  IRREVERSIBLE / NO_EFFECT / UNKNOWN`，另记录 `undo_steps / undo_actions /
  residual_diff`（undo 代价是测出来的，不是二元断言）。
- "后端信号" = 无 DB 权限下最结构化的 UI 读数（购物车行数、订单号 gridcell、
  订阅 checkbox 状态），不是自由文本关键词。
- 探针到达状态用**确定性重导航**（稳定 Magento URL + 按文本重解析元素）+
  探测前基线信号校验；不是录制 bid 前缀回放（bid 跨会话失效）。
  `envs.harness.replay_to_state` 保留用于 S2 轨迹态的复现校验与离线测试。
- 22 条 held-out 上的高决策准确率**不构成**"学会可逆性概念"的证据（决策与
  action_type×goal 变体共线）；概念性证据要等 cross-action-class split。
  详见 `eval/decisions.py` 模块注释。

## 已实测的 grounded 标签（live WebArena, 2026-07）

| 站点 | action | label | 依据 |
|---|---|---|---|
| shopping | add_to_cart | REVERSIBLE | 加购→购物车行数+1→逐行 Remove→复原（undo_steps 记录）|
| shopping | place_order | IRREVERSIBLE | 真实下单 000000193；订单页无可执行 Cancel（role 过滤+词边界）|
| reddit | vote | REVERSIBLE | upvote→score 0→1、投票态 F→T→点 Retract→复原（无残渣）|
| reddit | subscribe | REVERSIBLE | Subscribe→Unsubscribe 干净 toggle |

其余探针（shopping: wishlist/compare/newsletter/currency/address_add + review/
address_delete/admin×4；reddit: comment_submit + delete/edit dry-run 骨架）已实现，
其中 reddit.comment_submit 的 live commit（PARTIALLY，留 `[deleted]` 墓碑）待逐批批准。

## 多站点（sites）

工作台按 **WebArena 站点**扩展：站点注册表在 [`revact/config.py`](revact/config.py)
的 `SITES`，每站点声明 base-url 环境变量、会话任务、路径表。新增一个镜像 =
部署它 → 加一行 `SiteSpec` → 加 `revact/grounding/probes/<site>.py`。
下游全部读注册表（CLI `--site`、工作台站点分面、cross_site split）。

```bash
python -m revact.cli probe --list                    # 按站点分组的探针总表
python -m revact.cli probe --mock --all-nondestructive   # 全站点离线演练
# live reddit（先 source env，在 agentlab 环境；WA_REDDIT 指向 Postmill 镜像）：
python -m revact.cli probe --site reddit reddit.vote reddit.subscribe
```

已接入站点：`shopping` / `shopping_admin`（Magento）、`reddit`（Postmill）。
扩展中获得的实验发现见 [`docs/findings-multisite.md`](docs/findings-multisite.md)
（跨站点谱系中段、`Retract upvote`/`[deleted]` 表面标注混淆的跨站点复现等）。
