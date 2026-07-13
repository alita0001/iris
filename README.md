# IRIS / RevAct

IRIS 是一个研究工程，用 execute–then–undo trace 测量 web-agent 转移在给定权限、预算、观测信号和 solver 集下的 operational recoverability，并为未来的策略训练生成可溯源监督。

> 状态（2026-07-13）：**remediation prototype，不是可发布数据集。** 正式 point-level grounding 为 0；当前仓库中的 32 条 grounding、92 条 single SFT、62 条 multi SFT 和对应 DPO 是 legacy/开发资产，不得直接用于正式训练或论文结论。

## 核心语义

IRIS 不把 recoverability 当作 safety 的同义词。正式标签由两个正交维度组成：

```text
effect_status:
  CHANGED | NO_EFFECT | UNKNOWN

recovery_status:
  RECOVERED | PARTIALLY_RECOVERED |
  NOT_RECOVERED_WITHIN_BUDGET | UNKNOWN

undo_cost_steps: nullable integer
```

- `RECOVERED` 需要已执行 undo trace 的构造性证据。
- `NOT_RECOVERED_WITHIN_BUDGET` 只表示声明的 `solver_set` 在 `budget_k` 内失败，不是数学意义的 `IRREVERSIBLE`。
- 规范风险另由任务、政策、用户约束和授权决定。即使 UI/DB 状态复原，邮件、支付、第三方观察和声誉影响也未必能撤回。

完整数据边界见 [DATASET_CARD.md](DATASET_CARD.md)，研究限制见 [docs/Limitations.md](docs/Limitations.md)。

## 当前资产

| 资产 | 行数 | 状态 | formal 用途 |
|---|---:|---|---|
| `data/grounded/reversibility.jsonl` | 32 | frozen legacy/class smoke | 禁止 |
| `data/grounded/MANIFEST.jsonl` | 30 | legacy manifest | 禁止 |
| `data/grounded/probe_points.jsonl` | 0 | canonical point body | 待 live producer |
| `data/grounded/POINT_MANIFEST.jsonl` | 0 | canonical point manifest | 待 live producer |
| `data/train/sft/revact_sft.jsonl` | 92 | historical Magento template SFT | 禁止 |
| `data/train/sft/revact_sft_multiturn.jsonl` | 62 | historical；按 `trajectory_id` 前缀推断 12 mock / 50 WebArena | 禁止；62/62 均缺显式 `environment_origin/is_mock/collector_success/run_id` |
| `data/train/sft/revact_sft_distilled.jsonl` | 0 | teacher output | 无数据 |
| `data/train/dpo/revact_dpo.jsonl` | 230 | 按生成机制为 synthetic flips；230/230 缺 `negative_source` | 禁止；审计为 `legacy_unspecified` |
| `data/train/dpo/revact_dpo_multiturn.jsonl` | 155 | 按生成机制为 synthetic flips；155/155 缺 `negative_source` | 禁止；审计为 `legacy_unspecified` |
| `data/raw/candidates/iris_candidates.v3.jsonl` | 300 | 50 states 的 `iris.candidate.v2` legal 候选提案（当前无 decoy/on-policy） | 待 point grounding |
| `data/raw/candidates/CANDIDATE_MANIFEST.jsonl` | 300 | v3 candidate 逐行内容哈希；formal import/export 必检 | 完整性闸门 |
| `data/raw/quarantine/legacy_lineage_rows.jsonl` | 93 | 旧 meta 行的 index-only quarantine；原始文件未改 | 禁止 formal join |

32 条 legacy body 与 30 条 manifest 不匹配：前两条缺 provenance，已登记到 `data/grounded/quarantine/legacy_rows.jsonl`；其余 30 条仅登记为 class-level smoke，并标记 `formal_eligibility=EXCLUDED`。原文件未被覆盖。

## 目录

```text
revact/
  grounding/   # point schema, probes, migration, signals, solver traces
  data/        # collection, mutation/candidate discovery, assembly, splits
  train/       # SFT/DPO/offline-RLVR + fail-closed validators
  eval/        # truth-table metrics and guarded rollout
  prompts.py   # train/deploy shared message builder
  prompt_store.py
  server/ ui/  # dataset workbench
data/
  raw/
  grounded/    # frozen legacy, empty formal files, quarantine
  train/       # historical/development assets until rematerialized
docs/
  audit/       # baseline and remediation evidence
  plan/        # research plan and S1–S8 status
tests/
```

## 安全的快速开始

本地检查：

```bash
python -m pytest -q
wc -l data/grounded/{reversibility,MANIFEST,probe_points,POINT_MANIFEST}.jsonl
python scripts/migrate_grounding_v3.py --help
python -m revact.cli inspect
```

工作台：

```bash
conda run -n agentlab python -m revact.cli serve
# http://127.0.0.1:7788
```

mock probe：

```bash
conda run -n agentlab python -m revact.cli probe --list
conda run -n agentlab python -m revact.cli probe --mock shopping.add_to_cart
```

普通 `revact probe` 只写 `data/grounded/smoke/` 下的 class-level smoke artifact，既不修改冻结的 32/30 legacy 文件，也不是 formal point producer。`--formal-spec` 提供 label-free point identity 入口，但当前注册探针尚未产出完整 transition hashes/signals，因而会被 schema 拒绝；不要因为普通 probe 成功就升级证据强度。

完整的外部 measurement/truth 产物只能通过版本化导入边界进入 canonical 文件：

```bash
python -m revact.cli import-grounding --input measured-points.jsonl
python -m revact.cli import-eval-truth --input authored-policy-truth.jsonl
python -m revact.cli materialize-dpo --negatives reviewed-errors.jsonl
```

三个命令都会拒绝空文件、不完整 provenance、point/truth 不匹配和非法候选；它们不会从 recovery 推导 safety，也不会替调用者生成意见标签。

## 组装与训练闸门

默认 assembler 读取 point-level grounding：

```bash
python -m revact.cli assemble
python -m revact.cli assemble-multiturn
python -m revact.cli split
```

在 formal point 为 0 时，正确行为是产生空结果/明确阻塞，不是回退到 `action_type -> latest label`。`--legacy-class-smoke` 只用于开发回归，其输出不得命名为 formal。

正式样本必须同时满足：

- 精确回链唯一 `probe_point_id`；
- `prediction_source=probe_transition`，`undo_source=probe_point_id`；
- supervised `click(bid)` 在当前输入中可见；
- `is_mock=false`；专家 SFT 的 `collector_success=true`，且这些声明必须与不可变
  collection run 血缘精确连接，不能只信任样本内布尔字段；
- train/deploy 共用 `revact.prompts.build_policy_messages`；
- tokenizer dry-run 的静默 drop 为 0；
- DPO 的 legal/on-policy negative 达到预注册比例；正式物化器只接收与当前 snapshot 合法候选精确连接的 reviewed/on-policy error completion。

只有这些闸门通过后才应运行 GPU 训练。当前 `train/grpo.py` 是固定 prompt 的 **offline RLVR ablation**，不是 environment-in-the-loop 多轮 RL，不得宣称已解决 rollout distribution shift。

## 输入与 prompt 治理

训练和部署共用同一个 stateless message topology：

```text
system
user: <goal> + <history> + <observation>
assistant: IRIS completion
```

`POLICY_HISTORY_STEPS=6` 统一控制历史窗口。multiturn builder 会从连续 observation
重算 observed delta；single reached-state 路径目前只检查
`history_source=trajectory` 和列表形态，尚未把每条 history 与不可变 raw run/hash 精确连接，
所以不能把该字段声明当成已验证的真实历史。AXTree 按 target action 锚定修剪；组装和训练
都会硬性检查 supervised bid 可见性。新 Prompt bundle 使用内容寻址 `prompts_fp` 保存全文和父版本；独立的 `prompt_generation_fp` 对 `prompts_fp + producer + model + decode_config` 寻址，因此同一 prompt 的不同采样配置不会被静默合并。历史 SFT 的 `259fceac4e0c` 仍是无全文 bundle 的 orphan fingerprint，不得用当前 prompt 冒充。API key 只能通过环境变量提供，不得写入仓库、样本、prompt snapshot 或日志。

任何通过 formal truth-schema 校验的未来 rollout 都可离线重算全部计数、sample IDs、Wilson 区间、噪声敏感性和 state/site cluster bootstrap。当前仓库只有 legacy rollout；默认 formal audit 会 fail closed，尚无 formal audit 结果：

```bash
python -m revact.cli eval-audit --input outputs/rollout_eval/<tag>.jsonl \
  --output outputs/rollout_eval/<tag>_audit.json
```

工作台代码支持 single/multiturn SFT/DPO 的全文浏览和导出。当前供浏览的是历史资产；由于 formal point=0，完整的 point-grounded lineage 尚无样本可展示。

## live / destructive 边界

未经明确授权，禁止真实下单、发布、删除、退款、支付、发邮件或大规模训练。破坏性 probe 必须同时具备：

1. 项目负责人对精确命令的批准；
2. `--commit`；
3. `REVACT_ALLOW_DESTRUCTIVE=1`；
4. 可回滚/克隆账户和完整审计日志。

UI/API/DB signal 只能说明被观测的状态投影是否复原，不能保证撤回现实世界副作用。

## 当前论文口径

可以声称：

- 仓库实现了 effect/recovery 正交本体、point schema、legacy quarantine 和 fail-closed validators；
- 已有 execute–then–undo 的 pilot 探针与 class-level smoke 资产；
- 已有 300 条 snapshot-legal S4 候选提案；其中 50 条
  `source=expert` 只是 producer-declared target-bid proposal，没有成功 collector
  provenance，也尚未 point-grounded；这些类别不是行为标签；
- 训练/部署输入 builder、anchored pruning、prompt snapshot 和评测口径已在代码层对齐。

必须删除或明确标为未来工作：

- 旧 pilot `FSR=9.1%`（分母错误）；
- “普适 web-agent safety”或“不可逆证明”；
- 已有 cross-site/cross-action/cross-privilege 结果；
- 已完成 coupled-vs-decoupled 受控研究；
- 已经生效的 teacher 蒸馏（当前 0 行）；
- 现有 SFT 含真实 post-state supervision；
- offline RLVR 是在线多轮 RL。

详细实现状态见 [IRIS 项目计划书](docs/plan/IRIS项目计划书.md) 和 [S1–S8 流程文档](docs/plan/S1-S8完整流程文档.md)。
