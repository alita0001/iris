# IRIS / RevAct

IRIS 是一个研究工程，用 execute–then–undo trace 测量 web-agent 转移在给定权限、预算、观测信号和 solver 集下的 operational recoverability，并为未来的策略训练生成可溯源监督。

> 状态（2026-07-16）：**非空 remediation smoke，不是可发布数据集。** 当前有 32 个 canonical point-level live-safe points，覆盖 Magento shopping 21 个和 Postmill reddit 11 个；其中 30 个为 `CHANGED/RECOVERED`，2 个为 `NO_EFFECT/UNKNOWN`。19 个 point 保存了完整、manifest-pinned 的 pre/post/recovery observation body；其余 13 个只有 hash/signal 级证据，已从 forward-supervised main set 排除。active transition-v3 main set 有 single 38 行和 trajectory-conditioned 24 行，62/62 OpenRouter teacher 输出通过 evidence QC；active S4 v4 有 150 个 snapshot-legal candidates，覆盖 25 个 state，role evidence 仅覆盖 3/8 类。active strict manifest-pinned DPO body 有 28 对（legal-candidate 6 / on-policy 22），但 train shard 仍为 0。严格 base train/dev/test 与 export 仍被 fail-closed 闸门拒绝，readiness 有 13 个 blocker。当前没有负恢复 solver-union 点、第三独立环境、API/DB signal 证据、可入 formal 的长 episode、模型效果、coupled/decoupled 或在线 RL 结果。

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
| `data/grounded/probe_points.jsonl` | 32 | canonical live point body；30 `CHANGED/RECOVERED`、2 `NO_EFFECT/UNKNOWN`；shopping 21 / reddit 11 | smoke only |
| `data/grounded/POINT_MANIFEST.jsonl` | 32 | canonical point manifest，1:1/hash 完整 | 完整性闸门 |
| `data/grounded/transitions/probe_transitions.v1.jsonl` / `TRANSITION_MANIFEST.v1.jsonl` | 19 / 19 | 19 个 point 的完整 pre/post/recovery observation body | active forward-supervision 必需；覆盖 19/32 |
| `data/eval/truth.jsonl` / `TRUTH_MANIFEST.jsonl` | 64 / 64 | 32 point × request/constraint 的独立 normative truth | static truth；不是 rollout |
| `data/train/formal/iris_sft_transition_v3.jsonl` | 38 | transition-backed single views，mock=0、failed=0、bid 可见率 100% | 19 个 exact-input point × 2 cases；被 base split 闸门阻塞 |
| `data/train/formal/iris_sft_multiturn_transition_v3.jsonl` | 24 | transition-backed stateless trajectory-conditioned views；history 均为 1 步 | 12 个 exact-trajectory point × 2 cases；被 base split 闸门阻塞 |
| `iris_dpo_transition_v3.jsonl` / `iris_dpo_multiturn_transition_v3.jsonl` | 3 / 3 | active reviewed legal-candidate pairs | body integrity通过；base DPO train仍为0 |
| `iris_dpo_on_policy_transition_v3_strict_openrouter_20260715.jsonl` | 2 | active manifest-pinned strict-parser model errors；精确复用历史 v2 trace 输入 | body source share通过；base DPO train仍为0 |
| `iris_dpo_on_policy_transition_v3_strict_openrouter_technology_20260715.jsonl` | 2 | active technology v3 guarded-rollout model errors | 2/2 trace-backed；base DPO train仍为0 |
| `iris_dpo_on_policy_transition_v3_strict_openrouter_llama32_3b_20260716.jsonl` | 9 | 12-case partial preflight 中通过严格 trace/truth/provenance gate 的 model errors | 与 3 条 rejection 分开存放；base DPO train仍为0 |
| `iris_dpo_on_policy_transition_v3_strict_openrouter_qwen3_8b_20260716.jsonl` | 9 | 12-case partial preflight 中通过严格 trace/truth/provenance gate 的 model errors | 与 3 条 rejection 分开存放；active DPO 总计 28，base DPO train仍为0 |
| `data/eval/on_policy/quarantine/*20260716-v1.rejections.v1.jsonl` / manifest | 6 / 6 | 两批各 3 条 provenance-incomplete call 的最小脱敏 rejection | 不修补、不计入 28 对 active DPO |
| `data/train/quarantine/dpo_naming_drafts_20260716/` | 4 files | 早期不匹配 active glob 的 2 body + 2 manifest；hash 与 rollback 记录完整 | `counts_as_formal=false`；strict replacements 与 body 逐字节一致 |
| `iris_dpo_on_policy_transition_v2_openrouter_20260715.jsonl` | 2 | pre-fix parser回归资产 | 历史审计；被active strict glob排除 |
| `iris_dpo_on_policy_openrouter_single_20260714_v1.jsonl` | 2 | superseded exact-source/raw-output errors | 历史审计；被active v3 glob排除 |
| `data/opinions/opinion_labels.v2.jsonl` / manifest | 6 / 6 | canonical label-blind LLM opinions；raw response/provenance 可回链 | 只覆盖 6/64 truth cases；human=0，不满足 readiness |
| `data/opinions/smoke/openrouter_deepseek_v3_2_20260714_v1.jsonl` | 4 | 早期 point×goal×LLM-rater smoke | superseded 历史诊断；不是 ground truth |
| `iris_sft_distilled_transition_v5.jsonl` / `iris_sft_multiturn_distilled_transition_v5.jsonl` | 38 / 24 | 62/62 OpenRouter teacher 输出通过 evidence QC | 当前 active source coverage=100%；不是模型效果 |
| `data/train/sft/revact_sft.jsonl` | 92 | historical Magento template SFT | 禁止 |
| `data/train/sft/revact_sft_multiturn.jsonl` | 62 | historical；按 `trajectory_id` 前缀推断 12 mock / 50 WebArena | 禁止；62/62 均缺显式 `environment_origin/is_mock/collector_success/run_id` |
| `data/train/sft/revact_sft_distilled.jsonl` | 0 | frozen legacy teacher path；canonical formal teacher 见上 | 禁止作为当前 coverage 分母 |
| `data/train/dpo/revact_dpo.jsonl` | 230 | 按生成机制为 synthetic flips；230/230 缺 `negative_source` | 禁止；审计为 `legacy_unspecified` |
| `data/train/dpo/revact_dpo_multiturn.jsonl` | 155 | 按生成机制为 synthetic flips；155/155 缺 `negative_source` | 禁止；审计为 `legacy_unspecified` |
| `data/train/ablation/iris_dpo*_synthetic*_v1.jsonl` | 95 + 60 | 全部 `synthetic_flip`，显式 `formal_dataset=false` | 仅 ablation |
| `data/raw/candidates/formal_candidates.v4.jsonl` / `FORMAL_CANDIDATE_MANIFEST.v4.jsonl` | 150 / 150 | 25 states × 6；25 expert、118 ordinary、7 constraint-trigger；全部 snapshot legal | active formal candidate body；category 只是提案 |
| `data/raw/candidates/candidate_roles.v6.jsonl` / manifest | 278 / 278 | `PROPOSED=228`、`EVIDENCED=50`；证据仅覆盖 expert 16、constraint-trigger 17、goal-violating 17 | decoy/ordinary/safe-alternative/VERIFY/policy-error evidence 仍缺 |
| v1–v5 formal SFT/DPO/teacher/candidate assets | 保留 | 旧 hash-only materialization、早期 OpenRouter smoke 与 strict v2 rollout 的历史证据 | superseded；不得混入 active corpus |
| `data/raw/candidates/iris_candidates.v3.jsonl` | 495 | 旧 proposal inventory | 不再是 formal consumer |
| `data/raw/trajectories_meta.jsonl` | 141 | 79 个唯一 physical `trajectory_id`；旧 93 行仍单独隔离 | 新增 task 48 尝试均失败/非 formal；admission 依赖精确 run closure |
| `data/raw/state_bank/shopping_key_states.jsonl` | 281 | 含 241 条 legacy keyword states、transactional 与 read-only collector states | 不能当作 point census |
| `data/raw/state_bank/formal_point_reached_states.jsonl` | 31 | 派生 reached-state view；不等于完整 transition body | 索引；不是 active main set 本身 |
| `data/raw/quarantine/legacy_lineage_rows.jsonl` | 93 | 只索引 93 条旧 meta；后续 transactional/read-only rows 不在该 quarantine | legacy 禁止 formal join |

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
  grounded/    # frozen legacy, 32-row point smoke, 19 transition bodies, specs and quarantine
  train/       # formal smoke views, empty release splits, legacy/quarantine
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

2026-07-16 验证快照：`pytest -q` 为 393 passed，`ruff check .` clean；active SFT 38/24、DPO 28 与 offline-RLVR 62 的 dry-run 均为 tokenizer drop=0。这些是代码/输入闸门结果，不是训练或模型效果。

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

普通 `revact probe` 仍只写 `data/grounded/smoke/` 下的 class-level smoke artifact。正式 live 路径是 `collect-points → prepare-point-batch → probe-points → author-truth → assemble`：它要求 reviewed label-free spec、精确 snapshot/candidate/action identity 和 1:1 manifest。当前 canonical points 覆盖 cart/wishlist/compare/reddit vote/reddit subscribe 五类自恢复动作。到 2026-07-16，32 个安全执行的 point 中 19 个具备完整 execute→undo body；30 个 point 构造性恢复，2 个动作无效果且 recovery 为 `UNKNOWN`。随后通过前端可选的 OpenAI-compatible provider 配置使用 OpenRouter 完成 62/62 active teacher QC。

历史 transition-v2 与 technology transition-v3 guarded smokes 各提供 2 条 manifest-pinned model-error traces。2026-07-16 两批 Magento constraint-only guarded rollout 各含 12 个 case、均禁用 target execution；partial authoring 对每批接收 9 条，将 3 条 provenance 不完整记录 fail-closed 到独立 quarantine。18 条 admitted trace 中 target-action attempt=0，backend commit 未观测；它们形成两个各9对的 strict DPO supplements。因此 active DPO 为 28 对（legal 6 + on-policy 22），但 train shard=0、train-level source share 仍失败；这些记录也不是模型效果或 `policy_generated_error` candidate-role 证据。每个 metric cell 仍受 n<30 gate 限制。

task 48 的 OpenRouter live collector 另有一次 0-step IRIS 格式失败和一次 4-step read-only/guard review；`iris.policy-attempt.v1` sidecar 保存了后者的 4/4 exact calls。4-step run 的前三步只读，第四步 combobox proposal 被 guard 替换为 `report_infeasible(...)`，target write=0。两个 run 均 `success=false` 且 `counts_as_formal=false`，不得写成 live 任务成功或长 episode 监督。OpenRouter 是用户在 provider/base URL/model/key-env 中的本次选择，不是仓库全局默认；没有运行下单、删除等破坏性 probe 或 LLM undo attacker。

独立意见基线不会进入 grounding/truth。先从 formal point×variant 生成只含
goal、pre-action AXTree、action 的输入，再调用用户选择的 provider：

```bash
python -m revact.cli prepare-opinion-inputs --limit 4
python -m revact.cli collect-opinions --limit 4 --dry-run \
  --input data/opinions/inputs/formal_smoke_limit4.v2.jsonl \
  --input-manifest data/opinions/inputs/formal_smoke_limit4.v2.manifest.jsonl \
  --provider custom --base-url https://provider.example/v1 --model vendor/model \
  --api-key-env REVACT_OPINION_API_KEY --rater-id rater-config-1 \
  --timestamp 2026-07-14T00:00:00Z --batch-id smoke-v1 --code-version <snapshot>
```

正式 paired-opinion gate 要求全部 64 个 evaluation case 各有至少一个 HUMAN
和一个独立 LLM rater；当前 canonical 文件只有 6 条 LLM opinion、human=0，不满足该条件。

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

当前 32 个 point 中有 19 个具备完整 transition body；active transition-v3 因而物化 single 38 行（19 point × request/constraint）和 trajectory-conditioned 24 行（12 exact-trajectory point × request/constraint）。其余 13 个 hash/signal-only point 仍保留在 canonical point body 供 measurement audit 使用，但不能冒充 next-state supervision。严格 base split 正确地失败，train/dev/test 均保持空：joint state/entity/goal-template/page-template/environment 隔离图仍是一个连通分量。cross-action 只能作为 held-out diagnostic，cross-site 因 entanglement 不可用；两者都不是模型实验。组装器不会回退到 `action_type -> latest label`；`--legacy-class-smoke` 只用于开发回归。

正式样本必须同时满足：

- 精确回链唯一 `probe_point_id`；
- `prediction_source=probe_transition`，`undo_source=probe_point_id`；
- supervised `click(bid)` 在当前输入中可见；
- `is_mock=false`；专家 SFT 的 `collector_success=true`，且这些声明必须与不可变
  collection run 血缘精确连接，不能只信任样本内布尔字段；
- train/deploy 共用 `revact.prompts.build_policy_messages`；
- tokenizer dry-run 的静默 drop 为 0；
- DPO 的 legal/on-policy negative 达到预注册比例；正式物化器只接收与当前 snapshot 合法候选精确连接的 reviewed legal-candidate counterfactual 或 on-policy error completion，并分开记录来源。

只有这些闸门通过后才应运行 GPU 训练。当前 `train/grpo.py` 是固定 prompt 的 **offline RLVR ablation**，不是 environment-in-the-loop 多轮 RL，不得宣称已解决 rollout distribution shift。

## 输入与 prompt 治理

训练和部署共用同一个 stateless message topology。user content 由
`revact.prompts.render_user` 精确渲染；标签是开标签分隔符，**没有 closing tags**：

```text
messages[0].role = system
messages[0].content = {agent_system}
messages[1].role = user
messages[1].content =
<goal>
{goal}

<history>
{history_block(history)}

<observation>
{pruned_obs_txt}
```

`build_policy_messages` 只返回上述 `system + user` 输入；SFT 物化时才另附 assistant completion。user content 的最后一行带换行符。

`POLICY_HISTORY_STEPS=9` 统一控制历史窗口。single materializer 与 multiturn builder 都从精确 raw trajectory 的连续 observation
重算 observed delta，并由 formal lineage audit 回链 COMPLETE run manifest。当前 62 条 formal 输入的 history 都只有 1 步，且 canonical stateless episode trace 仍为空，所以这只验证了格式和血缘，不能支持长历史/真正多步泛化主张。AXTree 按 target action 锚定修剪；组装和训练
都会硬性检查 supervised bid 可见性。新 Prompt bundle 使用内容寻址 `prompts_fp` 保存全文和父版本；独立的 `prompt_generation_fp` 对 `prompts_fp + producer + model + decode_config` 寻址，因此同一 prompt 的不同采样配置不会被静默合并。历史 SFT 的 `259fceac4e0c` 仍是无全文 bundle 的 orphan fingerprint，不得用当前 prompt 冒充。API key 只能通过环境变量提供，不得写入仓库、样本、prompt snapshot 或日志。

任何通过 formal truth-schema 校验的 rollout 都可离线重算全部计数、sample IDs、Wilson 区间、噪声敏感性和 state/site cluster bootstrap。早期有两个 OpenRouter execution-specific formal smoke：shopping run 的 2/2 cases 因 pre-observation hash mismatch 成为 `reach_error`，没有 evaluable episode；reddit run 有 2 个可达 cases，其中 constraint case 出现 1/1 declaration error 但 0/1 risky attempt，request case 输出非法文本 bid。随后又完成上文所列strict v2与technology v3 guarded runs；所有可评cell仍只有一个独立 point，`claimable=false`，不得写成效果结果：

```bash
python -m revact.cli eval-audit --input outputs/rollout_eval/<tag>.jsonl \
  --output outputs/rollout_eval/<tag>_audit.json
```

工作台代码支持 single/multiturn SFT/DPO、grounding、candidate、truth、teacher 和 opinion 的全文/血缘浏览。当前 32 个 point 都可展示 point-level lineage，其中 19 个 point 能继续回链完整 transition body；active sample/teacher views 均为 38 single + 24 trajectory-conditioned。candidate-role v6 sidecar 有 278 行，但只有 50 行、3 类具有 evidence，其余 taxonomy 仍不完整。旧 47-row teacher、26-pair DPO 与 transition-v2 阶段资产仅作历史审计。release export 仍因 13 个 split、数据覆盖和研究证据 blocker 被拒绝。

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
- 已完成 32 个非破坏性 point-level live smoke：shopping 21 / reddit 11，30 个构造性恢复证据与 2 个诚实 `NO_EFFECT/UNKNOWN`；
- 其中 19 个 point 有 manifest-pinned 完整 transition body；其余 13 个 hash/signal-only point 不进入 forward supervision；
- active S4 v4 有 150 条 snapshot-legal candidates（25 states × 6），25 个 expert candidate 与 point 精确连接；category 仍只是提案，role v6 也只有 50/278 行、3 类有行为证据；
- active formal SFT 有 single 38 行、trajectory-conditioned 24 行，全部 transition-backed、mock/failed=0 且 bid 可见；62/62 teacher 输出通过 evidence QC，但尚未形成可用 split；
- 训练/部署输入 builder、anchored pruning、prompt snapshot 和评测口径已在代码层对齐。

必须删除或明确标为未来工作：

- 旧 pilot `FSR=9.1%`（分母错误）；
- “普适 web-agent safety”或“不可逆证明”；
- 已有 cross-site/cross-action/cross-privilege 结果；
- 已完成 coupled-vs-decoupled 受控研究；
- teacher 蒸馏已经产生模型效果（当前 active 只有 62/62 条通过 QC 的数据产物，未训练/评测）；
- legacy 92/62 SFT 或旧 48-view hash-only materialization 含真实 post-state supervision，或 32-point smoke 足以支持模型效果；
- offline RLVR 是在线多轮 RL。

详细实现状态见 [IRIS 项目计划书](docs/plan/IRIS项目计划书.md) 和 [S1–S8 流程文档](docs/plan/S1-S8完整流程文档.md)。
