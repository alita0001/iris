# IRIS 整改交付报告

日期：2026-07-13  
仓库：`/workspace/iris`  
最终状态：`ready=false`，`non_vacuous=false`

## 1. 结果先行

本轮已经完成的是 **P0 语义、血缘和发布闸门的代码整改**；尚未完成的是 **非空 live point-level 数据、正式训练集和论文实验结果**。因此当前工程不会再把旧 pilot 伪装成正式证据，但也还不能投稿为一个已完成的数据集或模型实验。

已经真正修复：

1. 正式标签改成正交的 `effect_status × recovery_status`，`undo_cost_steps` 独立保存；有限预算搜索失败只允许写成 `NOT_RECOVERED_WITHIN_BUDGET`，不能生成数学意义的 `IRREVERSIBLE`（`revact/grounding/schema.py:28-55,108-157,212-272`）。
2. 建立唯一、版本化的 point-level schema、记录哈希与 body↔manifest 1:1 校验；正式样本必须精确引用 `probe_point_id`，不再允许 `action_type -> latest non-UNKNOWN` 绑定（`revact/grounding/schema.py:177-339,348-370,394-458`；`revact/data/governance.py:43-97,171-290`）。
3. 正式 `<prediction>`、`<undo>`、`<rev_check>` 的 admission gate 要求真实 transition diff、实际 undo trace 和 evidence reference；模板 effect/`UNDO_HINTS` 不能进入 formal main set（`revact/data/assemble.py:631-794`；`revact/train/validators.py:154-238`）。
4. AXTree 改为 action-anchored pruning，并对 assistant answer 与 pinned risky action 做 bid 可见性硬闸门；训练与部署统一调用 stateless message builder（`revact/envs/obs_utils.py:95-229`；`revact/prompts.py:406-443`）。
5. 评测把 recoverability 与 normative risk 分开；实现可审计的 FSR-declaration、FSR-attempt、FSR-commit、FBR、IER、UCR、恢复性分类、undo、校准、risk-coverage、噪声敏感性与 cluster bootstrap（`revact/eval/truth.py:20-170`；`revact/eval/metrics.py:249-390,398-681`）。旧 `9.1% FSR` 已撤回。
6. legacy/mock/失败或来源不明的训练资产采用 index-only quarantine；原 JSONL 和哈希未变，formal export 默认 fail closed（`revact/grounding/migration.py:91-177`；`revact/data/quarantine.py`；`revact/server/export.py:100-340`）。
7. S4 已有 snapshot-legal candidate schema、manifest、合法 bid 校验和 formal DPO exact join；LLM/规则只能提案，不能提交标签（`revact/data/candidates.py:112-218`；`revact/data/preferences.py:120-250`）。
8. prompt 全文、parent/diff 与 model/decode provenance 使用 content-addressed bundle；formal export 可恢复并校验这些 bundle（`revact/prompt_store.py:26-125,128-193`）。
9. SFT/DPO/GRPO/teacher/eval 均增加非空、provenance、token-drop 和事实一致性闸门；GRPO 明确命名为 `offline RLVR ablation`（`revact/train/sft.py:207-310`；`dpo.py:193-284`；`grpo.py:74-198,344-438`；`distill.py:87-268,411-495`）。
10. README、计划书、流水线文档、Dataset Card 与 Limitations 已同步删除未落地声明。

仍被数据、live 环境或算力阻塞：

- canonical grounding body/manifest 为 `0/0`；没有一条可用于论文的 point-level live transition/probe。
- formal single/multiturn SFT 为 `0/0`，train/dev/test 都为 `0`；bid visibility 被明确标成 `evaluated=false/passed=false`，mock 与 split 等空集 integrity 结果也不是 100% 实证验收。
- 历史 collection lineage 仍为 `93 meta rows / 31 unique trajectory_id`，28 个逻辑 ID 重复，93/93 条 metadata 的 `environment_origin` 未记录。
- teacher formal source=0、output=0，coverage=`null`；没有 teacher 已生效的证据。
- 没有三路 solver 的 live negative trace、DB/API signal 对照、attack break rate 或 live mutation-miner precision/recall。
- 没有正式训练、在线 RL、coupled/decoupled、cross-site/action/privilege、校准或 task-success 结果。

保存的机器可读结论见 `docs/audit/IRIS-formal-readiness.json:2-11,276-328`。判定器要求五个非空 gate 与全部 integrity gate 同时通过（`scripts/audit_formal_readiness.py:296-323`）。

## 2. 分阶段交付状态

| 阶段 | 代码/治理状态 | 非空数据验收 | 结论 |
|---|---|---:|---|
| 1. 构念与 point schema | 已落地并有 round-trip/manifest/legacy 测试 | 0 formal points | 代码完成，数据阻塞 |
| 2. transition、undo、输入 | exact join、evidence gate、anchored pruning、统一 builder 已落地 | 0 formal samples | 代码完成，非空链路未验证 |
| 3. 评测真值与指标 | truth schema、可审计指标、噪声/区间已落地 | 0 formal episodes | 代码完成，无论文结果 |
| 4. 状态发现与 S4 | mutation miner fixture、300 legal proposals 已落地 | 0 grounded candidates | 部分完成；缺 live recall 与 decoy/on-policy |
| 5. 负标签与 probe authoring | solver union/schema/label-free authoring/双闸门已落地 | 0 live triple-solver negative | 代码骨架完成，协议证据阻塞 |
| 6. 卫生、split、prompt、teacher、工作台 | quarantine、strict split、prompt bundles、浏览/导出 gate 已落地 | split/teacher 为空；旧 lineage 未修复 | 部分完成 |
| 7. 训练及 coupled/decoupled | SFT/DPO/offline RLVR validator 与 dry-run 已落地 | 无 formal rows；coupled/decoupled 未实现 | 训练禁止启动；第三贡献删除 |
| 8. 文档与论文声明 | 已同步诚实口径 | — | 完成 |

## 3. 修改文件清单

以下是主要生产代码和审计入口；测试及生成资产另列。行号以本报告生成时的工作树为准。

| 模块 | 文件与关键行 | 交付 |
|---|---|---|
| Grounding ontology | `revact/grounding/schema.py:28-55,108-458` | 正交枚举、point record、formal validation、哈希 manifest、immutable append |
| Legacy migration | `revact/grounding/migration.py:91-177`；`scripts/{migrate_grounding_v3,quarantine_legacy_lineage}.py` | 非破坏迁移、legacy/class-smoke/旧 trajectory lineage 隔离与报告 |
| Probe execution | `revact/grounding/base.py`；`revact/grounding/solvers.py`；`revact/grounding/authoring.py` | formal context、solver union、无 label authoring、预算/trace gate |
| Probe registry | `revact/grounding/probes/{shopping,shopping_admin,reddit}.py` | 新 recovery 术语与 formal persistence 接口 |
| Formal import | `revact/data/formal_import.py`；`revact/cli.py:130-270` | 独立 point/truth import 与 `probe --formal-spec` fail-closed 入口 |
| Sample governance | `revact/data/governance.py:43-332` | point/truth/prompt/candidate exact join、release reasons |
| SFT assembly | `revact/data/assemble.py:631-971` | probe transition/undo source、formal exclusion report |
| Multiturn | `revact/data/multiturn.py:68-152,116-461` | 真实 trajectory history、mock/failed/ambiguous exclusion、完整 IRIS turn |
| Input alignment | `revact/prompts.py:319-443`；`revact/envs/obs_utils.py:95-229`；`revact/policies.py` | shared builder、统一 K、target-aware pruning、bid hard gate |
| Collection lineage | `revact/data/collect.py:30-270` | 全 legal control 枚举、immutable run_id、事务式新 collection manifest |
| Mutation/S4 | `revact/data/mutation_miner.py:66-259`；`revact/data/candidates.py:112-218` | signal diff detector、candidate schema/legality/manifest |
| DPO source | `revact/data/preferences.py:120-250` | reviewed legal/on-policy negative exact join 与 ≥50% gate |
| Quarantine | `revact/data/quarantine.py`；`scripts/quarantine_legacy_training.py` | 539 条旧记录索引隔离，源文件不改 |
| Split | `revact/data/splits.py:109-420` | state/entity/template/environment/action/privilege component isolation |
| Eval truth | `revact/eval/truth.py:20-170` | normative truth 与 recovery 分离、truth manifest |
| Eval metrics/audit | `revact/eval/metrics.py:249-681`；`revact/eval/audit.py:60-128` | numerator/denominator/IDs/CI、UNKNOWN、noise、bootstrap |
| Rollout/static eval | `revact/eval/rollout.py`；`revact/eval/decisions.py` | point/truth preflight、attempt/commit/guarded 字段、gold p99 budget |
| Training gates | `revact/train/validators.py:116-345`；`revact/train/{sft,dpo,grpo,distill}.py` | formal evidence、completion-only、token drop、DPO source、reward truth table、teacher QC |
| Prompt governance | `revact/prompt_store.py:26-202`；`revact/prompts.py:319-388` | immutable content/generation bundles、diff 与恢复 |
| Workbench/export | `revact/server/{datasets,quality,export,adapters,app}.py`；`revact/ui/views.js` | formal/legacy 分层、lineage 浏览、无直接 label authoring、发布 gate |
| Audit scripts | `scripts/audit_remediation_baseline.py`；`scripts/audit_formal_readiness.py:262-343`；`scripts/audit_mutation_miner.py` | 可复算基线、readiness、fixture detector audit |
| 配置/入口 | `configs/default.yaml`；`revact/config.py`；`revact/cli.py`；`revact/envs/harness.py` | formal 路径、统一历史配置、安全命令入口 |
| 文档 | `README.md`；`DATASET_CARD.md`；`docs/Limitations.md`；`docs/workbench.md`；`docs/tutorial.md`；`docs/plan/*.md` | 当前事实、协议、claims gate、限制与操作说明 |

新增或扩展的测试覆盖：

- `tests/test_grounding_schema.py`、`test_formal_import.py`、`test_solver_union.py`、`test_probe_authoring.py`；
- `tests/test_formal_readiness_audit.py`、`test_formal_export_governance.py`、`test_quarantine.py`；
- `tests/test_candidates_s4.py`、`test_mutation_miner.py`、`test_formal_preferences.py`；
- `tests/test_eval_truth.py`、`test_eval_metrics.py`、`test_eval_rollout_formal.py`、`test_eval_audit_cli.py`；
- `tests/test_train_remediation.py`，并扩展原有 assemble、multiturn、split、workbench、prompt、probe 与 offline pipeline 测试。

生成/迁移资产：

- `data/grounded/{probe_points,POINT_MANIFEST}.jsonl`：canonical formal 文件，当前 0/0；
- `data/grounded/quarantine/`：2 条缺 provenance legacy 索引、30 条 class-smoke 索引、migration report；
- `data/raw/quarantine/`：93 条旧 trajectory metadata 的 index-only lineage quarantine 与源资产哈希；
- `data/train/quarantine/`：539 条旧训练记录的不可变索引和报告；
- `data/raw/candidates/iris_candidates.v3.jsonl` 与 `CANDIDATE_MANIFEST.jsonl`：300/300；
- `data/train/formal/`：空的 formal assembly/split 产物及明确 blocked report；
- `data/manifests/prompts/` 与 `data/manifests/prompt_generations/`：content-addressed prompt provenance。

用户原有工作树状态 `D docs/plan/IRIS工程深度评审Prompt.md` 与 `?? docs/plan/项目整改prompt.md` 被保留，没有恢复、覆盖或纳入整改。

## 4. 数据迁移与 quarantine

### 4.1 Grounding

源 `data/grounded/reversibility.jsonl` 与 `MANIFEST.jsonl` 未修改。迁移报告位于 `data/grounded/quarantine/migration-report.json`：

- 旧 body 32 行、manifest 30 行；
- 前 2 行缺 `probe_id/state_id`，只登记进 legacy quarantine；
- 其余 30 行保留为 class-level probe smoke，不能作为 point label；
- canonical formal point=0；
- 原 body SHA-256 前后均为 `5c225b8caf35834f8bbfa04978efbc4957a94ba55485349fe023df9b83759a7a`。

唯一 `place_order=IRREVERSIBLE` 是第 2 条 legacy 行；它没有 provenance，正式训练中是 `EXCLUDED`，没有被凭空补 ID，也没有提升为负标签。

回滚：删除 `data/grounded/quarantine/`、空的 `probe_points.jsonl` 和 `POINT_MANIFEST.jsonl` 即可；旧源不需要恢复。

### 4.2 Training

`data/train/quarantine/migration-report.json` 记录：

- indexed=539、quarantined=539、formal eligible=0；
- 539/539 缺 formal flag、point/state/action-instance 或 collector-success proof；
- mock=12；failed-trajectory reason=6；ambiguous duplicate lineage=56；
- non-deployment negative=385；non-trajectory history=322；
- prediction/undo source 不合格各 154。

这些 reason 是可重叠集合，不能相加当作行数。迁移模式是 `index_only_source_files_unchanged`；回滚只需删除 `data/train/quarantine/`。

### 4.3 Historical trajectory lineage

`data/raw/quarantine/legacy_lineage_manifest.json` 与 `legacy_lineage_rows.jsonl` 对 93 条历史 metadata 做了 index-only 登记：31 个 unique logical trajectory ID，28 个 ID 重复，90/93 行属于重复 ID，formal eligible=0。每条登记保存 source line 与 row hash；manifest 保存 meta、31 个 raw trajectory 文件和 state-bank 文件的 SHA-256。原 `trajectories_meta.jsonl`、raw trajectory 与 state-bank 文件均未改。

这项迁移只隔离问题，没有把 93 次 append 复原成 93 个真实独立 run；readiness 继续把 collection lineage 标为失败。回滚只删除 `data/raw/quarantine/legacy_lineage_*`。

## 5. 新旧 schema 对照

| 旧数据/口径 | canonical `iris.grounding.point.v1` | 强度变化 |
|---|---|---|
| 单列 `label` 混入 `NO_EFFECT`、`REVERSIBLE_WITH_COST(k)`、`IRREVERSIBLE` | `effect_status={CHANGED,NO_EFFECT,UNKNOWN}` 与 `recovery_status={RECOVERED,PARTIALLY_RECOVERED,NOT_RECOVERED_WITHIN_BUDGET,UNKNOWN}` | effect 与 recovery 正交 |
| `REVERSIBLE_WITH_COST(k)` 字符串 | nullable integer `undo_cost_steps` | 可计算 cost MAE，不再扩类别 |
| 有限搜索失败=`IRREVERSIBLE` | `NOT_RECOVERED_WITHIN_BUDGET` + `budget_k` + `solver_set` + `budget_exhausted` | 不再夸大 one-sided evidence |
| action-class 最新行作 label | `probe_point_id`、`probe_run_id`、`state_id`、`action_instance_id`、`candidate_id` exact join | 防止类内恒定假设与错绑 |
| evidence 中零散 baseline/after/undo | pre/post/final signal、observation hash、undo actions/hashes、residual diff、solver trace | transition 与 recovery 可回放审计 |
| `site` 或轨迹名前缀推断 | environment family/instance/origin、mock、task/trajectory/run/seed、account/privilege | 明确环境与权限条件 |
| body 与宽松 manifest | record SHA、point ID、body↔manifest 1:1 | 篡改与缺行 fail closed |
| recovery 推导危险 | 独立 `iris.evaluation.truth.v1` 保存 policy/constraint truth、goal necessity 与 expected decision | recoverability ≠ safety |

## 6. 测试、静态审计与 dry-run

### 6.1 代码质量

最终执行：

```bash
git diff --check
conda run -n agentlab ruff check revact scripts tests
conda run -n agentlab python -m compileall -q revact scripts tests
conda run -n agentlab pytest -o addopts='' -q
```

结果：全部退出码 0；Ruff `All checks passed!`；测试 `198 passed in 1.37s`。整改前基线为 `78 passed`，见 `docs/audit/IRIS-remediation-baseline.md:23-35`。

新增测试覆盖用户要求的 20 类路径：schema round-trip、manifest 1:1、唯一 point、legacy/mock export gate、bid 可见、长 AXTree、K=9 历史、train/deploy byte-equivalence、失败轨迹、immutable run、第二 undo 通道、budget exhaustion、UI/DB fixture、reward truth table、split 泄漏、teacher evidence、prompt 恢复、token-drop、指标 IDs、single/multiturn 浏览导出。

### 6.2 数据审计

```bash
conda run -n agentlab python scripts/audit_remediation_baseline.py
conda run -n agentlab python scripts/audit_formal_readiness.py \
  --output docs/audit/IRIS-formal-readiness.json --allow-blocked
conda run -n agentlab python scripts/audit_mutation_miner.py \
  --output docs/audit/mutation-miner-fixture-report.json
```

结果：

- baseline 哈希与原始行数保持不变；
- readiness 为 `ready=false/non_vacuous=false`，阻塞 8 项；除五个非空数据 gate 外，还包括 bid visibility 未评估、旧 collection lineage 与 teacher；
- fixture mutation miner 枚举 240 controls、48 positives，precision/recall=1.0，Wilson 95% 下界 0.9259，240/240 reset；该数字仅是离线 fixture，不是 live WebArena/GitLab 结果。

S4 审计：300 unique candidates、50 states、每 state 6 个；300/300 的 `(state_id,snapshot_hash,bid)` 合法。来源为 a11y 250、recorded expert 50；类别为 ordinary 209、expert 50、uncertain 39、constraint-trigger 2。没有 decoy、on-policy 或 point-grounded candidate。4 个 `state_id` 对应多个 snapshot hash，formal join 必须继续用 snapshot hash 消歧。

### 6.3 Formal pipeline（预期 fail closed）

| 命令 | exit | 结果 |
|---|---:|---|
| `python -m revact.cli assemble` | 1 | reached 46、blocked 46、SFT/DPO 0；拒绝空发布 |
| `python -m revact.cli assemble-multiturn` | 1 | SFT/DPO 0、skipped 59；mock 6、failed 13、ambiguous 5、missing point 35 |
| `python -m revact.cli split` | 1 | train/dev/test 0/0/0；拒绝空 split |
| `python -m revact.cli train --dry-run` | 1 | 正式 SFT 0 行 |
| `python -m revact.cli train-dpo --dry-run` | 1 | 正式 DPO 0 行 |
| `python -m revact.cli train-grpo --dry-run` | 1 | 无 usable formal rows |
| `python -m revact.cli eval --dry-run` | 1 | 正式 test 0 行 |
| `python -m revact.cli eval-rollout --limit 1` | 1 | canonical point 0，未连接模型或环境 |
| formal dataset export dry-run | 返回 `ok=false` | 没有非空 split，不写 release |
| `eval-audit` 对旧 SFT test | 1 | 缺 point 与独立 truth，拒绝称 formal audit |

这里的非零退出码是验收结果：旧数据、空数据或无独立 truth 不能被误报成成功。

### 6.4 Legacy 只读诊断

使用显式 `--allow-legacy`/`--legacy-development`，全部退出 0，但不会启动训练或产生论文指标：

- SFT：92 rows，token p50=2699.5、p95=2867.3、max=2886、drop=0@4096，completion-only=true；
- DPO：230 pairs/460 sequences，p50=2666.5、p95=2855、max=2886、drop=0；negative source 全为 `legacy_unspecified`；
- GRPO：92 prompts，p50=2491、p95=2658.5、max=2669、drop=0；明确 `environment_in_loop=false`；
- static eval：22 rows，全部 cell n<30；gold completion p99=221，自动 generation budget=237。

基线 multi SFT 若按 4096 token 训练会 drop 6/62；当前 formal multi 为 0，因此不能声称 formal token-drop=0 已通过非空验收。

## 7. 修复前后关键统计

| 指标 | 修复前 legacy | 当前 formal | 诚实结论 |
|---|---:|---:|---|
| missing point provenance | 32/32 grounded 缺 state_id；2/32 缺 probe_id | 0/0 | schema/gate 已修，数据为空，不是修成 0 缺失 |
| body/manifest | 32/30 | 0/0 exact integrity | legacy mismatch 冻结；canonical 仍空 |
| mock contamination | multi 12/62（按 trajectory prefix 推断） | 0/0 | formal gate 会拒 mock，但无非空验证 |
| failed expert provenance | 158/241 key states 来自失败轨迹；success 22 meta rows/12 unique IDs | 0/0 | 旧专家资产不可用；新 collection gate 尚无数据 |
| gold risky bid visibility | multi 12/62=19.35%，missing 50 | checked=0，rate=`null` | anchored builder 有测试，无非空 corpus 结果 |
| tokenizer drop | single 0/92；multi 6/62@4096 | 无 rows | dry-run 可 fail on drop，尚无 formal p50/p95/max |
| split overlap | 旧 70/22 仅 sample-derived state key overlap=0 | 0/0/0，各轴 overlap=0 | formal 的 0 overlap 是空集，不是泛化证据 |
| teacher coverage | source=92 legacy、output=0 | source=0/output=0，coverage=`null` | teacher 未生效 |
| trajectory identity | 93 meta rows/31 unique IDs，28 重复 IDs | 新 collector gate 无新 run | 历史 lineage 仍是阻塞项 |
| candidate legality | 无正式 S4 | 300/300 snapshot-legal proposals | proposal 可用；0 grounded，谱系不完整 |

## 8. 未执行操作与待批准命令

本轮没有执行真实下单、发帖、删除、退款、支付、邮件、live 浏览器 probe、付费 API、GPU 训练或外部服务修改。用户在聊天中粘贴的 OpenRouter credential 没有被调用、回显、写入配置、日志或仓库；由于它已经暴露在会话文本中，应立即在提供商控制台撤销并轮换。

以下命令只是 **阻塞解除后的精确 runbook**，不是现在可安全直接运行的命令。所有路径与环境变量都必须先由负责人确认；密钥只用环境变量，不应出现在 shell history 或配置文件。

非破坏 live point（仍需先补齐 probe 对 pre/post/final signal 与 solver trace 的实际采集，否则 schema 会拒绝）：

```bash
source scripts/export_webarena_env.sh
conda run -n agentlab python -m revact.cli probe \
  shopping.add_to_cart --formal-spec data/grounded/specs/approved-shopping-add.jsonl \
  --product-url "$WA_SHOPPING/product/<approved-product-id>" --budget 12
```

破坏性 probe，只有在克隆账户/事务回滚沙盒与书面批准后：

```bash
source scripts/export_webarena_env.sh
REVACT_ALLOW_DESTRUCTIVE=1 conda run -n agentlab python -m revact.cli probe \
  shopping.place_order --formal-spec data/grounded/specs/approved-place-order.jsonl \
  --product-url "$WA_SHOPPING/product/<approved-product-id>" --budget 12 --commit
```

导入独立 normative truth 与重物化：

```bash
conda run -n agentlab python -m revact.cli import-eval-truth \
  --input data/eval/approved_truth.jsonl --data-root data
conda run -n agentlab python -m revact.cli assemble
conda run -n agentlab python -m revact.cli assemble-multiturn
conda run -n agentlab python -m revact.cli split
```

付费 teacher，仅在 formal source 非空、预算获批且使用新轮换密钥后：

```bash
export DEEPSEEK_API_KEY='<rotated-approved-key>'
conda run -n agentlab python -m revact.cli distill --limit <approved-row-count>
```

GPU 训练，仅在 readiness `ready=true` 后：

```bash
conda run -n agentlab python -m revact.cli train
conda run -n agentlab python -m revact.cli train-dpo
conda run -n agentlab python -m revact.cli train-grpo
```

这些命令仍只覆盖 SFT/DPO/offline RLVR；environment-in-the-loop RL 需要另行实现克隆账户、事务 rollback 与 episode-level 日志，当前没有可批准即运行的命令。

## 9. 论文声明 gate

当前仍可声称：

- 提出一个 **agent/privilege/solver/budget/signal-relative operational recoverability** 的待验证构念；
- 给出正交 point-level schema、构造性 positive evidence 与预算内 failed-recovery 的诚实命名；
- 实现了 fail-closed 的数据血缘、输入、候选、评测和训练工程协议；
- 提供离线 fixture 和单元测试，证明协议实现能检测已知的错绑、泄漏、格式、reward hack 与导出问题。

必须删除或继续禁止：

- “IRIS 已证明普适 web-agent safety”或“不可逆性”；
- “已有真实 forward supervision/teacher internalization 效果”；
- “已有 cross-site、cross-action、cross-privilege 泛化”；
- “已有 coupled-vs-decoupled 受控结果”或第三项贡献已落地；
- “GRPO/GSPO 解决了多轮 rollout shift”或属于 environment-in-the-loop RL；
- “pilot FSR=9.1%”、base 0% 代表安全、或任何 n<30 cell 的效果结论；
- calibration、task success、attack break rate、live mutation precision/recall 等不存在的数字。

投稿判断：当前只能作为 **方法/数据协议的工程预注册或 artifact draft**，不具备 main-track empirical paper 的证据。Dataset/benchmark track 也尚不成立，因为 formal point corpus 为空。最小可投稿门槛是非空、跨环境、point-grounded corpus 与独立 truth；coupled/decoupled 若不完整实现，应从贡献中永久删除。

## 10. 下一阶段及人日

按依赖和“数据优先于模型”排序：

| 顺序 | 工作包 | 依赖 | 估算 |
|---:|---|---|---:|
| 1 | 将现有 site probes 补齐真实 pre/post/final observation hash、signal、solver trace，使 `--formal-spec` 能产出第一条 admissible point | 无 | 4–6 人日 |
| 2 | 建立克隆账户/reset/事务回滚与 destructive approval runbook | 1 | 4–7 人日 |
| 3 | 在 shopping/reddit/admin 采集 ≥30/cell 的 point transition；三路 solver 攻击负例 | 1–2 | 15–30 人日 + 环境机时 |
| 4 | 人工双审独立 normative truth；计算 opinion-vs-behavior 分歧 | 3 | 5–8 人日 |
| 5 | 修复历史 collection lineage，或明确永久冻结并只用新 run_id 重采 | 1 | 3–5 人日 |
| 6 | 补 S4 decoy、goal-violation 与 policy on-policy error，保证 ≥50% deployment negatives | 3 | 4–6 人日 |
| 7 | 重物化 formal single/multi、跑 bid/token/split/readiness，要求非空 `ready=true` | 3–6 | 3–5 人日 |
| 8 | teacher 覆盖 ≥95%，事实一致率达到预注册阈值 | 7、付费批准 | 2–4 人日 + API 成本 |
| 9 | SFT→legal/on-policy DPO，分层跑静态/rollout 指标 | 7–8、GPU 批准 | 8–15 人日 + GPU |
| 10 | 接入 WebArena GitLab 或 WorkArena，复跑 detector/probe/split；之后才评估 cross-site | 1–7 | 12–25 人日 |
| 11 | 若保留第三贡献，实现 capacity/compute-matched coupled/decoupled 与 multi-head baseline | 7–9 | 15–25 人日 + GPU |
| 12 | 仅在前述失败显示多轮 shift 后，实现 sandboxed online RL | 2、9 | 20–40 人日 + GPU/环境 |

最短关键路径是 `probe instrumentation → approved live points → independent truth → formal materialization/readiness → teacher/model experiments`。在第 7 项通过前，不应投入 DPO/GRPO 调参。

## 11. 可复现入口

```bash
cd /workspace/iris
conda run -n agentlab pytest -o addopts='' -q
conda run -n agentlab python scripts/audit_remediation_baseline.py
conda run -n agentlab python scripts/audit_formal_readiness.py \
  --output docs/audit/IRIS-formal-readiness.json --allow-blocked
conda run -n agentlab python -m revact.cli train --dry-run
conda run -n agentlab python -m revact.cli train-dpo --dry-run
conda run -n agentlab python -m revact.cli train-grpo --dry-run
```

预期行为：测试通过；baseline 可复算；readiness 输出 `ready=false`；三个 formal trainer 在数据为空时退出 1。只有当 point、truth、formal samples、三路 split、lineage、teacher 全部非空且完整时，readiness 才应变为 true。
