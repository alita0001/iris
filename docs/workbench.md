# IRIS 数据集构建工作台（workbench）

> 本文描述当前工程能力，不把按钮、adapter 或历史文件当成已完成实验。新手可先读
> [`tutorial.md`](tutorial.md)；构念与数据边界见 [`Limitations.md`](Limitations.md)。

工作台是 CLI pipeline 的浏览与编排界面。生成、校验和导出逻辑仍在
`revact/{data,grounding,train,eval}`；UI 不能绕过 formal admission gates，也不能靠人工
覆核把意见写成行为标签。

## 当前数据闸门（2026-07-16）

| 资产 | 实测状态 | 用途边界 |
|---|---:|---|
| formal `probe_points.jsonl` / `POINT_MANIFEST.jsonl` | 32 / 32 | 非破坏性 point smoke；shopping 21 / reddit 11，30 recovered + 2 no-effect/unknown，不是 release corpus |
| transition body / manifest | 19 / 19 | 19 个 point 有完整 pre/post/recovery body；其余 13 个 point 不进 active forward supervision |
| formal truth / manifest | 64 / 64 | 32 point × request/constraint 的独立静态 policy truth；不是 rollout |
| legacy grounding body / manifest | 32 / 30 | 2 条 quarantine；30 条 class-smoke，全部 formal-excluded |
| S4 formal candidate / manifest | 150 / 150（25 states × 6，schema `iris.candidate.v2`） | active v4 全部 snapshot legal；25 expert、118 ordinary、7 constraint-trigger；角色证据仍不完整 |
| candidate role / manifest | 278 / 278 | v6 中 228 proposed、50 evidenced；证据仅覆盖 3/8 roles：expert 16、constraint-trigger 17、goal-violating 17 |
| active single / trajectory-conditioned SFT | 38 / 24 | 19 个 transition-backed point 的 single views 与其中 12 个 exact-trajectory point 的 conditioned views；mock=0、failed=0、bid=100%；base split blocked |
| active DPO body | 28 | legal-candidate 6 + on-policy 22；新两批 partial preflight 各9 valid/3 quarantine；body gate pass，base train=0 |
| historical single SFT | 92 | Magento-only legacy 资产，不是 formal set |
| historical multi SFT | 62 | 按 `trajectory_id` 前缀推断 12 mock / 50 WebArena；62/62 缺显式 origin/mock/success/run provenance，不得 formal export |
| active teacher distilled | 62 / 62 source views | single 38/38、trajectory-conditioned 24/24；数据 QC 通过不等于模型效果 |
| canonical opinion / manifest | 6 / 6 | 单一 LLM rater，只覆盖 6/64 truth cases；human=0，paired-opinion gate false |
| trajectory meta | 141 | 79 个唯一 physical `trajectory_id`；旧 93 行仍在 lineage quarantine；task 48 新尝试为 failed/non-formal |
| shopping key states | 281 | legacy keyword、transactional 与 read-only collector states；不是 point census |

formal grounding 使用唯一 `probe_point_id` 连接 state、action instance、transition、undo trace
和 label。旧的 `action_type -> latest non-UNKNOWN` 只可用于 legacy 浏览，不能作为训练 join。

## 启动

```bash
cd /workspace/iris
python -m revact.cli serve
# http://127.0.0.1:7788
```

离线浏览不要求 WebArena。live 环境、外部模型调用和破坏性 probe 有各自门控；站点注册或
“live ready”状态不代表已产生结果。未经批准，不得执行下单、删除、发布、支付等操作。

## Canonical ontology 与人工覆核边界

正式 schema 把两个问题分开：

- `effect_status`: `CHANGED | NO_EFFECT | UNKNOWN`
- `recovery_status`: `RECOVERED | PARTIALLY_RECOVERED |
  NOT_RECOVERED_WITHIN_BUDGET | UNKNOWN`

`undo_cost_steps` 是独立 nullable 整数。`IRREVERSIBLE` 仅为 legacy/display 兼容；预算内
solver 失败不得写成数学意义的不可逆。工作台可以收集 spec、审阅意见和 reject 决定，但
不提供“直接填写最终 label”的正式入口。标签必须来自 execute–then–undo evidence。

人工操作存入 `data/annotations/` overlay。formal probe authoring 不提供 label 字段；legacy
grounding 浏览页仍保留 `reversibility_override` 人审项，它只用于冲突/QC，不能改写 formal
probe truth。formal export 会拒绝 mock、失败专家轨迹、canonical history、缺 point provenance、
grounding 冲突或 bid 不可见的样本。

## 架构

```text
revact/server/
  datasets.py      浏览 single/multiturn/DPO/grounding/candidate/teacher 与 lineage
  adapters.py      pipeline action 到 CLI/纯函数的映射及前置条件
  jobs.py          子进程任务、日志与 secret 打码
  annotations.py   append-only 人工 overlay
  quality.py       数据质量统计
  export.py        formal admission gate 与审计导出
  app.py           HTTP 路由和内存配置
revact/ui/         原生 HTML/CSS/JS
```

prompt 内容以 fingerprint 关联 immutable bundle；新 assemble 应能由 fp 恢复全文、parent/diff、
模型和 decode 参数。历史 fingerprint 不一定有完整 bundle，工作台展示时必须标明缺口。

训练与部署的共享 user content 由 `revact.prompts.render_user` 精确生成；三个标签只有开标签，
不带 closing tags：

```text
<goal>
{goal}

<history>
{history_block}

<observation>
{pruned_axtree}
```

## 各页真实度

| 页面/能力 | 当前状态 | 不能据此声称什么 |
|---|---|---|
| Pipeline / Jobs | 可编排安全 CLI、显示产物与日志 | job 成功不等于数据通过 formal gate |
| 成功轨迹 | 可浏览 141 条 meta，合计 79 个唯一 physical ID | 旧 93 行仍不是 93 条独立轨迹；failed task-48 attempts 不是成功/formal episode |
| 关键状态 | shopping state bank 281 行，另有 reddit point state 资产 | 没有任务无关 detector 的 live recall 结果；32 points 不代表 coverage |
| 候选动作 | active v4 body + manifest 150 / 150；schema v2，25 states，bid/snapshot 合法 | category 是 proposal，不是 label；v6 role evidence 仍缺 5 类 |
| Probe authoring | declarative spec 可选动作、signals、undo、预算、安全等级；无 label 字段 | spec 保存或 fixture 通过不等于 live point |
| Grounding | 可分别浏览 32-row formal smoke 与 legacy/quarantine | 32 条旧记录不是正式 point dataset；32-row smoke 也不是统计充分的数据集 |
| Teacher | 可浏览 active evidence-aware QC 与 superseded fallback | active 62/62 是数据 QC；这不是蒸馏效果结果 |
| Dataset browser/export | 支持多数据族全文/血缘；formal export fail closed | base split 与 DPO train shard 未通过时拒绝 release |
| Eval | 有64条 versioned static truth、metric primitives 和多次历史 formal smoke artifact | strict3 每类仅 n=1；不是效果结果 |
| Train dry-run | SFT/DPO/offline RLVR validator 可审计输入 | 没有模型训练结果；GRPO 不是 environment-in-loop RL |

## S4 candidate 的精确边界

formal consumer 使用 `data/raw/candidates/formal_candidates.v4.jsonl`：150 条
`iris.candidate.v2`、覆盖 25 个 formal state，每个 state 6 条；其中 25 条 expert
candidate 可通过 state/action/snapshot 与一致的 point measurement 回链，另有 118 条 ordinary 和 7 条
constraint-trigger proposal。每条记录保存
candidate/state/bid/canonical action/category/source、
snapshot hash 与 snapshot legality，并由 150 行 manifest 做 1:1 哈希校验。278 行 candidate-role
v6 sidecar 包含 228 个 proposal 和 50 个 evidenced role；证据只覆盖 expert 16、
constraint-trigger 17、goal-violating 17。ordinary/decoy/safe-alternative/VERIFY/
policy-error 角色仍无合格证据。只有实际执行并关联唯一 `probe_point_id` 后，才能获得
effect/recovery 字段。495-row `iris_candidates.v3.jsonl` 是历史 proposal inventory，不是
formal consumer。

LLM 若用于提案，只能影响 candidate 排序或补充候选；bid 合法性由当前 snapshot 验证，
grounding 由行为探针产生。active DPO 已物化 single/multiturn 各 3 对 reviewed
legal-candidate counterfactual；另有 2 对 strict-parser OpenRouter raw-output
errors 通过 exact input/output hash 从历史 transition-v2 trace 复用于 manifest-pinned
transition-v3 supplement，新的 technology v3 guarded rollout 又产生 2 对 trace-backed errors。
新两批 Magento constraint-only guarded rollout 各含 12 cases，partial authoring 每批接收 9 条且
将 3 条 provenance 不完整记录送入独立 quarantine。因而 active on-policy 为 22、DPO
总计 28；18 条新 trace 的 target-action attempt=0，仍不等于合法 on-policy action-error
覆盖或模型效果。旧 24+2 对与 transition-v2 阶段保留但被 active glob 排除。最初两个
命名不匹配 strict glob 的 draft body/manifest 保留在
`data/train/quarantine/dpo_naming_drafts_20260716/`，`counts_as_formal=false`。

## 多站点与 live

代码注册了 shopping、shopping_admin、reddit，并包含 site-aware probe/split 组件。当前 32 个
formal point 来自 shopping 21 / reddit 11；这证明两个技术家族的 point pipeline 可运行，但不构成模型泛化
实验。因此：

- strict cross-site 仍不可用；cross-action held-out shard 只能作为诊断，不能当作已完成实验；cross-privilege 仍不可用；
- base train/dev/test 仍为空，cross-privilege 仍不可用；
- `SPLIT_REPORT.challenges.<axis>.available=true` 即便未来出现，也只表示非空数据分区，不表示完成实验；
- 11 个 reddit canonical points 可称 live-safe smoke；旧 fixture/class-smoke 仍不得追认为正式点；
- destructive probe 只有 dry-run/双闸门，未经批准不执行。

本次 live 执行通过工作台可配置的 provider/base URL/model/key environment 选择 OpenRouter；
OpenRouter 不是全局默认，DeepSeek/OpenAI/custom 等前端选项仍保留。历史 strict3 rollout
在同一 transition-backed Reddit point 上 2/2 可达；严格解析后两例均无可执行 action，target
execution=0、backend commit=0。constraint case 仍声明 `EXECUTE`，原始
FSR-declaration=1/1，但每个 metric cell 只有一个独立 point，均 `claimable=false`，不得写成
模型效果。它的精确输入负例可由 hash/manifest 追溯复用于 active v3 DPO，但没有因此成为 v3
rollout 结果。另一次 active-v3 technology smoke 也为 2/2 reach、两例均无可执行 action、
target execution=0 且 backend commit 未建立；constraint 同样声明 `EXECUTE`，原始
FSR-declaration=1/1、`claimable=false`。其 2 个 error traces 进入第二个 v3 supplement；该
n=1/cell smoke 仍不是效果结果。更早的 shopping drift、非法文本 bid 和贪婪解析器资产只作历史失败分析。

task 48 的 `iris.policy-attempt.v1` 安全 live 尝试包含一次 0-step IRIS 格式失败与一次
4-step read-only/guard review；后者前三步只读，第四步 combobox proposal 被替换为
`report_infeasible(...)`。两次均 `success=false`、`counts_as_formal=false`、target write=0，不是
live 任务成功或 canonical long episode。OpenRouter 仍是用户可选执行配置，不是全局默认。

工作台现在把 opinion rater 作为独立模型角色展示，仍由用户选择 provider、base URL、
model 和 key-env；key 值只保存在服务进程内存。`prepare-opinion-inputs` 只物化
goal、pre-action AXTree、action，`collect-opinions` 的 v2 记录以 point × goal
variant × rater 为键，并保存 prompt/call provenance。当前 canonical
`opinion_labels.v2.jsonl` 有 6 条 LLM opinions、无 human rater，只覆盖 6/64 truth cases；
它不会进入 grounding/truth，也不满足 paired-opinion readiness gate。早期 4-row smoke
保留为 superseded 历史诊断。

历史调试事件见 [`findings-multisite.md`](findings-multisite.md)，该文件已明确标为
legacy incident notebook。

## 验证与导出

```bash
python -m pytest -q
# 默认 fail closed：当前 point/SFT/DPO/teacher body 已非空，但 base split/DPO train shard 阻塞，以下 release readiness 命令应返回 1
python scripts/audit_formal_readiness.py
# 只在需要保存阻塞报告时放宽进程退出码；报告仍是 ready=false
python scripts/audit_formal_readiness.py --allow-blocked --output docs/audit/IRIS-formal-readiness.json
python -m revact.cli train --dry-run
python -m revact.cli train-dpo --dry-run
python -m revact.cli train-grpo --dry-run
```

2026-07-16 已完成的验证快照是 393 tests passed、Ruff clean；active SFT 38/24、DPO 28
和 offline-RLVR 62 的 dry-run 均为 tokenizer drop=0。不应把这些代码/输入检查写成训练结果。

当前仓库的正确结果应是：历史/quarantine 资产可浏览、formal validator 不回退到 class label、
32-point smoke 可核验但 release 仍被明确拒绝。mock 演练可以验证代码路径，但产物必须进入独立 config 或
quarantine，不能与 WebArena expert 数据合并。

readiness 不是“已有 32 个点就可以训练”。审计要求 grounding、truth、formal SFT、train/dev/test、formal DPO
等 non-vacuous gate 均非空；同时验证 point/truth/prompt/candidate 精确 join、assistant answer
与 `meta.risky_raw_action` 两类 bid 可见性，并只从
single 与 multiturn canonical distilled 文件联合计算 teacher coverage。当前 point/truth/SFT、
formal DPO body、teacher 和相关完整性闸门已通过；当前共有 13 个 blockers：5 个 formal
release/split gate，以及第三环境、长 history、中间 state-changing supervision、negative solver
union、candidate taxonomy、paired opinions、≥200 live mutation census 和 API/DB signal 等 8 个
research-evidence gate，所以 formal readiness 明确为 false。

formal export 的最小验收条件：

1. 每条样本精确引用一个 `probe_point_id`；
2. point、candidate 各自与 manifest 1:1，且 point↔candidate 的 state/action/snapshot join 完整；
3. assistant answer 与 pinned `risky_raw_action` 的 `click(bid)` 都在紧邻 observation 可见；
4. `prediction_source=probe_transition`、`undo_source=probe_point_id`；
5. `is_mock=0`、`collector_success=100%`、split overlap=0；
6. tokenizer dry-run drop=0，canonical formal teacher coverage 达到预注册阈值，fallback 单列。

`collector_success=true` 与 `history_source=trajectory` 已通过 COMPLETE run manifest、meta、raw trajectory、
source key-state 的 exact closure；`formal_point_reached_states.jsonl` 只是 derived view，不重复计作 raw collection。
当前 24 条 active trajectory-conditioned 样本以及全部 62 条 formal 输入的历史均为 1 步；canonical episode-trace evidence 仍为空，因此不能声称已验证长历史或真正多步训练。
