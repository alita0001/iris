# IRIS 数据集构建工作台（workbench）

> 本文描述当前工程能力，不把按钮、adapter 或历史文件当成已完成实验。新手可先读
> [`tutorial.md`](tutorial.md)；构念与数据边界见 [`Limitations.md`](Limitations.md)。

工作台是 CLI pipeline 的浏览与编排界面。生成、校验和导出逻辑仍在
`revact/{data,grounding,train,eval}`；UI 不能绕过 formal admission gates，也不能靠人工
覆核把意见写成行为标签。

## 当前数据闸门（2026-07-13）

| 资产 | 实测状态 | 用途边界 |
|---|---:|---|
| formal `probe_points.jsonl` / `POINT_MANIFEST.jsonl` | 0 / 0 | 尚无可训练的 point-grounded supervision |
| legacy grounding body / manifest | 32 / 30 | 2 条 quarantine；30 条 class-smoke，全部 formal-excluded |
| S4 canonical candidate / manifest | 300 / 300（50 states，schema `iris.candidate.v2`） | v3 body + 1:1 hash manifest；无 effect/recovery/safety label |
| historical single SFT | 92 | Magento-only legacy 资产，不是 formal set |
| historical multi SFT | 62 | 按 `trajectory_id` 前缀推断 12 mock / 50 WebArena；62/62 缺显式 origin/mock/success/run provenance，不得 formal export |
| formal teacher distilled | 0（文件尚未物化） | 不得声称 teacher 已生效 |

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

## 各页真实度

| 页面/能力 | 当前状态 | 不能据此声称什么 |
|---|---|---|
| Pipeline / Jobs | 可编排安全 CLI、显示产物与日志 | job 成功不等于数据通过 formal gate |
| 成功轨迹 | 可浏览历史 trace；未来 run 使用 immutable `run_id` | 旧 93 行 meta 不是 93 条独立轨迹 |
| 关键状态 | 未来采集枚举 legal controls；旧 241 states 来自三类英文关键词 | 没有任务无关 detector 的 live recall 结果 |
| 候选动作 | v3 body + 300 行 manifest 已物化；schema v2，50 states，bid/snapshot 合法 | category 是 proposal，不是 label；无 on-policy errors |
| Probe authoring | declarative spec 可选动作、signals、undo、预算、安全等级；无 label 字段 | spec 保存或 fixture 通过不等于 live point |
| Grounding | 可分别浏览 empty formal point 与 legacy/quarantine | 32 条旧记录不是正式 point dataset |
| Teacher | 可运行 evidence-aware QC 与隔离 fallback | 当前 distilled=0 且 formal source=0，覆盖率未定义（`null`） |
| Dataset browser/export | 支持多数据族全文/血缘；formal export fail closed | formal point=0 时不可能导出非空正式训练集 |
| Eval | 有 versioned truth schema 与静态 metric primitives | historical rollout 分母错误；当前没有 formal eval artifact 或新实证结果 |
| Train dry-run | SFT/DPO/offline RLVR validator 可审计输入 | 没有模型训练结果；GRPO 不是 environment-in-loop RL |

## S4 candidate 的精确边界

formal consumer 使用 `data/raw/candidates/iris_candidates.v3.jsonl`：300 条
`iris.candidate.v2`、覆盖 50 个 state，来源为 250 条 `a11y_enumeration` 与 50 条
producer-declared `source=expert` target-bid proposal。后者没有 trajectory/run/
collector-success provenance，不能称为成功专家示范。每条记录保存
candidate/state/bid/canonical action/category/source、
snapshot hash 与 snapshot legality，并由 300 行 manifest 做 1:1 哈希校验。active v3
没有 decoy/on-policy row；只有实际执行并关联唯一 `probe_point_id` 后，才能获得
effect/recovery 字段。旧 v1/v2 文件只作开发资产。

LLM 若用于提案，只能影响 candidate 排序或补充候选；bid 合法性由当前 snapshot 验证，
grounding 由行为探针产生。正式 DPO 尚未满足“至少 50% legal/on-policy negatives”闸门。

## 多站点与 live

代码注册了 shopping、shopping_admin、reddit，并包含 site-aware probe/split 组件。这只表示
工程可扩展。当前 formal point 为 0，也没有两站正式训练数据，因此：

- 不得生成或报告 cross-site 结果；
- cross-action/cross-privilege split 的代码存在不等于实验完成；
- `*_cross_site*.jsonl` 等路径可以 0 行形式存在，用于清除旧 shard；只有
  `SPLIT_REPORT.challenges.<axis>.available=true` 且 train/test 非空才表示可用 challenge；
- reddit fixture/class-smoke 不得写成 live 定标；
- destructive probe 只有 dry-run/双闸门，未经批准不执行。

历史调试事件见 [`findings-multisite.md`](findings-multisite.md)，该文件已明确标为
legacy incident notebook。

## 验证与导出

```bash
python -m pytest -q
# 默认 fail closed：当前 formal=0，以下命令应输出 ready=false 并返回 1
python scripts/audit_formal_readiness.py
# 只在需要保存阻塞报告时放宽进程退出码；报告仍是 ready=false
python scripts/audit_formal_readiness.py --allow-blocked --output docs/audit/IRIS-formal-readiness.json
python -m revact.cli train --dry-run
python -m revact.cli train-dpo --dry-run
python -m revact.cli train-grpo --dry-run
```

当前仓库的正确结果应是：历史/quarantine 资产可浏览、formal validator 不回退到 class label、
empty formal dataset 被如实报告。mock 演练可以验证代码路径，但产物必须进入独立 config 或
quarantine，不能与 WebArena expert 数据合并。

readiness 不是“所有计数为 0 且没有报错”。审计要求 grounding、formal SFT、train/dev/test
五个 non-vacuous gate 均非空；同时验证 point/truth/prompt/candidate 精确 join、assistant answer
与 `meta.risky_raw_action` 两类 bid 可见性，并只从
`data/train/formal/iris_sft_distilled_point_v1.jsonl` 计算 teacher coverage。当前这些
non-vacuous gate 均未通过，所以 formal readiness 明确为 false。

formal export 的最小验收条件：

1. 每条样本精确引用一个 `probe_point_id`；
2. point、candidate 各自与 manifest 1:1，且 point↔candidate 的 state/action/snapshot join 完整；
3. assistant answer 与 pinned `risky_raw_action` 的 `click(bid)` 都在紧邻 observation 可见；
4. `prediction_source=probe_transition`、`undo_source=probe_point_id`；
5. `is_mock=0`、`collector_success=100%`、split overlap=0；
6. tokenizer dry-run drop=0，canonical formal teacher coverage 达到预注册阈值，fallback 单列。

其中 `collector_success=true` 与 `history_source=trajectory` 目前仍只是 single reached-state
样本的声明式字段 gate；非空 formal 发布前还必须把它们与不可变 collection run、连续 observation
和内容 hash 精确连接。multiturn builder 已从连续 observation 计算 history delta。
