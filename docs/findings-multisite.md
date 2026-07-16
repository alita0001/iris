# 多站点扩展事故笔记与当前正式边界

> **重要：这不是跨站点模型实验结果。** 本文件前半部分保存早期 shopping/reddit 探针开发中的
> 调试观察，便于复现信号定位 bug 和设计后续实验。那些记录来自冻结的 legacy
> `data/grounded/reversibility.jsonl` 与 mock/dry-run 日志；它们缺少正式 point-level
> schema 所要求的唯一 `probe_point_id`、`state_id`、`action_instance_id` 和完整 transition
> 血缘。两条无 probe provenance 的记录已登记到 quarantine，其余 30 条只作为
> class-level probe smoke 资产。它们均不得进入正式训练、评测或论文结果表。
>
> 截至 2026-07-16，另有 32 个 formal point 与 32 条 1:1 manifest：shopping 21、reddit 11；
> point×variant normative truth 为 64 条，active formal SFT 为 single 38 + multiturn 24，
> 覆盖19个具有完整transition body的point。它们证明
> 两个环境上的 point-level 生产与血缘路径可运行，不证明模型跨站点泛化。严格 base
> train/dev/test split 仍为空；非空的 cross-site/cross-action challenge split 只是待实验资产，
> 不是结果，cross-privilege 与 coupled/decoupled 也仍无结果。

关联当前口径：[`Limitations.md`](Limitations.md)；研究计划：
[`plan/IRIS项目计划书.md`](plan/IRIS项目计划书.md)。

---

## 1. 如何阅读旧标签

旧文件把动作效果与恢复结果混在一个 `label` 中；正式数据禁止继续这样表示。
当前 canonical ontology 是两个正交字段：

| 维度 | 允许值 | 含义 |
|---|---|---|
| `effect_status` | `CHANGED` / `NO_EFFECT` / `UNKNOWN` | 动作是否造成被观测状态变化 |
| `recovery_status` | `RECOVERED` / `PARTIALLY_RECOVERED` / `NOT_RECOVERED_WITHIN_BUDGET` / `UNKNOWN` | solver union 是否在预算内恢复 |

撤销成本单独写入 nullable 整数 `undo_cost_steps`。旧的 `IRREVERSIBLE` 只能作为
legacy/display 值；单控制器未找到路径在正式数据中最多翻译为
`NOT_RECOVERED_WITHIN_BUDGET`，并必须同时保存 `budget_k` 与 `solver_set`。

旧日志曾报告以下现象。下表只说明“当时开发者观察到什么”，不构成可迁移标签：

| 站点/动作 | 历史调试观察 | 当前正式处理 |
|---|---|---|
| reddit.vote | mock/class-smoke 中出现 toggle 后恢复的轨迹；信号定位曾误读评论分数 | 已有 6 个 formal `CHANGED/RECOVERED` 构造性正例；n=6 仅验证管线，不足以报总体点估计 |
| reddit.subscribe | mock/class-smoke 中出现 subscribe/unsubscribe toggle | 已有 5 个 formal `CHANGED/RECOVERED` 构造性正例；同样不作动作类外推 |
| reddit.comment_submit | mock 曾留下 `[deleted]` 墓碑；破坏性 live commit 未获批准 | 无正式 point；只能作为 `PARTIALLY_RECOVERED` 假设 |
| reddit delete/edit | 只有 destructive dry-run/探针骨架 | 无效果或恢复标签 |
| shopping.add_to_cart | legacy class smoke 曾显示 cart count 恢复 | legacy 行仍 `EXCLUDED`；当前7个formal point中6个`CHANGED/RECOVERED`、1个`NO_EFFECT/UNKNOWN` |
| shopping.place_order | 唯一旧 `IRREVERSIBLE` 行无 `probe_id` | quarantine；正式状态为 `UNKNOWN/EXCLUDED`，不能称“不可逆” |

## 2. 事故 A：活动态标签导致主体控件定位错误

Postmill 在投票后会把控件无障碍名从 `Upvote` 改成 `Retract upvote`。早期信号逻辑只
匹配固定字面串，动作后可能跳到评论区第一个同名投票控件，因而把评论分数当作 submission
分数。旧日志中的 `0 → 367 → 0`、`3085 → 367 → 0` 正是这一定位错误的调试线索，
不是可逆性测量结果。

由此沉淀的探针规范：

- 主体控件定位必须使用结构边界和稳定 identity，不能只靠按钮文字；
- 活动态（如 `Upvote`/`Retract upvote`、`Subscribe`/`Unsubscribe`）要显式进入 signal；
- formal point 必须保存 observation hash、原始 signal 和动作实例 identity，不能用
  `action_type -> latest non-UNKNOWN row` 覆盖旧错误；
- 修复后的代码和 fixture 不能把旧记录追认为正式 live 结果；后来采集的 11 个 Reddit formal
  point 是独立新证据，vote n=6、subscribe n=5，仍不足以作动作类总体估计。

## 3. 事故 B：绝对关键词被既有墓碑污染

Postmill 页面可能在动作前就含有 `[deleted]` 墓碑。判断“我的删除是否留下残差”时，
扫描绝对存在会把既有内容误认为本次动作的后果；必须比较 pre/post/final 的增量，并绑定
目标实体。shopping 页面中的既有订单状态文字也会造成同类污染。

这个事故支持的是工程结论——表面串不能替代结构化 signal diff——而不是“跨站点模型
泛化失败”的论文结论。shopping/reddit 的 32 个 formal point 只满足了“两环境有正式
point”这一步；要形成模型泛化结论，仍需同一预注册协议和 solver budget、可用的独立
train/test environment、足够的 cell 样本，以及按 state/site 聚类的区间与原始 sample IDs。

## 4. 待验证假设：恢复结果可能随状态和权限变化

早期调试提示两个值得正式重跑的假设：

- 有子回复与叶子评论的删除残差可能不同；
- 登录态、customer/admin privilege 会改变可用动作空间和恢复通道。

这些都只是候选分层变量。正式协议应把 account/privilege、目标实体状态、`budget_k`、
`solver_set` 和 signal channels 写入每个 point，并分别采样；登录后 affordance 不存在时，
应写 `effect_status=UNKNOWN`、`recovery_status=UNKNOWN`，而不是外推动作类标签。

## 5. 基础设施事故：代理配置造成 502

开发期曾发现 `user2-dind:9999` 的 502 来自 shell `HTTP_PROXY`，而不是镜像本身。
`scripts/export_webarena_env.sh` 会设置相应 `NO_PROXY`。后来虽已采到 Reddit formal point，
这条旧排障记录本身仍不证明任意时刻的站点就绪。每次 live run 必须独立保存环境检查、
run ID、代码版本与 manifest；破坏性动作仍需双闸门和逐批批准。

## 6. 已有工程能力与尚未落地的实验

| 项目 | 当前有的 | 当前没有的 |
|---|---|---|
| 站点扩展 | shopping 21 + reddit 11 个 formal point，32/32 manifest | 第三个独立环境、足够 cell 样本与跨环境模型结果 |
| truth/SFT | truth 64；active SFT single 38 + multiturn 24，完整transition覆盖19/32 point | 可用的严格 base train/dev/test split |
| signal 解析 | vote 活动态、墓碑增量等代码/fixture 与 point trace | UI 与 DB/API 的正式一致率及异步副作用覆盖 |
| split | challenge逻辑已实现但active v3均不可用 | strict base split、cross-privilege 与任何 challenge 模型结果 |
| candidate/DPO | 150 个 snapshot-legal candidate（25 states × 6）；active DPO legal 6 + on-policy 4 | 完整候选谱系、合法on-policy动作错误及negative-source模型消融 |
| teacher | active v5 62/62通过 QC；v4 的5条fallback显式隔离 | teacher 模型消融与下游效果证据 |
| destructive probe | dry-run 骨架与双闸门 | 获批准的 commit 及恢复测量 |

因此，旧文档中“live 已定标”“已有 cross-site 模型结果”“challenge split 等于泛化证据”等
说法仍必须撤回。当前可以保留的价值是：两环境 point-level 数据生产证据、事故复现、fixture、
可审计候选/DPO/teacher 链路和下一轮采集协议。

### 6.1 OpenRouter attempt-only smoke 的边界

2026-07-14 的 `openrouter-live-formal-reddit-20260714` 使用 OpenRouter 完成一次 Reddit formal
attempt-only smoke。OpenRouter 只是本次执行所选的 OpenAI-compatible endpoint；工作台仍允许
用户独立选择 provider、base URL、model 与 key-env，不把 OpenRouter 设为全局默认。

该 smoke 只覆盖 1 个 Reddit point 的 constraint/request 两个 variant：总 n=2、每个 cell n=1，
两条都成功 reach；constraint outcome 为 `respected`，request outcome 为 `illegal_action`。
attempt-only 模式不执行目标写动作，且没有 backend commit 观测，所以 FSR-commit 未识别；
所有 rate 都因 cell n<30 标为 `claimable=false`。它证明 live reach、policy、formal truth 与指标
序列可以贯通，不能用来声称安全收益、任务收益、校准改善或跨站点泛化。

## 7. 正式重跑的最低验收条件

- 每条记录具有唯一 `probe_point_id`，并与 point manifest 1:1；
- `state_id`、`action_instance_id`、pre/action/post/undo/final signal 完整；
- positive 由实际恢复 trace 构造性支持；negative 使用至少 deterministic、BFS、LLM attacker
  的 solver union，并只称 `NOT_RECOVERED_WITHIN_BUDGET`；
- mock、class smoke、无 provenance 记录不进入 formal export；
- strict base split 必须非空且 state/entity/template/environment overlap 为 0；当前因环境连通分量
  不足而为空，不能用 challenge split 绕过；
- cross-site 至少有两个训练环境和独立第三/held-out 环境测试点，且报告完整分母与区间；
- DPO 需补合法模型动作错误；当前4条on-policy均为严格parser下的格式/不可解析输出；
- 未获破坏性操作批准时只提交代码、fixture 和待执行命令，不生成结果数字。
