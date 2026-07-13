# 多站点扩展事故笔记（legacy / non-formal）

> **重要：这不是跨站点实验结果。** 本文件保存早期 shopping/reddit 探针开发中的
> 调试观察，便于复现信号定位 bug 和设计后续实验。记录来自冻结的 legacy
> `data/grounded/reversibility.jsonl` 与 mock/dry-run 日志；它们缺少正式 point-level
> schema 所要求的唯一 `probe_point_id`、`state_id`、`action_instance_id` 和完整 transition
> 血缘。两条无 probe provenance 的记录已登记到 quarantine，其余 30 条只作为
> class-level probe smoke 资产。它们均不得进入正式训练、评测或论文结果表。
>
> 截至 2026-07-13，`data/grounded/probe_points.jsonl` 与
> `data/grounded/POINT_MANIFEST.jsonl` 都是 **0 行**；因此没有可报告的 live、cross-site、
> cross-action、cross-privilege 或 coupled/decoupled 实验结果。站点注册、探针代码和
> split 生成能力只是工程能力，不是实证。

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
| reddit.vote | mock/class-smoke 中出现 toggle 后恢复的轨迹；信号定位曾误读评论分数 | `EXCLUDED`；需以唯一 point 重跑并记录 pre/post/undo/final signal |
| reddit.subscribe | mock/class-smoke 中出现 subscribe/unsubscribe toggle | `EXCLUDED`；需 point-level 重跑 |
| reddit.comment_submit | mock 曾留下 `[deleted]` 墓碑；破坏性 live commit 未获批准 | 无正式 point；只能作为 `PARTIALLY_RECOVERED` 假设 |
| reddit delete/edit | 只有 destructive dry-run/探针骨架 | 无效果或恢复标签 |
| shopping.add_to_cart | legacy class smoke 曾显示 cart count 恢复 | `EXCLUDED`；不是正式 `RECOVERED` point |
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
- 修复后的代码和 fixture 只能证明解析器行为，不能把旧记录追认为正式 live 结果。

## 3. 事故 B：绝对关键词被既有墓碑污染

Postmill 页面可能在动作前就含有 `[deleted]` 墓碑。判断“我的删除是否留下残差”时，
扫描绝对存在会把既有内容误认为本次动作的后果；必须比较 pre/post/final 的增量，并绑定
目标实体。shopping 页面中的既有订单状态文字也会造成同类污染。

这个事故支持的是工程结论——表面串不能替代结构化 signal diff——而不是“跨站点模型
泛化失败”的论文结论。要形成后者，仍需：

1. 两个以上环境的正式 point；
2. 同一预注册协议和 solver budget；
3. 独立 train/test environment；
4. 按 state/site 聚类的区间与原始 sample IDs。

## 4. 待验证假设：恢复结果可能随状态和权限变化

早期调试提示两个值得正式重跑的假设：

- 有子回复与叶子评论的删除残差可能不同；
- 登录态、customer/admin privilege 会改变可用动作空间和恢复通道。

这些都只是候选分层变量。正式协议应把 account/privilege、目标实体状态、`budget_k`、
`solver_set` 和 signal channels 写入每个 point，并分别采样；登录后 affordance 不存在时，
应写 `effect_status=UNKNOWN`、`recovery_status=UNKNOWN`，而不是外推动作类标签。

## 5. 基础设施事故：代理配置造成 502

开发期曾发现 `user2-dind:9999` 的 502 来自 shell `HTTP_PROXY`，而不是镜像本身。
`scripts/export_webarena_env.sh` 会设置相应 `NO_PROXY`。这是一条排障记录，不证明当前
站点就绪，也不证明任何 probe 已 live 执行。每次 live run 仍须独立保存环境检查、run ID、
代码版本与 manifest；破坏性动作仍需双闸门和逐批批准。

## 6. 已有工程能力与尚未落地的实验

| 项目 | 当前有的 | 当前没有的 |
|---|---|---|
| 站点扩展 | shopping/admin/reddit 注册与 probe fixture | 可报告的多站点正式 point |
| signal 解析 | vote 活动态、墓碑增量等代码/fixture | UI 与 DB/API 的正式一致率 |
| split | 多站点数据存在时可生成 held-out-site split 的代码 | 两站正式训练数据及 cross-site 结果 |
| candidate | S4 v2 共 300 条、50 states，均为 snapshot-legal a11y proposal | point label、on-policy error、candidate→probe→DPO 闭环 |
| destructive probe | dry-run 骨架与双闸门 | 获批准的 commit 及恢复测量 |

因此，旧文档中“live 已定标”“跨站点判决数据”“test_cross_site 已有真实内容”等说法
全部撤回。当前可以保留的价值只有：事故复现、fixture、候选假设和下一轮采集协议。

## 7. 正式重跑的最低验收条件

- 每条记录具有唯一 `probe_point_id`，并与 point manifest 1:1；
- `state_id`、`action_instance_id`、pre/action/post/undo/final signal 完整；
- positive 由实际恢复 trace 构造性支持；negative 使用至少 deterministic、BFS、LLM attacker
  的 solver union，并只称 `NOT_RECOVERED_WITHIN_BUDGET`；
- mock、class smoke、无 provenance 记录不进入 formal export；
- cross-site 至少有两个环境的正式训练点和第三/held-out 环境测试点，且报告完整分母与区间；
- 未获破坏性操作批准时只提交代码、fixture 和待执行命令，不生成结果数字。
