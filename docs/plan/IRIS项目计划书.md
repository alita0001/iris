# IRIS 项目计划书 v1.3：从 legacy pilot 到 point-grounded benchmark

> 更新日期：2026-07-13
> 本文是“研究目标与当前证据”的唯一状态说明。凡标为“计划”的项目，不得在论文、README、Dataset Card 或答辩中写成已经完成。

## 0. 结论先行

IRIS 当前是一个经过语义整改的研究工程骨架，不是已经完成的 benchmark，也没有可以提交的安全效果结论。

- 已实现：版本化 point-level schema、legacy 隔离、统一输入 builder、anchored AXTree pruning、训练/评测 fail-closed validators、prompt 快照、split/leakage 与指标代码。
- 当前正式数据：`data/grounded/probe_points.jsonl` **0 行**，`POINT_MANIFEST.jsonl` **0 行**。
- legacy 数据：`reversibility.jsonl` 32 行、`MANIFEST.jsonl` 30 行；2 条无 probe provenance 的记录已进入 quarantine，其余 30 条只登记为 class-level smoke probes，全部禁止升级为正式监督。
- 历史训练资产：single SFT 92、multi SFT 62、single DPO 230、multi DPO 155；它们没有 point-level foreign key，仍是开发/审计资产，不是正式 IRIS 训练集。
- teacher distilled：0 行。
- 当前训练环境分布：single SFT 92/92 为 Magento shopping。multi 的 12 mock / 50 WebArena 只能按 `trajectory_id` 前缀推断；62/62 均缺显式 `environment_origin/is_mock/collector_success/run_id`，重复 logical ID 的历史 success 还可能冲突，因此不能逐样本证明成功或失败，formal gate 必须拒绝。
- 当前没有：正式 live point、正式模型训练结果、cross-site/cross-action/cross-privilege 实验结果、coupled-vs-decoupled 实现或结果、在线 RL 结果。
- 旧 pilot 的“FSR 9.1%”已撤回：旧代码把所有 constraint episode 当分母，唯一计数失败是可恢复的 add-to-cart；它不是论文定义下的 true-danger FSR。

复核命令：

```bash
wc -l data/grounded/{reversibility,MANIFEST,probe_points,POINT_MANIFEST}.jsonl
wc -l data/train/sft/*.jsonl data/train/dpo/*.jsonl
pytest -q
```

## 1. 研究问题与严格边界

IRIS 研究的是：在明确的 agent 动作空间、权限、步数预算、观测信号和 undo solver 集合下，一个状态转移是否被实际恢复，以及这种证据能否被模型消费。

它不研究、也不得声称已经解决普适的“web-agent safety”。恢复某个状态投影不等于撤回现实伤害：邮件可能已送达，支付可能已结算，第三方可能已看到内容，审计日志、缓存和声誉影响也可能仍然存在。

正式构念写作：

\[
R(s,a \mid A_{agent}, privilege, k, \sigma, \mathcal C)
\]

- `A_agent`：部署 agent 实际可执行的动作空间；
- `privilege`：账户和权限；
- `k`：undo 搜索预算；
- `σ`：被比较的状态信号；
- `C`：实际运行的 solver 集合。

### 1.1 正交标签本体

正式 schema 不再把 effect 与 recovery 混成一个枚举。

`effect_status`：

- `CHANGED`
- `NO_EFFECT`
- `UNKNOWN`

`recovery_status`：

- `RECOVERED`
- `PARTIALLY_RECOVERED`
- `NOT_RECOVERED_WITHIN_BUDGET`
- `UNKNOWN`

`undo_cost_steps` 是独立 nullable 整数。`IRREVERSIBLE` 只作为 legacy 读取兼容值；它映射到 `NOT_RECOVERED_WITHIN_BUDGET`，不能由有限预算、单一控制器失败升级成数学意义的不可逆证明。

### 1.2 证据方向不对称

- `RECOVERED` 有构造性证据：保存并重放实际 undo sequence，最终 σ 与 pre-state 相符。
- `PARTIALLY_RECOVERED` 有补偿 trace 和 residual diff。
- `NOT_RECOVERED_WITHIN_BUDGET` 只表示记录的 solver 集在预算内没有找到恢复路径；它不是“不存在其他路径”的证明。
- controller-human agreement 只能作为 QC，不能证明 existential complement。

### 1.3 Recoverability 与 normative risk 分离

一个动作可以可恢复但违反用户约束，也可以难恢复但经用户明确确认。正式 decision truth 至少包含两个独立轴：

1. 行为测得的 recoverability；
2. 来自任务/政策的 `normative_risk` 或 `policy_constraint_truth`。

论文不得再从 recovery label 单独推导“安全/危险”。

## 2. 正式 point-level grounding 协议

### 2.1 Admission schema

正式点必须含唯一、可审计的：

- `probe_point_id / probe_run_id / probe_name`
- `state_id / action_instance_id / raw_action / canonical_action`
- `site / environment_family / environment_instance / is_mock`
- `task_id / trajectory_id / run_id / seed / url`
- `account / privilege / budget_k / solver_set / controller_version`
- `pre_observation_hash / pre_signal`
- `post_observation_hash / post_signal`
- `undo_actions / undo_observation_hashes / final_signal`
- `effect_status / recovery_status / undo_cost_steps / residual_diff`
- `budget_exhausted / timestamp / code_version / evidence`

正式 body 与 manifest 必须按 ID 和内容 hash 1:1；训练样本只能通过唯一 `probe_point_id` join，禁止 `action_type -> latest label`。

### 2.2 当前迁移结果

迁移脚本没有伪造 provenance，也没有把 legacy 行“补齐”为正式点：

- legacy line 1–2：进入 `data/grounded/quarantine/legacy_rows.jsonl`；
- 其余 30 行：进入 `class_probe_smoke_index.jsonl`，`formal_eligibility=EXCLUDED`；
- 正式 point 和 point manifest 均为空；
- 原 `reversibility.jsonl` 和 `MANIFEST.jsonl` 保留不改。

因此当前正式训练集必须为空；这是正确的 fail-closed 结果，而不是流水线失败。

### 2.3 正式负标签计划

当前代码主要拥有站点专用 deterministic controller。正式负点还需 solver union：

1. site-specific deterministic controller；
2. 当前页面 affordance BFS，深度 2–3；
3. 强 LLM undo attacker，多 seed，在同一预算内搜索；
4. 可用时增加 API/DB diff，以增强 σ 覆盖。

任一路恢复成功即翻为 positive。正式报告必须给出原负点被 attacker 攻破率。DB diff 加强“是否恢复”的观测，但不证明搜索完备。

## 3. 环境与外部效度

代码注册了 shopping、shopping_admin 和 reddit，但这不等于有三个环境的正式数据。

| 范围 | 当前事实 | 可声称程度 |
|---|---|---|
| Magento shopping | 有 legacy smoke probes 和历史训练资产 | 仅可说明工程 pilot |
| Magento admin | probe 代码/安全闸门存在，正式 committed points 为 0 | 不能声称 cross-privilege 结果 |
| Postmill reddit | 有少量 legacy smoke rows，正式 points 为 0，训练为 0 | 不能声称 cross-site 泛化 |
| GitLab/WorkArena/Stripe | 未进入当前数据 | 仅是扩展计划 |

`cross_action_class`、`cross_privilege` 和 `cross_site` 是不同的泛化轴，不能替代环境多样性。优先扩展顺序：WebArena GitLab → WorkArena/ServiceNow → 外部副作用沙盒（如 Stripe test mode）→ VisualWebArena。Mind2Web 是静态语料，不能执行 undo，只能作候选发现或意见/OOD 辅助集。

当前 split 过程可以物化 0 行的 `*_cross_<axis>*.jsonl` 作为清除旧 shard 的 sentinel；
只有 `SPLIT_REPORT.challenges.<axis>.available=true` 且对应 train/test 都非空，才能称为
可用 challenge split 或实验结果。

## 4. 输入、输出与证据可见性

### 4.1 已实现的统一输入

`revact/prompts.py::build_policy_messages` 是训练和部署唯一 serializer：

```text
system
user: <goal> + <history> + <observation>
assistant: IRIS completion
```

- history 统一保留最后 `POLICY_HISTORY_STEPS=6` 个条目；multiturn 路径从连续 observation 重算 observed delta，single reached-state 路径目前只信任 `history_source=trajectory` 声明，尚未做 raw-run/hash exact join；
- 每条 history 为 action + `[nav/state-change/update/no-effect]` + delta；
- AXTree 使用 action-anchored pruning；
- supervised click bid 若不在输入中，assembler/trainer hard gate 拒绝样本；
- formal trajectory sample 代码要求 `is_mock=false` 与 `collector_success=true`；single 路径当前仍是字段 gate，只有与不可变 collection run 精确连接后才能称为成功 provenance。

历史 multi 文件仍是旧 chat topology：62 条中只有 12 条当前 risky bid 可见。新 validator 会拒绝其余 50 条；未经 point-level 重物化不得训练。

### 4.2 正式输出目标

正式 point-grounded 输出版本为 `iris.v3`；`iris.v2` 仅用于读取历史资产。结构化完成为：

```text
<think>
<observation>...</observation>
<reasoning>...</reasoning>
<prediction>...</prediction>
<rev_check>...</rev_check>
<reversibility>...</reversibility>
<undo>...</undo>
<decision>...</decision>
</think>
<answer>...</answer>
```

但正式目标必须满足：

- `<prediction>` 来自 point 的真实 pre/action/post diff，不来自 action-type 手写模板；
- `<undo>` 来自实录 `undo_actions`，手写 hint 只能是 legacy explanation；
- `<rev_check>` 明确区分当前输入可见证据、post-action probe evidence 和模型预测；
- 样本 meta 记录 `prediction_source=probe_transition`、`undo_source=probe_point_id`；
- 每个 state-changing/decision assistant turn 显式标类型并接受完整格式检查。

当前正式 point 为 0，所以这些规则已有代码 gate，但没有正式样本可验证模型效果。

## 5. S1–S8 工程状态

| 阶段 | 当前实现 | 当前数据/证据 | 状态 |
|---|---|---|---|
| S1 环境与日志 | 新 writer 使用不可变 run 血缘和 IN_PROGRESS/COMPLETE manifest | 旧 93 meta/31 logical ID 已做 index-only quarantine，原文件未改 | 新工程测试通过；历史仍禁止 formal join |
| S2 状态发现 | 全 legal control 枚举；关键词仅作 legacy/ranking hint | 241 legacy key-state rows 仍是 210/29/2；live mutation census 未完成 | detector 骨架已改，live recall blocked |
| S3 状态到达 | reach/scale 代码存在 | 主要 Magento | pilot only |
| S4 候选动作 | snapshot 交互控件枚举、精确 bid 合法性、版本化 schema、300 行 1:1 hash manifest | v3 产物 300 条/50 状态（250 a11y + 50 producer-declared `source=expert`）；后者无成功 collector provenance；无 decoy/point label/on-policy error | 完整性 gate 已落，部署分布未闭环 |
| S5 grounding | 新 point schema、migration、probe registry | formal points=0 | schema landed，数据 blocked |
| S6 decision truth | normative risk 与 recovery 已分轴 | 无正式 point truth table | code only |
| S7 teacher | evidence-aware formal QC 已加 | distilled=0 | 无训练资产 |
| S8 assembly/train | formal validators、SFT/DPO/offline RLVR 入口 | legacy SFT/DPO 不能晋升 | code only |

## 6. S4 候选动作

已实现 `iris.candidate.v2` schema、当前 AXTree 交互元素枚举、snapshot hash、精确 bid 合法性闸门和每状态 4–6 候选的工作台物化。formal consumer 使用 `data/raw/candidates/iris_candidates.v3.jsonl`：300 条、50 个 state，其中 `source=a11y_enumeration` 250 条、producer-declared `source=expert` 50 条；后者只表示 producer 传入的 target bid，candidate v2 不含 trajectory/run/collector-success provenance。`CANDIDATE_MANIFEST.jsonl` 以 300 行逐条哈希保护 body。grounding import、训练和 export 会重新验证 point 的 source candidate 与偏好负例的 `negative_candidate_id`，而不是只相信字段名。类别只是覆盖提案，不是风险、恢复标签或专家质量证明。

每状态目标覆盖：

- 专家动作；
- AXTree 可交互元素枚举；
- constraint-trigger 和 goal-violating 动作；
- decoy：词面高危但可撤回或尚未 commit；
- 信息不足、需要 VERIFY 的动作；
- 当前策略的 on-policy 错误。

LLM 只能提案。合法性由 snapshot bid/role 验证；effect/recovery label 必须由 execute-then-undo 产生。当前 v3 候选尚未与 formal point 关联，也未包含 decoy 或模型 on-policy 错误。旧 v1/v2 候选文件只作开发资产，不进入 formal DPO。

当前 DPO negative 按生成机制都是文本/标签翻转，但历史 single 230 与 multi 155 行均未显式记录 `negative_source`；legacy audit 因此将其视为 `legacy_unspecified`/non-deployment，不得伪造为行级 `synthetic_flip` provenance。未来 formal ablation 才必须显式标记 `negative_source=synthetic_flip`。formal DPO gate 要求 legal/on-policy negatives 至少占 50%，因此在 candidate→point→DPO 闭环未完成时会正确拒绝训练。

## 7. 训练算法的诚实定位

### 7.1 SFT

代码实现 completion-only loss 和 formal turn-type/input/provenance gates。当前无正式 point-grounded SFT，不能报告效果。

### 7.2 DPO

四类 synthetic preference pair 仍可作开发 fixture，但不匹配部署错误分布。正式数据需加入 legal candidate 和 on-policy errors，并按 negative source 分层做 ablation。

### 7.3 GRPO/GSPO

`revact/train/grpo.py` 是固定 prompt 上的 **offline RLVR ablation**：使用可验证格式、decision、recovery 和约束 reward。环境不在训练回路中，它不能声称治疗 step-2 二次暴露，也不是 VAGEN 式多轮在线 RL。

只有在 point labels、输入、SFT/DPO 和指标通过后，才考虑有事务回滚/克隆账户的 environment-in-the-loop RL。不得用真实订单作为训练 rollout。

### 7.4 Coupled vs decoupled

这是待检验问题，不是现有贡献。当前仓库没有三 LoRA verifier/selector、capacity-matched trainer 或 paired cost logs。

若实现，必须满足：

- 相同 base checkpoint、point labels 和 evidence packet；
- 参数量差不超过预注册阈值，train FLOPs 和 token budget 对齐；
- 同时报 calls、tokens、latency、任务效果和校准；
- 增加 single-backbone multi-head 对照。

做不完则从论文贡献中删除。

## 8. 评测与假设

### 8.1 已实现但尚无正式结果的指标代码

- FSR-declaration / FSR-attempt / FSR-commit；
- FBR、IER、UCR；
- recovery macro-F1/per-class；
- undo success 与 undo-cost MAE；
- Brier、ECE、risk-coverage；
- Wilson/Bootstrap 区间、UNKNOWN bounds 和 label-noise sensitivity。

每个指标必须保存 numerator/denominator counts 及 sample IDs。任何 cell `n<30` 只列计数和区间，不作效果结论。

### 8.2 旧 pilot 降级

旧日志可证明 rollout harness 曾运行，并出现过一次“第一步声明 AVOID、下一步又点击同类控件”的案例。它不能证明：

- FSR 改善；
- safety；
- recovery 泛化；
- calibration；
- coupled 优势；
- 在线 RL 必要性。

旧 `9.1%` 只能被称为当时实现下的 constraint-action attempt rate，不能称为 FSR。base 的 `0%` 伴随大量不可解析/无效动作，也不能解释为安全。

### 8.3 可证伪假设

- H1：在同动作文字、同 one-step outcome、只切换 undo channel/权限的 matched pairs 上，forward-only 与 recovery-grounded 模型是否分离；当前未测。
- H2：grounded recovery supervision 是否在不增加 FBR 的前提下降低 false-safe；当前无正式模型/数据。
- H3：同监督和计算下 coupled 与 decoupled 的成本/泛化/校准差异；当前无实现。

按钮文字 logistic 在当前少数动作类上会形成天花板，不能单独检验 H1。正式 H1 需要机制干预 matched pairs，并按 state/site cluster。

## 9. Prompt 与数据治理

- registry 覆盖当前 agent/collector/teacher/目标模板；
- 新 assemble 保存 content-addressed 全文 bundle 和父版本，并以独立 `prompt_generation_fp` 记录 producer/model/decode；single/multiturn/teacher 代码均已写入该字段，但 formal rows=0，尚无非空数据证明这条血缘；
- 历史 SFT 的唯一 `prompts_fp=259fceac4e0c` 目前没有对应全文 bundle，不得用当前 prompt 冒充；
- 工作台可编辑 prompt；formal probe authoring 不能直接编辑 label，legacy grounding 页仍有 `reversibility_override` 人审字段，仅用于冲突/QC，不能改写 formal truth；
- 工作台代码已支持 single/multiturn SFT/DPO 浏览和导出；当前显示的仍是历史资产，因 point=0 无法展示完整 formal lineage；
- teacher 改坏后必须被 input/evidence/label/decision consistency QC 拒绝；
- mock、失败 trajectory、legacy hints 和 synthetic negatives 都必须有显式 source 标记。

## 10. 投稿前硬门槛

1. 正式 point/manifest 1:1，缺 provenance 为 0；
2. 每个正式样本唯一回链 point transition；
3. supervised bid 可见率 100%；
4. formal train/eval 中 mock=0、失败 expert=0；
5. 每个主表 cell 至少 30 个独立 state groups；
6. 至少三个独立后端家族进入评测；
7. 负点报告多 solver 攻破率；
8. teacher coverage 和事实一致率达到预注册阈值；
9. H1 matched mechanism intervention 完成；
10. 若声称 coupling，公平对照和成本日志必须完成。

在这些门槛前，项目定位应是“measurement/workbench prototype”，不是完成的 main-track 方法论文。

## 11. 当前允许与禁止的论文措辞

允许：

- “We implement a fail-closed point schema and workbench surfaces for solver- and signal-relative recovery evidence; the formal live producer is not yet wired.”
- “Legacy class-level probe records are quarantined and excluded from formal supervision.”
- “Positive recovery labels are intended to be constructive execution evidence.”

禁止：

- “IRIS makes web agents safe.”
- “The current dataset proves actions are irreversible.”
- “We have 32 formal grounded points.”
- “We demonstrate cross-site/cross-privilege generalization.”
- “We demonstrate coupled-vs-decoupled superiority.”
- “GRPO closes multi-step rollout shift.”
- “Pilot FSR is 9.1%.”

完整限制见 [`../Limitations.md`](../Limitations.md)，数据发布声明见 [`../../DATASET_CARD.md`](../../DATASET_CARD.md)。
