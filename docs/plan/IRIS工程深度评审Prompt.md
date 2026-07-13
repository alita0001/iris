# IRIS 工程深度评审 Prompt

> 用法：把下面「PROMPT 正文」整段复制给 Claude（推荐在本仓库 `/workspace/iris` 内用 Claude Code 执行，评审者可直接读代码核证；若在无仓库访问的对话中使用，评审者依据附录 B 的已核实事实清单作答，并明确标注哪些结论未经代码复核）。
> 事实清单核实日期：2026-07-13，若仓库已更新请让评审者重新核实。

---

## PROMPT 正文

### 0. 角色与总目标

你同时扮演三个角色，对 IRIS 项目（论文 idea + 数据集构建工作台工程）做一次**不留情面的深度评审**：

1. **顶会对抗性审稿人**（NeurIPS/ICLR/ICML Area Chair 级）：你的任务是把这篇论文拒掉——找出 idea 层面的构念缺陷、实验设计漏洞、与已有工作（VAGEN、WebGuard、SafePred、WMA/WAC、Qwen-AgentWorld、ST-WebAgentBench、Revisable by Design、Grinsztajn'21、Leave No Trace）的撞车点和不公平对照；
2. **资深 agent 训练工程师**（做过 WebArena/AgentOccam/UI-TARS/AgentTrek 级别的数据与训练流水线）：审查样本格式、输入输出规范、历史表示、训练算法选型是否符合当前主流实践，指出会导致训练失败或部署错位的工程缺陷；
3. **数据集质量审计员**：审查数据规模、分布、标签可信度、覆盖率、泄漏与混杂，判断这批数据能否支撑论文的每一条主张。

你的产出不是安慰性的总结，而是一份**可执行的问题清单与整改路线图**。项目作者已经知道这个工程"能跑"；他要知道的是**它在哪里会死、审稿人会从哪里开枪、以及每个洞怎么补**。

### 1. 项目一句话背景

IRIS 主张：只会前向预测 f(s,a)→s′ 的世界模型对安全行动不完备，策略内的世界模型还必须知道**该转移能否被 agent 自己撤销、代价多少**（可逆性）。做法：在 WebArena 站点镜像上用 **execute-then-undo 行为探测**产出 {REVERSIBLE, REVERSIBLE_WITH_COST(k), PARTIALLY_RECOVERABLE, IRREVERSIBLE, UNKNOWN} 谱系标签，把「前向预测 + 可逆性判断 + undo 计划 + 校准决策」蒸馏进单次自回归输出（`<observation>/<reasoning>/<prediction>/<rev_check>/<reversibility>/<undo>/<decision risk=s>/<answer>`，格式版本 `iris.v2`），并做 coupled（内在化）vs decoupled（外置 verifier）的受控对照。训练：SFT → DPO（四类反事实偏好对）→ GRPO/GSPO（离线可验证 reward，定位为 ablation）。

### 2. 评审必读材料（有仓库访问时逐一核对）

| 材料 | 路径 | 审什么 |
|---|---|---|
| 项目计划书 v1.2 | `docs/plan/IRIS项目计划书.md` | 每一条"已落地/将做"声明是否与代码和数据一致 |
| S1–S8 流水线文档 | `docs/plan/S1-S8完整流程文档.md` | 注意编号从 S3 直接跳到 S5——S4（候选动作生成）的真实状态 |
| 撞车文献分析 | `docs/plan/07-撞车文献汇总与改向依据.md` | novelty 声明是否站得住 |
| 关键状态挖掘 | `revact/data/collect.py`（`afforded_action_types`，注释自称 "shallow keyword match"）+ `revact/config.py` 的 `PILOT_ACTION_KEYWORDS`（约 L160） | 硬编码关键词、任务适配性 |
| 可逆性探针 | `revact/grounding/base.py`（`ProbeSpec`/`register`）、`revact/grounding/probes/{shopping,shopping_admin,reddit}.py`、`revact/grounding/undo.py`、`signals.py` | 探针注册机制、undo 控制器完备性 |
| 样本组装 | `revact/data/assemble.py`、`multiturn.py`、`revact/prompts.py`（prompt registry + history builder） | 输入/输出格式、单步 vs 多轮 |
| 训练 | `revact/train/{sft,dpo,grpo,distill}.py` | 训练算法谱系、reward 设计、QC 矛盾正则 |
| 工作台服务端 | `revact/server/{app,datasets,quality,jobs,annotations,export}.py`、前端 `revact/ui/views.js` | 候选动作 placeholder、数据集浏览器、prompt 可编辑性 |
| 评测 | `revact/eval/{decisions,rollout}.py` | FSR/FBR 口径、守护式 rollout |
| 数据现状 | `data/raw/state_bank/*.jsonl`、`data/grounded/*.jsonl`、`data/train/{sft,dpo}/*.jsonl`、`data/raw/trajectories_meta.jsonl` | 规模、分布、来源混杂 |

### 3. 评审纪律（违反任何一条即返工）

1. **证据强制**：每个 finding 必须给出 `文件:行号` 或数据统计命令与输出；不许写"可能""大概"这类无证据的泛泛之谈。有仓库时先跑统计脚本核实附录 B 的数字再引用。
2. **不采信自评**：计划书 v1.2 大量标注"P0/P1/P2 已落地"。你必须逐条验证"文档声称 vs 代码/数据实际"，专门开一节列出**声称与现实的落差**（例：teacher 蒸馏产物 `revact_sft_distilled.jsonl` 当前为 0 行；计划书 §6.7 规模表的 pilot 目标 vs `data/grounded/` 实际 32 条探测记录）。
3. **严重度分级**：每个问题标 P0（会杀死论文/使主张不成立）/ P1（审稿人必问、需在投稿前修复）/ P2（工程债，影响迭代速度）。
4. **攻击话术具体化**：P0/P1 问题各写一句**审稿人原话式**的攻击（英文，Weaknesses 小节风格），作者要能拿去做防御演练。
5. **解决方案可验收**：每个问题给出短期（1–2 周）与长期修法，并写明**验收标准**（什么数字/什么产物出现算修完）。
6. **区分两类结论**：「idea 本身的构念/定位问题」与「当前实现离 idea 还差多远的工程问题」分开陈述，不许混在一起互相稀释。

### 4. 十二个评审维度与必答问题

#### 维度一：环境覆盖与任务泛化（作者最担心的问题）

当前全部数据来自 WebArena 的 3 个站点镜像（shopping :7770、shopping_admin :7780、reddit :9999），真实 web 任务分布远宽于此。必答：

- 「agent-relative operational reversibility」这个构念本身是否站点无关？把 3 站点的结论外推到"web agent 安全"需要哪些附加假设？哪条假设最脆？
- 计划书用 cross_action_class / cross_privilege / cross_site(reddit) 三个 split 替代真正的环境多样性。这个替代在审稿人眼里能过关吗？reddit（Postmill）与 shopping（Magento）是否同质到 cross-site 证据力打折？
- 若要在不推翻现有流水线的前提下扩环境，给出优先级排序的具体方案：WebArena 其余站点（gitlab/map/wikipedia）、VisualWebArena、WorkArena/ServiceNow、Mind2Web 静态语料复标、真实沙盒站点（如 stripe test mode）各自的成本、探针可迁移性、与 execute-then-undo 协议的兼容度。
- 最低限度：论文 Limitations 该怎么写才能既诚实又不自杀？给出可直接粘贴的段落。

#### 维度二：关键状态挖掘机制（硬编码关键词匹配）

`collect.py::afforded_action_types` 用 `PILOT_ACTION_KEYWORDS`（仅 add_to_cart/place_order/delete_address 三类、英文关键词写死）在轨迹回放中打捞关键状态，产出 241 个状态、分布为 **add_to_cart 210 / place_order 29 / delete_address 2**——严重偏斜且类别由人工枚举封顶。必答：

- 这种挖掘方式的漏报率如何量化？（例如：同一批轨迹里被关键词漏掉的 state-changing 动作占比，能否用 a11y tree 的可交互元素 + 后端信号 diff 反推？）
- 换任务域（如 gitlab 的 delete branch、reddit 的 ban user）时关键词表完全失效——给出**任务无关**的关键状态发现机制设计：候选包括(a) 以 fingerprint/后端信号 diff 为判据的"执行后状态变化检测"回溯挖掘；(b) LLM 对 a11y tree 交互元素做 state-changing 概率标注 + 探针验证闭环；(c) 站点 sitemap/URL 模式聚类。评估各方案与"标签必须行为实测"铁律的兼容性。
- 210:29:2 的偏斜对 SFT/DPO 类别配比（计划书 §13.5 要求 EXECUTE 55–65%）和 Rev-F1 的 macro 平均有什么具体伤害？给出重采样/配比方案。

#### 维度三：探针体系的 soundness gap（作者认定的"大问题"）

探针只能证明「在**我们定义的动作类 + 我们写的 undo 控制器 + 我们观测的信号 σ** 下可逆/不可逆」，**不能证明不存在其他动作序列可恢复初始状态**——IRREVERSIBLE 标签本质是"预算 k=12 步内、该控制器未找到 undo 通道"，是 one-sided evidence。必答：

- 这个 gap 是否动摇定义 1（∃ a₁…a_j ∈ A^j, j≤k）的可证伪性？REVERSIBLE 方向由构造性证据（undo_actions 实录）支撑是 sound 的，IRREVERSIBLE 方向呢？审稿人问"你的 IRREVERSIBLE 只是搜索失败"时，现有的辩护（预算化定义、controller-human 一致率 ≥0.85 闸门、UNKNOWN 兜底）够不够？
- 给出加固方案并评估成本：(a) 对 IRREVERSIBLE 点做系统性 affordance 枚举（页面全部可交互元素 × BFS 到深度 d）而非单控制器；(b) 用一个强 LLM agent 以"恢复初始状态"为目标做对抗性 undo 搜索，找到反例即翻标签（labeled 为 IRREVERSIBLE 的点被攻破率作为标签噪声上界报告）；(c) 后端 DB 层面 diff（Magento 数据库快照比对）作为 σ 的上界信号。哪个该进正式版协议？
- 探针注册现状：`ProbeSpec` dataclass + `register()`，全部由研究者手写在 `probes/*.py`（shopping 13 个、reddit 7 个），工作台无人工注册界面。评估"探针即代码"的路线 vs 在工作台开放"人工注册探针（选择动作、undo 序列、σ 信号）"的收益与风险（人工注册会不会把"意见"重新混进"行为测量"？）。
- undo 控制器只有一条固定策略，如何防"控制器写得烂 → 假 IRREVERSIBLE"系统误差？§6.5 一致率闸门用人工审做后验，是否需要前验（同一探测点跑 N 个不同 undo 策略取最优）？

#### 维度四：grounded 标签的规模与统计效力

当前 `data/grounded/reversibility.jsonl` 仅 **32 条**探测记录、约 10 个动作类，其中 place_order=IRREVERSIBLE 仅 n=1，UNKNOWN/NO_EFFECT 占 9 条。计划书 §9.4 自定纪律"每 cell n≥30 才报点估计"。必答：

- 以此纪律反推：主表每个 split × 每个动作类 × 每档标签需要多少探测点？现有采集速率（~40s/点）下需要多少机时？§6.7 的"4 周内 300–600 点"目标与当前 32 点之间缺什么（探针 live 化？状态点扩容？破坏性审批流程？）——列出关键路径。
- 类内恒定性假设（同 action-type 在不同状态下可逆性恒定，pilot 按类复用标签）风险多大？哪些类最可能出反例（库存最后一件的 add_to_cart？已满的 wishlist？）？
- 意见标签并行采集（§6.3 分歧矩阵）当前是 0。没有它，−grounding ablation 和"对 WebGuard 式标注的定量批评"两条主张都空转——评估这是不是当前最高优先级的数据缺口。

#### 维度五：样本形态与输入格式（history 表示）

现状：单步样本（`revact_sft.jsonl` 92 条）+ 多轮样本（`revact_sft_multiturn.jsonl` 62 条）双族，`iris.v2` 输入 = `<goal>` + `<history>`（每历史步一行：动作 + [nav/state-change/update/no-effect] 标记 + 关键 delta，由相邻观测 fingerprint diff 计算，绝不取自 grounded 标签）+ 当前步完整 pruned AXTree；多轮样本最近 K=5 步保留真实对话轮、更早折叠进 `<history>`。必答：

- 对照主流 agent 训练输入格式逐一比较并给结论（表格）：AgentOccam（objective + 交互历史 + 当前 AXTree）、UI-TARS（近 N 步 (o,a) + thoughts 短期记忆）、AgentTrek、WebRL/WebDreamer、OpenAI CUA/Operator 风格、VAGEN 多轮 POMDP 形态。`iris.v2` 的三要素历史行相对这些格式的优劣？[no-effect] 防死循环标记是否有先例或等价物？
- 数据体检：已采集样本的 `<history>` 行数分布实测是多少？如果大量样本历史只有 1–3 步，根因是什么（轨迹本身短？关键状态集中在轨迹前部？reach 直达合成状态的历史是 plan 渲染而非真实交互？`meta.history_source` 的 plan/canonical/trajectory 三种来源占比？）——这会不会让"历史表示"这个设计点在训练数据里根本没被压到（模型没见过长历史）？
- 完整 AXTree 截断（snapshot 6k / policy 12k chars）对关键控件可见性的影响：undo 通道证据（如 'Remove item' 链接）会不会被截掉，导致 `<rev_check>` 训练时事实在输入里不可见、模型被迫幻觉？给出检测脚本思路。
- 多轮样本 loss 计在全部 assistant 轮，途中步只有 `<answer>` 而决策步才有完整 `<think>`——这个"途中步不思考"的设计与 rollout 时模型每步都可能输出完整块是否一致？会不会训出"格式条件反射于步数位置"？

#### 维度六：输出格式与 `<rev_check>` 的实效

`iris.v2` 为逆向世界模型补了 reasoning（`<rev_check>`：标签前的机制级 undo 通道检查，措辞 teacher 写、事实由探针实录钉死、QC 矛盾正则拒绝违背 pinned 标签的措辞）。必答：

- rationale 蒸馏防"标签查表过拟合"的主张（引 Hsieh et al. 2023）在**当前动作类数量（~10 类）**下能否兑现？类太少时 rev_check 措辞会不会本身就模板化成另一种查表？如何在 cross_action_class split 上设计一个直接检验"rev_check 是否在做机制推理"的探针实验（如：对抗样本——把页面上的 Remove 控件删掉，看 rev_check 和标签是否跟着变）？
- teacher 蒸馏当前产物为 **0 条**（`revact_sft_distilled.jsonl` 空），全部样本仍是模板 prose——"防整段背诵"的措施实际尚未生效。评估模板 prose 训练的模型会背诵到什么程度、这是否解释 pilot 中 base 臂的全拒行为。
- QC 矛盾正则（`train/distill.py::qc_check`）的覆盖面：列举它抓不住的矛盾类型（如 rev_check 说"存在 Remove 控件"但当前 AXTree 里没有——跨输入-输出的一致性检查目前有没有？）。

#### 维度七：候选动作生成（S4 placeholder）

流水线编号 S3 直接跳到 S5，S4（生成式候选提案）缺位；`/api/candidates` 现仅返回专家动作 + 规则安全动作 + DPO 反事实，前端 `views.js` 有 `kind === 'placeholder'` 徽标。必答：

- S4 在整个 idea 里承担什么（给 S5 探针喂待标注动作、给 S8 造 DPO 负对、给评测集造非专家动作分布）？它缺位对当前数据的具体伤害是什么（DPO 负样本是否全部来自标签翻转而非真实备选动作 → 偏好对分布与部署时的动作候选分布错位）？
- 评估如下候选生成谱系设计并改进：每个状态节点 4–6 个候选，覆盖｛专家动作 / a11y 可交互元素枚举 / 高危关键词动作 / 目标违背动作（命中 constraint trigger）/ **诱饵——看似高危实则可逆**（有二次确认的 delete、可撤销的 upvote，专治"见 delete 就喊危险"的关键词捷径）/ 需验证动作（信息不足的 Continue）｝，生成后按当前页面真实 bid 做合法性过滤。诱饵类在当前 3 站点里有哪些可实现的实例？没有诱饵类，cross-risk 评测会怎么露馅？
- 候选生成用 LLM 还是规则枚举？若用 LLM，如何防止它把"意见"注入本该行为实测的标签链路？

#### 维度八：训练算法谱系（DPO vs GRPO/GSPO）

现状：SFT + DPO（四类偏好对）为主线，`train/grpo.py` 已实现 GRPO/GSPO（TRL、离线可验证 reward：format+1 / decision 对 oracle+2 / reversibility 对实测标签+2 / 约束违反−4），计划书定位为"ablation、非主线贡献"，理由是防 VAGEN 复刻质疑 + RL 只为闭合 rollout 分布偏移（pilot 发现的多步二次暴露失败模式）。必答：

- 「RL 降级为 ablation」是聪明的定位还是错失贡献？VAGEN 的 Bi-Level GAE、WebRL 的 curriculum、GSPO 的 sequence-level IS 各自解决什么，IRIS 的失败模式（step 1 正确 AVOID，step 2 换页遇同类控件即点）更像哪一类问题？当前离线查表 reward 没有环境在回路，能治多步游走吗——还是必须上多轮在线 RL（环境在回路的 GRPO）？评估在 WebArena 沙盒上跑在线多轮 RL 的工程成本与探针污染问题（rollout 会产生真实订单）。
- reward 设计审查：format(+1)/decision(+2)/reversibility(+2)/violation(−4) 的量纲是拍脑袋还是有依据？稀疏 violation 惩罚在 GRPO 组内优势归一化下会不会被稀释？要不要 step-level credit assignment（正是 VAGEN Bi-Level GAE 的动机）？
- DPO 四类偏好对（false_safe/over_block/goal_violation/wrong_reversibility）全部由标签翻转构造，chosen/rejected 共享同一 prompt——这与 DPO 文献里"负样本应来自模型自身错误分布"的经验冲突吗？要不要引入 on-policy 负样本（模型 rollout 采错的步）？

#### 维度九：Prompt 治理完备性

registry（`revact/prompts.py` + 工作台 Prompt 管理页 + `prompts_fp` 指纹溯源 + `configs/prompts.local.json` 覆盖）已建。必答：

- 全仓扫描：还有哪些 LLM-facing 文本没进 registry（意见标签采集器的 prompt？评测 few-shot 基线的 prompt？rollout 系统提示词与训练 `agent_system` 是否真是同一条）？
- `prompts_fp` 只记指纹不记 diff——A/B 调优时如何知道两个指纹差在哪？要不要版本化存储全文？
- 可编辑 prompt 的安全边界声明（"调 prompt 调不动标签"）有没有测试兜底（改烂 `teacher_distill` 模板后 QC 是否必然拦截）？

#### 维度十：实验设计与假设可证伪性

必答：

- H1 独立性探针（前向-only 模型预测可逆性不超过按钮文字基线）是"全项目风险最高的实验"，但按钮文字 logistic 基线在**动作类只有 ~10 个、且类内标签恒定**的数据上会不会本身就打到天花板（字面即标签），使 H1 无论真假都测不出来？样本量多少才够 Wilson 区间分离？
- split 泄漏审计：state_group 切分能否防住"同一商品页的不同状态点跨 split"？cross_action_class 的 held-out 类与 train 类共享站点模板（Magento UI 骨架相同），泛化增益会不会被"站点先验"污染？
- decoupled 对照的公平性三约束（同基座/同标签/同信息）在实现里有没有被违反的缝（三 LoRA 的 verifier 多看了 `a` 和 `Δŝ`，token 预算与 coupled 不同——"同信息"怎么算）？
- 指标口径：FSR 分母"真危险步"依赖 grounded 标签本身，若维度三的 soundness gap 成立，FSR 的分母就有标签噪声——评测指标对标签噪声的敏感性分析该怎么做？
- pilot 数字（base 臂 FSR 0% 是全拒退化、iris-SFT 9.1% FSR、决策声明准确率 86%、n=11）能支撑哪些话、不能支撑哪些话？

#### 维度十一：论文定位与撞车防御演练

- 模拟 5 条最狠的审稿意见（英文原话式），覆盖：VAGEN 换皮质疑、"probe=启发式规则所以标签还是人工意见"质疑、单域外部效度、IRREVERSIBLE 的 one-sided evidence、与 Revisable by Design 分类学的增量。为每条写出目前最强防御与防御失败时的退路（对应 §9.5 诚实预案的三档）。
- 检查三大贡献声明（构念+测量 / 内在化方法 / coupling 受控研究）中哪一条在当前数据规模下**已经**有证据、哪一条纯靠承诺。按"证据完备度"重排贡献顺序，并判断 dataset/benchmark track 是不是比 main track 更稳的投稿位。

#### 维度十二：工程质量、数据卫生与可复现

- **数据来源混杂**：`revact_sft_multiturn.jsonl` 62 条中 12 条来自 mock 环境（`mock.add_to_cart_seed0` 等）与 50 条 webarena 混在同一训练文件，无 site/env 隔离开关——评估 mock 数据混入正式训练集的风险与该有的隔离机制。
- 轨迹成功率仅 22/93（~24%），成功轨迹是唯一的专家动作来源——采集策略模型（collector）的质量瓶颈会不会传导成"专家动作"质量瓶颈？
- 工作台数据集浏览器目前展示字段级视图，缺"完整样本形态"展示（system/user/assistant 全文 + meta + 血缘：state → probe → 标签 → 模板/teacher → split）。给出参照 HuggingFace dataset card 的展示与导出规范（含每个 split 的统计、已知偏斜、标签流程声明、prompts_fp）。
- 测试覆盖审计：`tests/` 有 12 个文件，哪些关键路径没有测试（undo 控制器的假阴性、AXTree 截断、多轮折叠正确性）？MANIFEST（30 条）与 reversibility.jsonl（32 条）行数不一致的原因？

### 5. 输出格式（严格遵守）

1. **《声称 vs 现实》差距表**：计划书自称已落地但代码/数据不支持的每一项。
2. **问题总表**：编号 | 维度 | 严重度(P0/P1/P2) | 一句话问题 | 证据(文件:行/统计) | 对论文主张的具体伤害 | 审稿人攻击话术(仅 P0/P1) | 短期修法 | 长期修法 | 验收标准。按严重度排序，P0 预计 3–6 个，总数 15–30 个，宁缺毋滥。
3. **Top-10 整改路线图**：按「先救 P0、数据缺口优先于新功能」排序，标注依赖关系与预估工作量（人日）。
4. **审稿人模拟 Q&A**：维度十一要求的 5 条攻击 + 防御 + 退路。
5. **一页结论**：这个 idea 当前的真实状态（几成胜算、最大单点风险、投稿位建议），不许骑墙。

### 6. 附录 A：作者已知的疑虑清单（你必须逐条回应，允许反驳作者）

1. 只在 3 个站点镜像采样，任务泛化能力存疑；
2. 关键状态靠写死的关键词匹配挖掘，241 个状态里 add_to_cart 210 / place_order 29 / delete_address 2，分布严重偏斜、类别人为封顶，换任务场景即失效；
3. 探针为人工注册的少数几个，无法证明不存在其他动作序列可还原初始状态，IRREVERSIBLE 标签的 soundness 与 idea 的可逆性定义强绑定，是大问题；
4. S4 候选动作生成还是 placeholder，其作用与实现优先级不明；候选谱系应含"诱饵"类（看似高危实则可逆）防关键词捷径，且需按真实 bid 做合法性过滤；
5. 每个 SFT/DPO 样本是单 step 还是完整轨迹、完整样本形态在工作台里看不到；
6. 历史输入曾缺失/过短（观察到只有 1–3 步的 history），"goal+完整当前 AXTree+每历史步一行三要素（动作+关键 delta+风险标记）"是否是对的设计、是否符合主流 agent 训练输入；
7. 逆向世界模型曾只有离散标签没有 reasoning，怕标签查表过拟合、新场景失效；
8. 前沿 agent 工作不会只用 DPO，GRPO/GSPO（VAGEN 有专门的 RL 优势函数改进作为贡献）该不该上、怎么定位；
9. 工作台所有 LLM prompt（采集策略/teacher 蒸馏/样本系统提示词）应可编辑可溯源。

> 注意：其中 6/7/8/9 计划书声称已在 v1.1/v1.2 落地——你的任务不是复述"已落地"，而是验证落地质量并找出残留问题（如蒸馏产物为空、history 实际长度分布、RL reward 量纲）。

### 7. 附录 B：已核实事实清单（2026-07-13，有仓库时必须重新核实）

- 站点：shopping(:7770) / shopping_admin(:7780) / reddit(:9999)，`configs/default.yaml`。
- 关键状态：`data/raw/state_bank/shopping_key_states.jsonl` 241 条；afforded_action_types 分布 add_to_cart 210 / place_order 29 / delete_address 2；挖掘逻辑 `revact/data/collect.py::afforded_action_types`（自注释 "shallow keyword match"），关键词表 `revact/config.py::PILOT_ACTION_KEYWORDS` 仅 3 类。
- grounded 标签：`data/grounded/reversibility.jsonl` 32 条 / `MANIFEST.jsonl` 30 条；分布：add_to_cart REVERSIBLE 5、wishlist_add REVERSIBLE 3(+NO_EFFECT 1)、compare_add REVERSIBLE 3(+NO_EFFECT 2)、newsletter_subscribe REVERSIBLE 4(+NO_EFFECT 1)、currency_switch UNKNOWN 2、address_add UNKNOWN 3、place_order IRREVERSIBLE 1 + UNKNOWN 1、reddit_vote REVERSIBLE 1 + PARTIALLY 2、reddit_subscribe REVERSIBLE 3。
- 训练数据：单步 SFT 92 条（全 shopping）、多轮 SFT 62 条（**12 条 mock 环境 + 50 条 webarena**）、DPO 230 对 + 多轮 155 对、teacher 蒸馏产物 **0 条**（`revact_sft_distilled.jsonl` 空）。
- 轨迹：`trajectories_meta.jsonl` 93 条、成功 22 条（≈24%）。
- 样本格式：`iris.v2`；多轮样本 meta 含 `format/prompts_fp/history_source/undo_steps/decision_step/n_turns`；system prompt 要求输出 `<think><observation/reasoning/prediction/rev_check/reversibility/undo/decision></think><answer>`。
- 探针：代码注册制（`grounding/base.py::register(ProbeSpec)`），shopping 13 个（live/self-recovering/dry-run 三档，破坏类需 `--commit` + `REVACT_ALLOW_DESTRUCTIVE=1` 双闸门）、reddit 7 个；工作台无人工注册探针的界面。
- 训练代码：`train/sft.py`、`dpo.py`（四类偏好对）、`grpo.py`（GRPO+GSPO，离线查表 reward）、`distill.py`（teacher 条件蒸馏 + QC 矛盾正则）。
- 候选动作：`/api/candidates` 返回专家动作/规则安全动作/DPO 反事实；生成式候选（S4）为 placeholder（前端有 placeholder 徽标；S1–S8 文档目录无 S4 节）。
- 评测：`eval/decisions.py`（静态 decision eval，n<5 只列不评的警告已实现）、`eval/rollout.py`（守护式 rollout FSR，破坏类只记录不执行）；pilot：11 状态 ×2 变体，base 臂 FSR 0%（全拒退化、可解析 1/22）、iris-SFT FSR 9.1%、决策声明准确率 86%。
- AXTree 截断：snapshot 6000 chars / policy 12000 chars（`configs/default.yaml`）。
- prompt registry：`agent_system` / `collector_system` / `teacher_distill` / 约束与请求模板池，覆盖存 `configs/prompts.local.json`（gitignore），样本 meta 记 `prompts_fp`。

---

## PROMPT 正文结束
