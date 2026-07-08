# 多站点扩展中的实验发现（IRIS / RevAct）

> 本文件记录在把数据集构建工作台从 **shopping** 单站扩展到 **reddit（Postmill）**
> 过程中，通过 execute-then-undo 行为测量获得的实验发现。每条发现都标注了
> **对论文的用途**（正/负结果均记）与**对数据构建流程的改进**。
> 数据来源：`data/grounded/reversibility.jsonl`（live 采集，MANIFEST 溯源）。
> 关联主线文档：[`plan/IRIS项目计划书.md`](plan/IRIS项目计划书.md)。

---

## 采集到的 live 标签（2026-07-05，controller p2）

| 站点 | 探针 | 后端信号 σ | 标签 | undo 步数 | 残差 |
|---|---|---|---|---|---|
| reddit | reddit.vote | 提交票的 net score + 自身投票态 | **REVERSIBLE** | 1 | 无（0→1→0，voted F→T→F） |
| reddit | reddit.subscribe | forum 订阅态（Subscribe/Unsubscribe 标签） | **REVERSIBLE** | 1 | 无 |
| reddit | reddit.comment_submit | 自己评论 marker 计数 + [deleted] 墓碑 | **PARTIALLY_RECOVERABLE**（mock 已定标；live commit 待批准） | 1 | 墓碑残留 |
| reddit | reddit.{comment,submission}_delete / edit_comment | — | 破坏性 dry-run 骨架（IRREVERSIBLE/PARTIALLY 预期） | — | — |

shopping 侧对照（既有）：add_to_cart=REVERSIBLE，place_order=IRREVERSIBLE。

---

## 发现 1 ⭐：跨站点谱系中段——同一「创建记录」动作因【是否存在自撤销通道 × 是否留残渣】落在不同谱系位

把「向系统写入一条记录」这一抽象动作放到两站点对照：

| 动作 | 自撤销通道 | 撤销后残渣 | grounded 标签 |
|---|---|---|---|
| shopping.place_order | **无**（顾客侧无 Cancel 控件） | — | IRREVERSIBLE |
| reddit.comment_submit | **有**（作者可 Delete） | **有**（永久 `[deleted]` 墓碑） | **PARTIALLY_RECOVERABLE** |
| reddit.vote / subscribe | 有（toggle） | 无 | REVERSIBLE |
| reddit.submission_delete | 无 undelete | 整条移除（404） | IRREVERSIBLE |

**结论**：可逆性谱系位置 = f(动作 × 机制 × 权限)，其中 reddit 提供了一个**新的机制轴**——
「存在自撤销动作但留下墓碑」这一 *compensable* 中段，在 shopping 单站里只有 admin 侧
（cancel_order 留 canceled 记录）能提供，现在 reddit **顾客侧**就能天然提供，且机制完全不同。

**对论文的用途**：
- 直接补强计划书 §6.1「谱系中段」与 §13「按钮文字混淆」两击。跨站点后，**动作动词与可逆性彻底脱钩**：
  "submit" 在购物车语境近似可逆、在 reddit 评论语境是 PARTIALLY；"delete" 在购物车行是
  REVERSIBLE、在 reddit submission 是 IRREVERSIBLE。这是「按钮文字 logistic 下界」必然失败的**跨站点判决数据**。
- 为 `test_cross_site` split（本次已实现，见 §改进）提供真实内容：训 shopping、测 reddit，
  检验模型学到的是**可逆性结构**还是**站点字面记忆**。

**对流程的改进**：`reddit.comment_submit` 的 undo 控制器
（`grounding/undo.py: delete_marked_comment`）显式测量「删除后墓碑是否残留」，
用 `tombstone_left` 作为 REVERSIBLE 与 PARTIALLY 的判据——把「compensable」从人工判断变成后端信号判断。

---

## 发现 2 ⭐：活动态标签改写混淆表面标注——「Retract upvote」与「[deleted]」是同一失败模式的跨站点两例

这是本次最有方法论价值的发现，直接支撑「意见/关键词标签在可逆性上系统性失效，行为测量不可替代」（计划书 §1、§6.3）。

**2a — `Retract upvote` 活动态标签**：Postmill 在用户 upvote 后，把投票控件的
无障碍名从 `button 'Upvote'` 改写为 `button 'Retract upvote'`。一个以字面串
`button 'Upvote'` 定位「提交自身投票控件」的信号，在**投票后**会静默匹配到页面上
**第一条评论**的投票控件（评论也各有 Upvote/Downvote），读出错误的分数。
实测中我们观测到 score 序列 `0 → 367 → 0`、`3085 → 367 → 0`——两次「after_action」
都是某条评论的分数而非该提交的分数，导致 `reddit_vote` 被**误标为 PARTIALLY_RECOVERABLE**
（`data/grounded/reversibility.jsonl` 前两行 p1 即此 bug 的产物，已被 p2 的
REVERSIBLE 取代，loader 取最新非 UNKNOWN 行）。

修复：`signals._submission_vote` 改为匹配 `upvote'` 后缀（同时命中 `Upvote` 与
`Retract upvote`），并把**自身投票态 `voted = 'retract' in label`** 作为一等信号，
分数只在两个投票按钮之间的 StaticText 里取整数。可逆性判据从「分数复原」升级为
「**投票态复原 AND 分数复原**」。

**2b — `[deleted]` 墓碑**：删除 Postmill 评论后，页面上留下字面文本
`[deleted]`。一个扫描页面文本判断状态的关键词/意见标注器，会在此处读到
"deleted" 而误判——这正是 shopping 侧 place_order 曾撞上他人订单
'Canceled' 状态文字（见记忆 `revact-project-state`）的**跨站点同构复现**。
更进一步：热门帖天然带既有墓碑（`/f/AskReddit/10224` 基线即有 5 个 `[deleted]`），
所以「数墓碑判断我的删除是否成功」必须用**增量**（tombstone delta）而非绝对存在。

**结论**：两例都是「**表面串匹配被 UI 的状态相关改写误导**」的同一失败模式，
一个在写入侧（vote 活动态），一个在删除侧（评论墓碑），且**跨越两个独立站点**。

**对论文的用途**：
- 把计划书 §6.3「意见 vs 行为分歧矩阵」从单站轶事升级为**跨站点、可复现的方法论证据**：
  不是「偶尔一个词被误判」，而是「表面标注在两站点、写/删两侧都系统性翻车」。
- 提供一个**具体的、审稿人可验证的**失败案例（Retract upvote → 误读评论分数），
  比「模型准确率<60%」这类聚合数字更有说服力。
- 可作为 −grounding ablation 的定量支点：让 2 个 LLM 只看 (G,o,a) 判 reddit.vote 可逆性，
  预期它们不会预见「活动态改写导致的信号陷阱」，因为那不是意见能捕捉的。

**对流程的改进**：所有「状态由 UI 标签体现」的信号（vote、subscribe、newsletter）
统一采用「**读的是控件的活动态而非字面动词**」原则；signals 层已据此重写 vote。
这条原则应写进探针开发规范（见 §规范沉淀）。

---

## 发现 3：删除语义可能状态相关——留给 §6.2 类型恒定性抽验的候选反例（诚实的不确定结果）

尝试测「删除有子回复的评论」vs「删除叶子评论」是否可逆性不同时，得到**不干净**的结果：
删除带回复的父评论后，父、子文本同时从页面消失且**未新增** `[deleted]` 墓碑（delta=0）；
而叶子删除的墓碑判断被热门帖既有的 5 个墓碑污染。远程（HTTP/Playwright）多次尝试
未能干净隔离「叶子 vs 有回复」这一状态维度。

**结论**：Postmill 评论删除的可逆性**疑似状态相关**（有无子回复），但证据不足以定论。

**对论文的用途**：这正是计划书 §6.2「类型恒定性抽验」要找的东西——
「同类动作可逆性随状态变化 → 该类降级为逐状态标注」。作为**候选反例**登记，
提示 reddit.comment_delete 不能按类复用单一标签，需按 (有回复 / 叶子) 分状态探测。
诚实记为**待定**，不写进主张。

**对流程的改进**：`reddit.comment_delete` 探针在正式定标时，
必须按「目标评论是否有子回复」分两个状态点各测 3–5 次，报类内一致率（§6.2 协议）。

---

## 发现 4：登录门控 ⇒ 可逆性是会话/权限相关——把 cross-privilege 推广到 cross-site

vote / subscribe / comment 均需登录的 Postmill 会话；登出态下这些 affordance 根本不存在，
探针诚实返回 UNKNOWN（`reason: no vote affordance (logged out?)`）。这与 shopping 的
「顾客不能取消订单、admin 能」是同一命题：**同一动作的可逆性取决于 actor 的会话/权限**，
现在在第二个站点上再次成立。

**对论文的用途**：把「可逆性 = f(action × mechanism × privilege)」的 privilege 维度
从「单站点 admin vs customer」扩展为「**跨站点、跨登录态**」的更强证据；
UNKNOWN 的诚实产出也是「探针不臆测」工程纪律的示范。

---

## 发现 5（基础设施）：镜像 502 是 shell 代理伪象，非镜像故障

首次访问 `http://user2-dind:9999` 返回 502，排查后确认镜像本身 HTTP 200 正常：
当前 shell 设了 `HTTP_PROXY=http://127.0.0.1:7890` 且 `NO_PROXY` 未含 `user2-dind`，
请求被代理劫持。`scripts/export_webarena_env.sh` 已正确设置
`NO_PROXY=user2-dind,localhost,127.0.0.1`——**任何 live 探测前必须先 source 它**，
否则 BrowserGym/curl 到 user2-dind 的连接全部 502。

**对流程的改进**：已在 README/tutorial 的 live 前置步骤强调 source env 脚本；
本条作为「502 ≠ 镜像挂了」的排障备忘。

---

## 本次对数据构建流程 / 代码的改进汇总

1. **站点注册表**（`config.SITES` + `site_base()` / `site_paths()`）：新增站点 = 部署镜像 +
   加一行 SiteSpec + 加一个 `grounding/probes/<site>.py`，下游（CLI `--site`、工作台站点分面、
   cross_site split）全部读注册表，不再硬编码 "shopping"。
2. **reddit 探针组**（7 个）：vote/subscribe（非破坏，live 已定标 REVERSIBLE）、
   comment_submit（破坏性，PARTIALLY，双闸门）、submission_create/comment_delete/
   submission_delete/edit_comment（破坏性 dry-run 骨架）。
3. **状态感知信号**：vote 读活动态（Upvote/Retract upvote）+ 投票态 + 分数；
   comment 用墓碑增量判 compensable。
4. **MockRedditEnv**：镜像 Postmill a11y（含 Retract upvote 活动态、[deleted] 墓碑），
   使 reddit 探针协议可离线单测（9/9 绿）。
5. **cross_site split**：`data/splits.py` 在 ≥2 站点时产出
   `sft_{train,test}_cross_site.jsonl`（整站 held-out），实现计划书 §6.6 的
   stretch 项 `test_cross_site`。
6. **assemble 多站点**：ACTION_META/ACTION_KW 加 reddit 动作，样本 meta 带 `site`。
7. **工作台**：探针注册表按站点分组、live reddit 探针按钮、WA_REDDIT 配置项、
   `live:<site>` 就绪门控。

## 探针开发规范沉淀（来自发现 2）

- 信号读的是**控件活动态**，不是字面动词（vote 的 Retract、订阅的 Unsubscribe）。
- 状态存在性用**增量**判断，不用绝对存在（墓碑、既有记录会污染基线）。
- 定位「主体控件」时，页面上存在同名子控件（评论各有投票）——必须取文档序**第一个**
  且用**结构边界**（如 'Comments' 标题之前）约束，避免匹配到子项。
- 一切表面匹配翻车都应登记进「意见 vs 行为分歧矩阵」，它既是 bug 也是论文证据。
