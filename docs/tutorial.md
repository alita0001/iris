# IRIS 数据集构建工作台 · 新手上手教程

> 这份教程假设你**完全不了解**这个项目。读完第 1–4 章你就能打开工作台浏览数据；
> 第 5 章演示如何浏览、隔离和 dry-run；当前正式数据为空，不能完成非空 formal 导出。
> 项目的研究背景见 [`plan/IRIS项目计划书.md`](plan/IRIS项目计划书.md)，
> 工程边界速查见 [`workbench.md`](workbench.md)，本文只管"怎么用"。

> **当前数据状态（2026-07-13）**：本教程中的 92 条 single SFT、62 条 multi SFT、
> 32 条 grounding 和 mock/live 操作截图都是 legacy/开发资产示例，不是正式训练集。
> formal `probe_points.jsonl` 与 `POINT_MANIFEST.jsonl` 均为 0 行；旧 grounding 中 2 条
> 缺 provenance，另 30 条只作 class-level smoke，全部隔离。历史 multi 中 12/62 为 mock，
> 已由 formal export gate 排除。S4 v2 有 300 条/50 states，但都只是 snapshot-legal
> candidate proposal，尚无 point label。历史 rollout 百分比因真值分母错误已撤回；当前没有 cross-site、
> cross-action、cross-privilege、coupled/decoupled 或 live 实验结果。以下演练用于浏览和
> 验证工程路径；在正式 point 为 0 时，正确的 formal 导出结果是空或明确拒绝。

---

## 目录

1. [三分钟看懂这个项目在做什么](#1-三分钟看懂这个项目在做什么)
2. [名词速查表（先扫一眼，忘了随时回来查）](#2-名词速查表)
3. [准备工作：目录认路 + 两档运行模式](#3-准备工作)
4. [第一次启动：5 分钟跑起来](#4-第一次启动)
5. [三个实战演练（浏览、隔离与 dry-run）](#5-三个实战演练)
6. [十一个页面逐个讲解](#6-十一个页面逐个讲解)
7. [不用界面：命令行等价操作](#7-命令行等价操作)
8. [数据文件长什么样](#8-数据文件长什么样)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [安全须知（必读）](#10-安全须知必读)

---

## 1. 三分钟看懂这个项目在做什么

### 1.1 背景故事

我们在训练一种"会上网操作的 AI 代理"（web agent）：它能在购物网站上搜索、加购物车、下单。
危险在于：动作造成状态变化后，agent 在当前权限和有限预算内不一定能恢复。把商品加进
购物车通常有 Remove 通道；点 "Place Order" 后，早期 customer UI controller 没找到取消
通道。但后者只能称“该 solver 在预算内未恢复”，不能据此证明数学意义的不可逆。

IRIS 项目的核心想法：训练模型在动手之前先判断"这个动作能不能撤销、撤销要几步"，
正式标签不能靠人或 LLM 直接填写，而应由程序在固定 point 上
**执行动作 → 尝试撤销 → 对比 pre/post/final signal**测量（execute-then-undo）。
当前仓库已经实现 schema 和 gate，但尚无满足该 schema 的正式 point 数据。

这个仓库里的 `revact/` 就是把上面的想法变成一条**数据集生产流水线**，
而你现在要学的"工作台"（workbench）是这条流水线的**图形化驾驶舱**：
在浏览器里点按钮跑流水线、逐条检查数据、人工打标、最后导出训练集。

### 1.2 流水线一张图

数据从左到右流动，每一步的产物都存成文件（第 8 章讲文件长什么样）：

```
①环境初始化 → ②成功轨迹采集 → ③关键状态采集 → ④约束注入 → ⑤候选动作
   (检查配置)     (AI 代理逛网站,     (挑出"高危页面",   (给任务加一句      (每个状态列出
                  记录每一步)         如结账页)          "但是不许下单")     可选的动作)
        → ⑥反事实动作 → ⑦effect/recovery 测量 → ⑧teacher 蒸馏 → ⑨质量校验 → ⑩导出
          (构造"错误示范"   (真去执行+撤销,        (让大模型把理由    (统计+挑出   (train/val/test
           作对比教材)       保存两维结果⭐)         写通顺,结论不动)   坏样本)      + 数据集说明书)
```

⭐ 第⑦步是整个项目的证据闸门：只有带唯一 `probe_point_id` 的实际 trace 才能进入正式数据。

### 1.3 最终产出什么

目标产物是两种训练数据；当前仓库只有 legacy/quarantine 样例，formal 集为空：

- **SFT 样本**：一段对话——用户给任务+网页内容，助手输出一段带标签的思考
  （`<observation>观察 <reasoning>推理 <prediction>预测 <rev_check>逆向机制检查
  <reversibility>恢复判断 <undo>撤销计划 <decision>决策</think><answer>动作`）。旧文件是
  `iris.v2`；正式 point-grounded 输出使用 `iris.v3`，且 prediction/undo 必须来自实测 trace。
- **DPO 偏好对**：同一个场景下"好回答 vs 坏回答"成对出现。historical 文件只有
  synthetic flip；正式主集还要求 legal candidate 或模型自身错误，并保留 negative source。

---

## 2. 名词速查表

| 名词 | 大白话解释 |
|---|---|
| **trajectory（轨迹）** | AI 代理做一次任务的完整录像：每一步点了什么、页面变成什么样、有没有拿到分 |
| **key state（关键状态）** | 轨迹里值得注意的页面，比如出现了"Add to Cart / Place Order / Delete"按钮的页面 |
| **reached state（风险状态）** | 用固定路线直接导航到的高危页面（不用跑完整任务），是造样本的"锚点" |
| **约束（constraint）** | 给任务加的限制条款，如"看看就行，别加购物车"。有 11 种显式说法 + 5 种隐式说法（隐式=整句没有"不要"二字），防止模型靠背句式作弊 |
| **候选动作（candidate）** | 在某个状态下模型可能采取的动作清单：专家动作、安全替代动作等 |
| **反事实（counterfactual）** | historical DPO 中由文本/标签翻转构造的 4 类 synthetic 坏回答；必须与未来 legal/on-policy negatives 分开 |
| **grounded / grounding** | “有据可查”：formal point 具有唯一 `probe_point_id`，并保存 pre/action/post/undo/final evidence；旧 class-smoke 不满足此义 |
| **probe（探针）** | 到指定 point 执行动作、尝试撤销并读取 signal 的程序；注册了多少 probe 只表示代码覆盖，不表示采到了正式数据 |
| **undo** | 撤销。探针会记录撤销用了几步（undo_steps）、每一步点了什么（undo_actions） |
| **动作效果** | formal `effect_status`：`CHANGED` / `NO_EFFECT` / `UNKNOWN` |
| **恢复结果** | formal `recovery_status`：`RECOVERED` / `PARTIALLY_RECOVERED` / `NOT_RECOVERED_WITHIN_BUDGET` / `UNKNOWN`；`undo_cost_steps` 单列。`IRREVERSIBLE` 只用于 legacy/display |
| **oracle 决策** | `EXECUTE/VERIFY/CONFIRM/AVOID`；formal truth 必须独立记录 policy constraint、授权和 goal necessity，不能只从恢复结果推导 |
| **teacher 蒸馏（distill）** | 请一个强模型（如 DeepSeek）把样本里模板化的"思考文字"改写得更自然。**结论（可逆性/决策/动作）被钉死不许改**，改了就会被质检打回 |
| **SFT / DPO** | 两种微调数据格式：SFT=标准问答示范；DPO=好坏成对的偏好数据 |
| **split** | train/val/test；formal split 要按 state group、实体、页面模板和环境隔离并审计 overlap |
| **overlay（标注叠加层）** | 你在工作台里做的一切人工操作（确认/驳回/改写/覆核）都存成独立的小文件，**绝不改动流水线原始产物**，导出时才合并生效 |
| **mock 模式** | 不连真实网站的 fixture/开发演练；产物必须进独立 config 或 quarantine，formal export 硬拒绝 |
| **live 模式** | 连接 WebArena 沙盒的受控执行模式；“可运行”不等于已有 live 结果，破坏性动作另需批准 |
| **job（任务）** | 你在界面上点"运行"后，后台实际执行的那条命令。每个 job 有独立日志 |

---

## 3. 准备工作

### 3.1 目录认路

```
iris/                          # ⭐ 工程根目录，后面所有命令都在这里执行
├── revact/                    #    Python 源码（cli / envs / grounding / data / train / server / ui ...）
├── data/                      #    数据产物（raw / formal grounding / legacy+quarantine / train / annotations）
├── configs/                   #    配置（default.yaml + workbench.example.json 模板）
├── scripts/                   #    live 模式环境变量脚本 + vLLM 启动脚本
├── outputs/                   #    输出（静态报告、job 日志、导出的数据集）
├── docs/                      #    你正在读的文档；docs/plan/ 是研究方案（06 号是总纲）
├── tests/  ci.sh              #    自检
└── README.md
```

### 3.2 两档运行模式（先想清楚你要哪档）

| | **离线档**（推荐新手从这里开始） | **live 档** |
|---|---|---|
| 能做什么 | 浏览 legacy/quarantine、mock/fixture 演练、运行校验；formal 导出会 fail closed | 在获准范围内采集新的 trace/point，或调用 teacher；结果仍须过 formal gate |
| 需要什么 | 只要 Python ≥3.10，**零第三方依赖、零 API key、零网络** | WebArena 站点可访问 + `agentlab` conda 环境 + DeepSeek API key |
| 怎么启动 | 任意 Python 直接跑 | 见 5.3 演练三 |

### 3.3 自检一下（可选但建议）

```bash
cd /workspace/iris
./ci.sh          # 跑代码检查 + 离线测试，最后一行应显示 "CI OK"
```

---

## 4. 第一次启动

### 4.1 启动服务

```bash
cd /workspace/iris
python -m revact.cli serve
```

看到这样的输出就是成功了：

```
[workbench] IRIS dataset workbench -> http://127.0.0.1:7788
[workbench] data root: /workspace/iris/data
[workbench] API keys: memory-only; saved config strips secrets.
```

然后用浏览器打开 **http://127.0.0.1:7788**。

- 端口被占用？换一个：`python -m revact.cli serve --port 8899`
- 服务在远程服务器上、你在本地笔记本？**不要**用 `--host 0.0.0.0` 裸奔公网，
  用 SSH 端口转发：本地执行 `ssh -L 7788:127.0.0.1:7788 你的服务器`，
  然后本地浏览器照常打开 `http://127.0.0.1:7788`。
- 停止服务：在终端按 `Ctrl+C`。

### 4.2 认识界面

页面顶部是 11 个页签，对应流水线的各个环节：

```
Pipeline | 全局配置 | 成功轨迹 | 关键状态 | 约束注入 | 候选/反事实 | Undo 标注 | Teacher 蒸馏 | 质量统计 | 数据集浏览器 | Jobs
```

右上角一行小字是**健康状态**，5 秒刷新一次：
`shopping · … 轨迹 / 92 SFT / 230 DPO · offline · 0 任务运行中`
——这些计数包含历史资产，不表示 formal 可训练量；`offline` 只表示当前没有 live 环境。

几个通用的界面元素，所有页面都长这样：

- **彩色圆点** = 阶段/任务状态：灰=未开始，蓝闪=运行中，绿=成功，橙=部分完成，红=失败。
- **彩色胶囊（badge）**：旧页面可能显示 `REVERSIBLE/IRREVERSIBLE` legacy 值；正式页应
  分别展示 effect/recovery。`EXECUTE/VERIFY/CONFIRM/AVOID` 是独立 normative decision，
  不能从 recoverability 单独推出。
- **左列表 + 右详情**：大部分页面左边是条目列表（点击切换），右边是详情。
- **虚线框"人工覆核"**：这就是打标区，点"确认/接受/待复核/驳回"即写入一条标注。
- 右下角会弹**黑色小提示条**（toast）告诉你操作成功还是失败。

---

## 5. 三个实战演练

### 5.1 演练一：离线审查 legacy/quarantine 资产（10 分钟）

目标：学习辨认资产层级，而不是把 92 条历史 SFT 导出成正式数据集。

1. 打开「Pipeline」和「数据集浏览器」，确认页面把 `formal_point`、`legacy`、
   `quarantine` 分开计数。绿色 job 状态只代表命令跑完，不代表数据可投稿。
2. 在「Undo 标注」页分别查看 empty formal point 和 legacy class-smoke。旧记录中的
   `REVERSIBLE/IRREVERSIBLE/NO_EFFECT` 是混合 ontology 展示值，不是 formal label。
3. 打开任意 historical SFT，检查完整 system/user/assistant、meta 和已有 lineage；会看到
   它无法回链唯一 `probe_point_id`。multi 资产中还应能审计到 mock origin。
4. 人工“确认/驳回”只写 overlay。它可以排除坏样本或记录意见，不能把 legacy 记录升级成
   point truth，也不能直接改 effect/recovery。
5. 尝试 formal export/dry-run。当前正确行为是导出空集或明确列出 exclusion reason；如果
   系统回退到 `action_type -> latest label` 并导出 92 条，应立即视为严重回归。

这个演练验证的是 fail-closed 数据治理，不是完成了一次正式数据集发布。

### 5.2 演练二：mock/fixture 流程（15 分钟）

目标：不连任何真实网站，用模拟环境把流水线从头到尾跑一遍。

1. 打开「Pipeline」页，左上角"全流程链"下拉框保持 **mock（离线可跑）**。
2. 点 mock/fixture 相关操作。工作台可依次验证：
   `环境检查 → mock 采集 → mock 探针 → 开发组装 → 校验/审计`。
   每步启动时页面会跳到 Jobs 页显示实时日志；全部跑完后回到 Pipeline 页，
   底部"链运行日志"能看到每一步的 ✓/✗。
3. 中途想看细节：切到「Jobs」页，点任意任务看完整日志（比如 mock 探针的日志里
   能看到 `add_to_cart -> REVERSIBLE {baseline:0, after_action:1, after_undo:0}`）。
4. 跑完后去「成功轨迹」页查看 `mock.` 记录，并确认 `is_mock/environment_origin` 可审计。
   只靠 trajectory 前缀不构成正式隔离；formal export 必须硬拒这些记录。

> 两个刻意的设计，别误会成 bug：
> ① mock/fixture 结果只能进入独立 dataset config 或 quarantine，不能进入 formal grounding；
> ② mock 环境里 `compare_add` 会报 `UNKNOWN(action affordance not found)`，
>   这是模拟站点没实现该控件，属正常现象。

也可以不跑整链，只单步跑：点某张阶段卡片 → 在"操作"区点对应按钮
（每个按钮旁边标了它需要什么：`mock` 徽章=离线可跑，`live`=要真实环境，`key: xxx`=要 API key）。
**从某阶段继续**：先点选一张卡片，再点顶部"从选中阶段继续"，链会从那一步开始往后跑。

### 5.3 live 准备检查（不在教程中自动执行）

本节只检查环境和生成待执行命令，不声称仓库已有 live 结果。WebArena 写操作与外部模型
调用是两个独立权限面：有 API key 不等于获准下单/删除/发布，也不等于能把 legacy smoke
写成 formal point。

1. **在正确的环境里启动服务**（关键！live 任务是服务的子进程，
   继承服务的 Python 环境，所以必须用装了 browsergym 的 `agentlab` 环境启动）：

   ```bash
   cd /workspace/iris
   source scripts/export_webarena_env.sh
   conda run -n agentlab python -m revact.cli serve
   ```

   刷新页面，右上角应从 `offline` 变成 `live 就绪`。

2. 先只运行环境 health check、snapshot 读取和 dry-run。不要把“站点可访问”记成 probe 结果。
3. 若要调用付费 teacher/collector，单独取得预算批准，并通过进程环境注入 key；不要把 key
   写进文档、命令参数、仓库文件或日志。当前 distilled 文件为 0 行。
4. non-destructive point 也必须生成唯一 `probe_point_id/state_id/action_instance_id` 和 1:1
   manifest；当前 live adapter 尚无正式 point 产物可展示。破坏性动作保持 `--commit` 与
   `REVACT_ALLOW_DESTRUCTIVE=1` 双闸门，并逐批取得明确批准。

---

## 6. 十一个页面逐个讲解

### 6.1 Pipeline（总控台）

- 10 张阶段卡片 = 第 1.2 节那张图。每张卡片显示序号、状态、产物和最近一次运行信息；
  状态只说明代码/job，不代表 formal 数据或论文实验已经落地。
- 点卡片 → 下方详情：阶段说明、**操作按钮**（带参数输入框，如 seeds、limit）、
  **产物文件表**、最近 job 入口。
- 顶部三件套：`全流程链`选 mock/live、`全流程运行`、`从选中阶段继续`。
- 第⑤阶段 S4 已有 snapshot 交互控件枚举与合法性校验：v2 产物为 300 条/50 states。
  它们仍是 proposal-only；没有 point grounding 或 on-policy error，见 6.6。

### 6.2 全局配置

- 三张模型卡（策略/teacher/judge）+ 运行参数卡（task_file、seeds、max_steps、
  采样数量、是否存截图、data_root 覆盖）+ 环境卡（WA_SHOPPING 站点地址）。
- 顶部数字块是当前数据规模和环境状态；下面的路径表告诉你数据在哪、报告在哪。
- 三个按钮：**应用配置**（写入服务内存，立即生效）、**保存到本地文件**
  （写 `configs/workbench.local.json`，剥离 key）、**重新加载**。

### 6.3 成功轨迹

- 筛选：全部/成功/失败/异常(截断)。列表项：绿点=成功、`已选`徽章=你标记过。
- 详情：任务 id、模型最后一条回复、逐步表格（点行看 axtree + 截图）。
- **选入下一阶段**按钮：把这条轨迹标记为"关键状态采集的候选来源"（写 overlay，
  供你在人工筛选时记录取舍）。

### 6.4 关键状态

- 每条 key state 显示：所属轨迹与步号、任务目标、URL、页面快照和发现来源。旧 241 条
  state 是仅覆盖三类英文关键词的 legacy 产物；新 detector 枚举 legal interactive controls，
  keyword/LLM 只能排序，真正 mutation 仍须 execute+signal diff 验证。
- 人工覆核区可以：选**状态类型**（bottleneck 瓶颈 / precondition 前置 /
  irreversible-risk 不可逆风险 / goal-progress 目标推进 / constraint-sensitive 约束敏感）、
  填**置信度**（0–1）、确认/驳回。顶部可按类型和覆核状态筛选。

### 6.5 约束注入

- 顶部三个折叠条列出全部模板池：11 条显式 + 5 条隐式 + 5 条请求措辞。
- 左列表显示当前可浏览的历史状态；详情页可对比同一状态的约束版与请求版。
  historical 页面上的预期决策只是开发视图；formal `expected_decision` 必须来自独立、版本化的
  point×variant policy truth，不能由模板类型或 recovery status 自动推出。
- 每栏下方可**改写目标文本**、选**约束类型**（safety/resource/instruction/
  environment/temporal/reversibility），点保存写入 overlay。
- 注意：改写只是标注资产；要让改动进入训练文件，需点**物化（运行 assemble）**重新组装。

### 6.6 候选/反事实

- 先在顶部选一个状态。S4 v2 共 300 条/50 states，保存 candidate ID、state、bid、
  canonical action、category、source、snapshot hash 和 legality。类别（含 decoy）只是提案覆盖
  假设，不是 effect/recovery/safety 标签。
- 页面也可浏览 4 类 historical synthetic-flip DPO 坏例子；它们不是部署时 legal/on-policy
  错误。正式 DPO 要求至少 50% 来自 legal candidates 或模型自身错误。
- 底部可**人工添加候选**（描述 + raw action + 备注）和删除，全部走 overlay。
- LLM 若接入只能提案；bid legality 由 snapshot 验证，label 必须由 point probe 产生。

### 6.7 Undo 标注（核心页）

- 页面必须区分 formal point、legacy class-smoke 和 quarantine。当前 formal/manifest 均为 0 行；
  历史 32/30 不能因 UI 显示而升级。
- formal 详情应展示唯一 point ID、state/action instance、pre/post/final signal、solver set、预算、
  undo trace、effect/recovery、cost 和 residual。旧“当前 action-type 标签”仅供兼容浏览。
- Probe authoring 可填写动作、signal、undo sequence、预算与安全等级，但不能直接填写最终
  label。人工覆核只能确认/reject evidence 或记录意见；冲突样本 formal export 必须排除。

### 6.8 Teacher 蒸馏

- 顶部：limit 输入 + 运行按钮（需要单独批准的 teacher key）+ 覆盖率提示。当前 distilled=0，
  formal source 也为 0，因此 teacher coverage 未定义（`null`）；页面能力不是已完成蒸馏结果。
- 详情：模板 prose vs teacher prose、结构与 evidence 一致性检查、完整序列和人工审阅。
  template fallback 必须显式进入独立分组/quarantine，不能静默混入 formal teacher set。

### 6.9 质量统计

- 数字块：按 asset tier 分开的样本量、成功率、grounding point 数、蒸馏覆盖率、teacher 一致率、
  反事实覆盖率、低质量样本数。
- 条形图：effect/recovery 分布、决策分布、约束风格、DPO source、动作类和 split；
  legacy 混合 label 必须单列。
- **决策矩阵**：action × variant → decision 的完整对照表（检查数据是否配比合理）。
- **低质量样本表**：自动质检抓出来的问题条目及原因（缺标签、措辞与标签矛盾、
  人工驳回等——矛盾检测复用的是蒸馏 QC 的同一套规则）。
- 三个按钮：跳转导出、**重建 HTML 报告**（重新生成 `outputs/dataset_viz.html`，
  一个可以脱机发给别人看的自包含网页）、重新计算。

### 6.10 数据集浏览器

- 顶部：全文搜索 + 7 个下拉筛选（action/variant/style/decision/可逆性/split/覆核状态）。
- 详情应展示 `state → candidate → transition → probe point → effect/recovery → teacher → split`。
  当前没有 formal point，因此该正式链条没有可展示样本；historical lineage 必须标 legacy。
- 「导出」是 admission gate：当前 formal 导出应为空/阻塞，不得回退历史 label。

### 6.11 Jobs

- 所有后台任务的历史与实时日志（日志里的 API key 已自动打码）。
- 运行中的任务有**终止**按钮。服务重启后历史仍在（读
  `outputs/workbench/jobs.jsonl`），中断的任务标记为"中断"。

---

## 7. 命令行等价操作

界面上每个按钮背后都是一条 CLI 命令，不开浏览器也能干活：

```bash
cd /workspace/iris

python -m revact.cli serve                    # 启动工作台（本教程主角）
./ci.sh                                       # 自检：测试数以本次 pytest 实测为准

# —— 离线可跑 ——
python -m revact.cli collect --mock --seeds 0            # mock 采集轨迹
python -m revact.cli probe --list                        # 列出当前注册的探针代码
python -m revact.cli probe --mock shopping.add_to_cart   # mock 探针
python -m revact.cli inspect                             # 轨迹/关键状态统计
python scripts/audit_formal_readiness.py --allow-blocked # 当前输出 ready=false；仅放宽退出码
python -m revact.cli assemble                            # formal point=0 时应非零退出/阻塞
python -m revact.cli train --dry-run                     # formal split=0 时应非零退出/阻塞
python -m revact.cli eval --dry-run                      # formal test=0 时应非零退出/阻塞
python -m revact.cli viz                                 # 生成静态报告 outputs/dataset_viz.html

# —— 待独立批准的 live 命令（以下未构成当前实验结果）——
source scripts/export_webarena_env.sh
conda run -n agentlab python -m revact.cli collect --task-file data/raw/pilot_task_ids.json --seeds 0 --wa-judge deepseek
conda run -n agentlab python -m revact.cli reach          # 直达风险状态
conda run -n agentlab python -m revact.cli crawl --cap 40 # 爬商品 URL
conda run -n agentlab python -m revact.cli scale          # 多商品规模化
conda run -n agentlab python -m revact.cli probe --all-nondestructive  # live 非破坏探针

# —— 待预算批准且 formal 输入非空后才运行 teacher ——
# export DEEPSEEK_API_KEY=...   # 只放进进程环境，不写文件/日志
# python -m revact.cli distill --limit 10
```

工作台**独有**的能力（CLI 没有的）：人工标注 overlay、质量报告面板、
应用标注的导出（train/val/test + dataset card + 审计）。

---

## 8. 数据文件长什么样

所有数据都是 **JSONL**（每行一个 JSON 对象），用文本编辑器就能打开。

```
data/
├── raw/
│   ├── trajectories/webarena.48_seed0.jsonl   # 轨迹：一行=一步
│   │     {"trajectory_id":"webarena.48_seed0","step_id":1,"action":"click('227')",
│   │      "url_after":"...","reward":0.0,"obs_after_axtree":"RootWebArea ...","screenshot":"raw/screenshots/..."}
│   ├── trajectories_meta.jsonl                # 一行=一条轨迹的总结（成败/步数/得分）
│   └── state_bank/
│       ├── shopping_key_states.jsonl          # 关键状态（含 replay_prefix 复现路径）
│       └── {pilot,scaled}_reached_states.jsonl# 风险状态（含 risky_action 和 safe_answer）
├── grounded/
│   ├── probe_points.jsonl                     # formal point；当前 0 行
│   │     {"schema_version":"iris.grounding.point.v1","probe_point_id":"...","state_id":"...",
│   │      "action_instance_id":"...","effect_status":"CHANGED",
│   │      "recovery_status":"RECOVERED","undo_cost_steps":1,"budget_k":12,
│   │      "solver_set":[...],"pre_signal":{...},"post_signal":{...},"final_signal":{...}}
│   ├── POINT_MANIFEST.jsonl                   # 与 formal point 1:1；当前 0 行
│   ├── reversibility.jsonl / MANIFEST.jsonl   # 冻结 legacy：32 / 30，不可作 formal join
│   ├── smoke/                                 # 未来普通 probe 的独立 class-smoke 输出
│   └── quarantine/                            # 2 条无 provenance + 30 条 class-smoke 索引
├── eval/
│   ├── truth.jsonl                            # point×variant normative truth；当前未物化/有效 0 行
│   └── TRUTH_MANIFEST.jsonl                   # 与 truth 1:1；当前未物化/有效 0 行
├── train/
│   ├── formal/                                # 只收 point-grounded rows；当前均为 0 行
│   │   ├── iris_sft_point_v1.jsonl
│   │   ├── iris_sft_multiturn_point_v1.jsonl
│   │   ├── iris_sft_distilled_point_v1.jsonl  # 唯一正式 teacher 路径；当前未物化/有效 0 行
│   │   └── splits/
│   ├── quarantine/                            # historical/mock/failed/legacy 索引或开发产物
│   └── {sft,dpo}/                             # 冻结 historical 资产；formal 命令不会读取
└── annotations/                               # 你的人工标注（overlay，append-only）
      grounded.jsonl / sample.jsonl / key_state.jsonl / constraint.jsonl / ...
      {"kind":"grounded","target_id":"add_to_cart-cee07921",
       "payload":{"review_status":"confirmed","confidence":0.95},"timestamp":"..."}
```

规则记三条：原始 legacy JSONL 不覆盖；formal admission fail closed；人工 annotation 只能
排除/审阅，不能生成 probe truth。同一 target 的多条 annotation 可按时间合并，但不会因此
改变 formal effect/recovery。

导出产物在 `outputs/workbench/exports/<时间戳>__<名称>/`，
job 日志在 `outputs/workbench/jobs/`。这些都不进 git。

---

## 9. 常见问题 FAQ

**Q1：页面打不开 / 一直"连接中…"**
确认终端里 serve 还在运行、端口号和浏览器地址一致；远程机器用 SSH 转发（见 4.1）。

**Q2：右上角显示 offline，live 按钮点了报"需要 live WebArena"**
这是门控在保护你：没配置真实站点。要么先 source 环境脚本再启动服务，
要么在「全局配置→环境」里填 WA_SHOPPING。只想离线用则无需理会。

**Q3：live 任务日志里报 `ModuleNotFoundError: browsergym`**
服务是用普通 Python 启动的，live 子进程继承不到浏览器依赖。
用 `conda run -n agentlab python -m revact.cli serve` 重启服务（见 5.3 第 1 步）。

**Q4：蒸馏报 "API key not found: export DEEPSEEK_API_KEY=..."**
没给 teacher 配 key。「全局配置」teacher 卡片里填 api_key 后点应用，或启动前 export。

**Q5：任务失败了怎么排查？**
「Jobs」页点开那条任务看日志，最后几行通常就是原因。修好后回 Pipeline 单步重跑即可
（每步产物落盘，不用从头再来）。

**Q6：mock 跑完，Undo 标注页怎么没有新记录？**
mock/fixture 记录只能出现在独立开发视图或 quarantine；formal grounding 不应新增。如果
mock 被写进 `probe_points.jsonl` 或 formal export，应视为 gate 回归。

**Q7：我驳回/标错了，怎么撤销？**
再追加一条带纠正原因的 annotation；不要删除或改写旧行。append-only 历史是审计链的一部分。

**Q8：为什么我在 Undo 页写了覆核标签，导出的样本反而少了？**
见 6.7 的规则：人工覆核与行为标签**冲突**的样本会被整条排除进 excluded.jsonl
（宁可少一条数据，不让意见污染行为标签）。检查你的覆核是否写错。

**Q9：轨迹列表里的 `mock.` 开头的条目能删吗？**
不要删除原始轨迹或手改 meta。前缀不构成可靠隔离；应依赖 `environment_origin/is_mock/run_id`
和 formal export gate，并把 legacy mock 登记到 quarantine。回滚只删除生成的 quarantine 索引，
不改源 JSONL。

**Q10：`dataset_viz.html` 和工作台什么关系？**
那是一张**自包含静态快照**（数据和截图都内嵌，单文件可发给别人离线看），
由 `python -m revact.cli viz` 或质量统计页的按钮生成；工作台是**活的**（实时读文件、可操作）。

**Q11：怎么确认没把 key 泄漏进仓库？**
`git grep -E "sk-[A-Za-z0-9]{16,}"` 应无任何结果；`configs/workbench.local.json`
已被 gitignore 且保存时剥离 key 值；job 日志展示前自动打码。这三层都有测试覆盖。

**Q12：Pipeline 和其他页面是什么关系？其他页面会不会也在"采集数据"？**
把「Pipeline」理解成总控台：它负责启动 mock/live 流程、调 CLI 命令、产生产物。
大多数其他页面是专门视图，负责读取这些产物、筛选、审阅、标注和导出。例如「成功轨迹」
只读取 `data/raw/trajectories/*.jsonl` 和 `trajectories_meta.jsonl`，不会在这个页面里重新采集；
采集要在 Pipeline 点 `collect_mock` / `collect_live`，或命令行跑 `revact collect`。少数页面也有
"运行"按钮（如 Undo 标注、Teacher 蒸馏、约束注入的 assemble），但它们底层仍然是转调同一套
Pipeline action / CLI 命令，不会复制一份业务逻辑。

**Q13：「成功轨迹」里的"选入下一阶段"和"查看关键状态 →"有什么区别？**
"选入下一阶段"是给**整条 trajectory**写一条人工 overlay，形如
`{"kind":"trajectory","target_id":"webarena.48_seed0","payload":{"selected":true}}`。
它不会重新采集、不会重新挖 key states，也不会改原始轨迹文件，只表示"我人工看过，这条轨迹
值得后续审阅"。"查看关键状态 →"只是跳到「关键状态」页，去看 collect 阶段已经自动挖出来的
key states。一个 trajectory 是完整行动历史；key state 是这条历史里的某一个页面状态。

**Q14：轨迹里的 key state 是怎么挖出来的？为什么用关键词匹配？**
旧 S2 只匹配 `Add to Cart`、`Place Order`、`Delete Address` 三类英文关键词，产生的 241 条
state 是 legacy 偏置资产，不能声称高召回。新路径枚举当前 snapshot 的 legal interactive
controls，以执行后的 UI/API/DB signal diff 确认 mutation；关键词和 LLM 只做排序。换站点时
应复用 detector 主逻辑，只新增站点 signal/adapter，而不是扩一张人为封顶的关键词表。

**Q15：「约束注入」是 Pipeline 流程的一步，还是人工选择？**
它是 Pipeline 的真实阶段，但页面主要用于预览和审计。真正把约束目标写进 SFT/DPO 的是
`assemble`：对同一个 reached state 自动生成 `constraint` 和 `request` 两个目标变体。
模板不是人逐条选择的，而是 `assemble.build_goal()` 用 `(state, variant)` 的 hash 确定性抽取。
例如同一个商品页 `Add to Cart` 状态可生成约束版和请求版目标。expected decision 必须由
独立的 policy constraint/authorization/goal necessity truth 决定，不能由 recoverability 单独
推出。页面里的人工改写会存为
`constraint` overlay，目前主要是审计/扩展资产，不会直接改写任何 formal JSONL；只有重新
assemble 并通过 point/truth/prompt admission gate 后才能进入正式产物。

**Q16：「候选/反事实」页面和 Pipeline 里的"候选动作生成"是什么关系？**
S4 v2 已自动枚举 snapshot 中的 legal controls，并物化 300 条/50 states。每条 candidate 有
精确 bid 与 snapshot hash；category（包括 decoy）只是 proposal metadata。它们尚未关联
point label 或 on-policy model error，不能进入 formal DPO。旧四类 `false_safe/over_block/
goal_violation/wrong_reversibility` 都是 synthetic flip，应与未来 legal/on-policy negatives 分层。

**Q17：「Undo 标注」里的 `●` 和 `○` 分别是什么意思？**
这是 legacy UI 的 class-level 展示惯例，`●` 曾表示“该 action type 的最新非 UNKNOWN 行”。
这个 join 已禁止用于 formal 训练/评测：每条样本必须精确引用唯一 `probe_point_id`，历史行
只能审计，不能靠后一行覆盖前一行，也不能把 action class 当作 point truth。

**Q18：`S→A→S′ 证据链` 是什么意思？为什么有时显示 `baseline="—"`？**
概念上 `S` 是动作前 baseline，`A` 是被测试动作，`S′` 是执行后状态，
再执行 undo 得到 `S″`，看 `S″` 是否恢复到 `S`。例如 `add_to_cart` 可能显示
`baseline=0 → after_action=1 → after_undo=0`，表示购物车 0 件、加购后 1 件、撤销后回到 0 件。
如果某些 reddit 探针顶部摘要显示 `baseline="—"`，通常是 UI 汇总字段只识别通用键
`baseline/after_action/after_undo`，而该 probe 的 evidence 用的是更具体的键，如
`baseline_subscribed/after_action_subscribed/after_undo_subscribed`；下面的详细 evidence 表才是
历史记录中的调试证据；它不弥补缺失的 point provenance，也不把该行升级为 formal。

**Q19：「数据集浏览器」应该怎么看？**
它能浏览 historical single/multiturn/DPO、candidate、grounding 和 teacher 资产，并显示其
可用 lineage。formal 视图要求 `state → candidate → transition → probe point → effect/recovery
→ teacher → split`；当前 point=0，所以没有完整 formal 样本。页面不采集轨迹，也不跑 probe；
浏览能力不代表可以导出 release。

**Q20：探针是人工写的，怎么能证明没有别的动作能恢复状态？这是 bug 吗？**
不能证明。formal 结论是 point-level、agent/privilege/budget/signal/solver-relative。任一 solver
成功给出 `RECOVERED` 的构造性证据；全部失败只给
`NOT_RECOVERED_WITHIN_BUDGET`。正式负例要保存 deterministic controller、深度 2–3 BFS、
多 seed LLM attacker 的 trace（可用时再加 API/DB diff）。旧 place-order 行无 probe provenance，
已 quarantine，正式状态不是 `IRREVERSIBLE`。

---

## 10. 安全须知（必读）

1. **API key 三不原则**：不进命令行参数、不进磁盘配置、不进日志。
   在界面填的 key 只活在服务进程内存里，服务一停就没了（需要重填或用环境变量）。
2. **破坏性操作双闸门**：真实下单、删地址、发布、admin 写操作等不得从普通工作台路径
   执行。必须在命令行同时给出 `--commit` 参数
   **和** `REVACT_ALLOW_DESTRUCTIVE=1` 环境变量（缺一个都强制 dry-run），
   且按项目规定需逐批人工批准后才可执行。别绕过这个设计。
3. **服务只绑本机**（127.0.0.1）。它没有登录认证，不要用 `--host 0.0.0.0`
   暴露到公网；跨机器访问一律走 SSH 端口转发。
4. **数据不可变边界**：流水线产物只读、人工操作全走 overlay、导出带完整审计
   （excluded.jsonl + MANIFEST 溯源）。这不是繁文缛节——本项目的可信度就建立在
   "point-level effect/recovery 证据可审计、不可被悄悄改写"上。

---

*配套阅读：工程边界与扩展点 → [`workbench.md`](workbench.md)；
流水线各阶段原理 → [`plan/S1-S8完整流程文档.md`](plan/S1-S8完整流程文档.md)。*
