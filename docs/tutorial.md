# IRIS 数据集构建工作台 · 新手上手教程

> 这份教程假设你**完全不了解**这个项目。读完第 1–4 章你就能打开工作台浏览数据；
> 跟着第 5 章的三个演练做一遍，你就能独立完成"采集 → 标注 → 质检 → 导出"的完整数据集构建。
> 项目的研究背景见 [`../../RevAct重定位方案/06-IRIS项目计划书.md`](../../RevAct重定位方案/06-IRIS项目计划书.md)，
> 工程边界速查见 [`workbench.md`](workbench.md)，本文只管"怎么用"。

---

## 目录

1. [三分钟看懂这个项目在做什么](#1-三分钟看懂这个项目在做什么)
2. [名词速查表（先扫一眼，忘了随时回来查）](#2-名词速查表)
3. [准备工作：目录认路 + 两档运行模式](#3-准备工作)
4. [第一次启动：5 分钟跑起来](#4-第一次启动)
5. [三个实战演练（从零到导出数据集）](#5-三个实战演练)
6. [十一个页面逐个讲解](#6-十一个页面逐个讲解)
7. [不用界面：命令行等价操作](#7-命令行等价操作)
8. [数据文件长什么样](#8-数据文件长什么样)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [安全须知（必读）](#10-安全须知必读)

---

## 1. 三分钟看懂这个项目在做什么

### 1.1 背景故事

我们在训练一种"会上网操作的 AI 代理"（web agent）：它能在购物网站上搜索、加购物车、下单。
危险在于：**有些操作点下去就收不回来了**。把商品加进购物车 → 随时可以移除（**可逆**）；
点"Place Order"真的下单 → 顾客账户里没有取消按钮（**不可逆**）。

IRIS 项目的核心想法：训练模型在动手之前先判断"这个动作能不能撤销、撤销要几步"，
而且这个"可逆性"标签**不靠人拍脑袋、不靠 GPT 猜**，而是靠程序真的去网站上
**执行一次动作 → 再尝试撤销 → 对比前后状态**测出来的（叫 execute-then-undo，行为测量）。

这个仓库里的 `revact/` 就是把上面的想法变成一条**数据集生产流水线**，
而你现在要学的"工作台"（workbench）是这条流水线的**图形化驾驶舱**：
在浏览器里点按钮跑流水线、逐条检查数据、人工打标、最后导出训练集。

### 1.2 流水线一张图

数据从左到右流动，每一步的产物都存成文件（第 8 章讲文件长什么样）：

```
①环境初始化 → ②成功轨迹采集 → ③关键状态采集 → ④约束注入 → ⑤候选动作
   (检查配置)     (AI 代理逛网站,     (挑出"高危页面",   (给任务加一句      (每个状态列出
                  记录每一步)         如结账页)          "但是不许下单")     可选的动作)
        → ⑥反事实动作 → ⑦undo 可逆性标注 → ⑧teacher 蒸馏 → ⑨质量校验 → ⑩导出
          (构造"错误示范"   (真去执行+撤销,      (让大模型把理由    (统计+挑出   (train/val/test
           作对比教材)       测出可逆标签⭐)       写通顺,结论不动)   坏样本)      + 数据集说明书)
```

⭐ 第⑦步是整个项目的灵魂：可逆性标签来自真实执行，其他一切都围绕它展开。

### 1.3 最终产出什么

两种训练数据（喂给 Qwen2.5 这类模型做微调）：

- **SFT 样本**：一段对话——用户给任务+网页内容，助手输出一段带标签的思考
  （`<observation>观察 <reasoning>推理 <prediction>预测 <reversibility>可逆性
  <decision>决策</think><answer>动作`）。
- **DPO 偏好对**：同一个场景下"好回答 vs 坏回答"成对出现（坏回答=反事实，
  比如把不可逆动作当成可逆并执行了），教模型分辨对错。

---

## 2. 名词速查表

| 名词 | 大白话解释 |
|---|---|
| **trajectory（轨迹）** | AI 代理做一次任务的完整录像：每一步点了什么、页面变成什么样、有没有拿到分 |
| **key state（关键状态）** | 轨迹里值得注意的页面，比如出现了"Add to Cart / Place Order / Delete"按钮的页面 |
| **reached state（风险状态）** | 用固定路线直接导航到的高危页面（不用跑完整任务），是造样本的"锚点" |
| **约束（constraint）** | 给任务加的限制条款，如"看看就行，别加购物车"。有 11 种显式说法 + 5 种隐式说法（隐式=整句没有"不要"二字），防止模型靠背句式作弊 |
| **候选动作（candidate）** | 在某个状态下模型可能采取的动作清单：专家动作、安全替代动作等 |
| **反事实（counterfactual）** | 故意构造的错误回答，共 4 类（把危险当安全、无脑拒绝、明知违规还执行、可逆性标签写反），用于 DPO 的"坏例子" |
| **grounded / grounding** | "有据可查的"：标签来自环境里的真实执行测量，不是人或 LLM 的意见 |
| **probe（探针）** | 一段自动化程序：到指定页面 → 执行动作 → 尝试撤销 → 读网站后端信号（如购物车行数）→ 判定可逆性。本项目有 13 个探针 |
| **undo** | 撤销。探针会记录撤销用了几步（undo_steps）、每一步点了什么（undo_actions） |
| **可逆性标签** | `REVERSIBLE` 可逆 / `REVERSIBLE_WITH_COST(k)` 可逆但要 k 步 / `PARTIALLY_RECOVERABLE` 只能部分挽回 / `IRREVERSIBLE` 不可逆 / `NO_EFFECT` 动作没生效 / `UNKNOWN` 测不出来 |
| **oracle 决策** | 由确定性规则算出的"标准答案决策"：`EXECUTE` 直接做 / `VERIFY` 先探查 / `CONFIRM` 先问用户 / `AVOID` 拒绝并绕开 |
| **teacher 蒸馏（distill）** | 请一个强模型（如 DeepSeek）把样本里模板化的"思考文字"改写得更自然。**结论（可逆性/决策/动作）被钉死不许改**，改了就会被质检打回 |
| **SFT / DPO** | 两种微调数据格式：SFT=标准问答示范；DPO=好坏成对的偏好数据 |
| **split** | 数据集切分：train 训练 / val 验证 / test 测试（test 按商品隔离，防止模型背题） |
| **overlay（标注叠加层）** | 你在工作台里做的一切人工操作（确认/驳回/改写/覆核）都存成独立的小文件，**绝不改动流水线原始产物**，导出时才合并生效 |
| **mock 模式** | 不连真实网站，用一个模拟的假购物网站离线演练，没网没账号也能跑通全流程 |
| **live 模式** | 连真实部署的 WebArena 购物网站（本项目的实验沙盒），采集真数据 |
| **job（任务）** | 你在界面上点"运行"后，后台实际执行的那条命令。每个 job 有独立日志 |

---

## 3. 准备工作

### 3.1 目录认路

```
iris/                          # ⭐ 工程根目录，后面所有命令都在这里执行
├── revact/                    #    Python 源码（cli / envs / grounding / data / train / server / ui ...）
├── data/                      #    数据产物（raw 原始 / grounded 可逆性标签 / train 训练集 / annotations 人工标注）
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
| 能做什么 | 浏览/审查/标注全部已有数据、mock 全流程演练、质检、导出 | 额外：真实采集轨迹、真实探针测可逆性、teacher 蒸馏 |
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
`shopping · 25 轨迹 / 92 SFT / 230 DPO · offline · 0 任务运行中`
——`offline` 表示当前没连 WebArena（离线档），不影响浏览和标注。

几个通用的界面元素，所有页面都长这样：

- **彩色圆点** = 阶段/任务状态：灰=未开始，蓝闪=运行中，绿=成功，橙=部分完成，红=失败。
- **彩色胶囊（badge）**：绿 `REVERSIBLE`/`EXECUTE`、红 `IRREVERSIBLE`/`AVOID`、
  橙 `PARTIALLY_RECOVERABLE`/`CONFIRM`、蓝 `VERIFY`——颜色语义全站统一。
- **左列表 + 右详情**：大部分页面左边是条目列表（点击切换），右边是详情。
- **虚线框"人工覆核"**：这就是打标区，点"确认/接受/待复核/驳回"即写入一条标注。
- 右下角会弹**黑色小提示条**（toast）告诉你操作成功还是失败。

---

## 5. 三个实战演练

### 5.1 演练一：离线审查数据 + 打标 + 导出（10 分钟，什么都不用装）

目标：把仓库里已有的 92 条 SFT 样本过一遍人工审查，剔掉一条，导出正式数据集。

1. **看全局**：打开「Pipeline」页。你会看到 10 张阶段卡片，大部分是绿色"成功"
   （因为仓库里已经带了一批数据）。点任意卡片，下方显示这个阶段的说明、
   可执行操作和产物文件清单（路径/行数/更新时间）。

2. **看轨迹**：切到「成功轨迹」页。左边 25 条轨迹，绿点=成功。点开一条成功的
   （比如 `webarena.48_seed0`），右边出现逐步表格；**点击任意一行 step**，
   下方会展开这一步之后的页面快照（axtree 文本）和网页截图。
   这就是"AI 代理逛网站的录像回放"。

3. **看灵魂数据**：切到「Undo 标注」页。左边是每一次探针运行记录，
   `●` 开头的是"当前生效标签"。点开 `add_to_cart`：
   右边能看到 S→A→S′ 证据链（`baseline=0 → after_action=1 → after_undo=0`，
   意思是购物车 0 件 → 加购后 1 件 → 撤销后回到 0 件，所以判 REVERSIBLE）、
   实测的 undo 动作序列、甚至当时的网页截图。

4. **打你的第一个标**：还是这条记录，滚到底部虚线框"人工覆核"：
   置信度填 `0.95`，备注填 `证据链完整`，点**确认**。右下角弹绿色提示即成功。
   这条标注被追加写入 `data/annotations/grounded.jsonl`，原始数据一个字节没动。

5. **审样本**：切到「数据集浏览器」页。顶部一排筛选器，
   把 `decision` 选成 `AVOID` 看看约束类样本。点开任意一条，右边是完整
   **lineage（血缘）**：来源状态 → 约束文本 → grounded 标签 → DPO 反事实 → teacher 状态，
   往下是完整的 assistant 输出序列（带彩色标签高亮）和可折叠的 DPO 好坏对比。
   随便挑一条你觉得不好的，在底部"人工覆核"点**驳回**，备注写原因。

6. **导出**：同页顶部点开「导出最终数据集」折叠条：
   名称填 `my-first-release`，val 比例保持 `0.15`，点**导出 train/val/test + dataset card**。
   几秒后出现绿色结果框，告诉你导出目录和各 split 条数（被你驳回的样本会进
   `excluded.jsonl` 审计文件而不是训练集）。去
   `outputs/workbench/exports/<时间戳>__my-first-release/` 看看：
   `sft_train/val/test.jsonl`、`dpo_train.jsonl`、`samples.csv`、`stats.json`、
   还有一份自动生成的 `dataset_card.md`（数据集说明书）。

恭喜，你已经完成了一次完整的"审查 → 标注 → 导出"闭环。

### 5.2 演练二：mock 全流程（15 分钟，体验流水线怎么转）

目标：不连任何真实网站，用模拟环境把流水线从头到尾跑一遍。

1. 打开「Pipeline」页，左上角"全流程链"下拉框保持 **mock（离线可跑）**。
2. 点**全流程运行**。工作台会按顺序执行 7 步：
   `环境检查 → mock 采集 → mock 探针 → assemble 组装 → split 切分 → 质量统计 → 导出`。
   每步启动时页面会跳到 Jobs 页显示实时日志；全部跑完后回到 Pipeline 页，
   底部"链运行日志"能看到每一步的 ✓/✗。
3. 中途想看细节：切到「Jobs」页，点任意任务看完整日志（比如 mock 探针的日志里
   能看到 `add_to_cart -> REVERSIBLE {baseline:0, after_action:1, after_undo:0}`）。
4. 跑完后去「成功轨迹」页，列表里多了几条 `mock.` 开头的轨迹——这就是模拟环境
   产生的演练数据（和真实的 `webarena.` 轨迹靠前缀区分，不会混淆）。

> 两个刻意的设计，别误会成 bug：
> ① mock 探针的结果**不写入** grounded 标签文件（防止演练数据污染真实标签）；
> ② mock 环境里 `compare_add` 会报 `UNKNOWN(action affordance not found)`，
>   这是模拟站点没实现该控件，属正常现象。

也可以不跑整链，只单步跑：点某张阶段卡片 → 在"操作"区点对应按钮
（每个按钮旁边标了它需要什么：`mock` 徽章=离线可跑，`live`=要真实环境，`key: xxx`=要 API key）。
**从某阶段继续**：先点选一张卡片，再点顶部"从选中阶段继续"，链会从那一步开始往后跑。

### 5.3 演练三：live 真实采集（需要 WebArena 沙盒 + DeepSeek key）

前提：你有一个部署好的 WebArena shopping 站点，以及 DeepSeek API key。

1. **在正确的环境里启动服务**（关键！live 任务是服务的子进程，
   继承服务的 Python 环境，所以必须用装了 browsergym 的 `agentlab` 环境启动）：

   ```bash
   cd /workspace/iris
   source scripts/export_webarena_env.sh
   conda run -n agentlab python -m revact.cli serve
   ```

   刷新页面，右上角应从 `offline` 变成 `live 就绪`。

2. **填 key**：打开「全局配置」页。三张模型卡片：
   - **策略模型**（采集轨迹用）：base_url 填 `https://api.deepseek.com/v1`，
     model 填 `deepseek-v4-pro`，api_key 栏粘贴你的 key；
   - **Teacher 模型**（蒸馏用）：model 填 `deepseek-chat`，key 同上；
   - **Judge**（判轨迹成败用）：模式保持 `deepseek`。

   点**应用配置（内存）**。想下次启动少填几项，可再点**保存到本地文件**——
   放心，保存时 key 值会被剥离，文件里只留环境变量名（见第 10 章）。

   > 也可以完全不在界面填 key：启动服务前 `export DEEPSEEK_API_KEY=sk-...` 效果相同。

3. **跑 live 链**：回「Pipeline」页，下拉框切 **live**，点**全流程运行**。
   live 链 = `环境检查 → live 采集 → reach 直达风险状态 → live 非破坏探针
   → assemble → split → 蒸馏 → 质量统计 → 导出`。
   live 采集一条轨迹要几分钟，去 Jobs 页盯日志即可；也可以只单步跑你需要的阶段。

4. **蒸馏单独跑**：「Teacher 蒸馏」页，limit 填 10（先小批量试水控制成本），
   点**运行蒸馏**。跑完后该页出现逐条对比：左列模板措辞、右列 teacher 措辞，
   下方三行 pinned 结论逐字校验（都应显示 ✓；出现 ✗ 说明 QC 漏网，请驳回该条）。

---

## 6. 十一个页面逐个讲解

### 6.1 Pipeline（总控台）

- 10 张阶段卡片 = 第 1.2 节那张图。每张卡片显示：序号与文档编号（S1–S8）、
  状态圆点、`real/partial/placeholder` 实现程度徽章、最近一次运行信息。
- 点卡片 → 下方详情：阶段说明、**操作按钮**（带参数输入框，如 seeds、limit）、
  **产物文件表**、最近 job 入口。
- 顶部三件套：`全流程链`选 mock/live、`全流程运行`、`从选中阶段继续`。
- **placeholder 是什么**：第⑤阶段"候选动作生成"目前没有自动生成的后端实现
  （点了会明确告诉你，并给出接入位置），候选动作靠已有数据推导+人工添加，见 6.6。

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

- 每条 key state 显示：所属轨迹与步号、任务目标、URL、页面快照、
  **"为什么是关键状态"**（命中了哪些高危动作关键词——如实告诉你这是浅层规则，
  精确判定靠第⑦步探针）。
- 人工覆核区可以：选**状态类型**（bottleneck 瓶颈 / precondition 前置 /
  irreversible-risk 不可逆风险 / goal-progress 目标推进 / constraint-sensitive 约束敏感）、
  填**置信度**（0–1）、确认/驳回。顶部可按类型和覆核状态筛选。

### 6.5 约束注入

- 顶部三个折叠条列出全部模板池：11 条显式 + 5 条隐式 + 5 条请求措辞。
- 左列表是 48 个可注入状态；详情页左右两栏对比同一状态的两个变体：
  **注入约束版**（预期 oracle 决策=AVOID）vs **请求版**（预期=EXECUTE/CONFIRM）——
  这就是"注入前后行为变化"的直观对照。
- 每栏下方可**改写目标文本**、选**约束类型**（safety/resource/instruction/
  environment/temporal/reversibility），点保存写入 overlay。
- 注意：改写只是标注资产；要让改动进入训练文件，需点**物化（运行 assemble）**重新组装。

### 6.6 候选/反事实

- 先在顶部选一个状态。上表=**候选动作**：专家动作（带 grounded 可逆性标签）
  和规则安全替代动作；下表=**反事实动作**：4 类 DPO 坏例子，实时用真实组装逻辑算出，
  每行可点"序列"看完整的错误示范文本。
- 底部可**人工添加候选**（描述 + raw action + 备注）和删除，全部走 overlay。
- 页面底部的橙字说明了当前的诚实边界：自动"提案新候选"尚未实现（placeholder）。

### 6.7 Undo 标注（核心页）

- 顶部两个按钮跑 mock/live 探针；折叠条里是 13 个探针的注册表
  （名字、破坏等级、后端信号、当前 live 标签、预期谱系）。
- 详情页从上到下：标签徽章（含是否当前生效）→ S→A→S′ 证据链 → residual_diff →
  实测 undo 动作序列 → 截图 → **训练字段预览**（`undoable / undo_action /
  reversibility_label / grounding_evidence`，即导出时每个样本携带的可逆性元数据）。
- 人工覆核区可给出**标签覆核意见**（reversibility_override）+ 置信度 + 理由。
  ⚠️ 重要规则：人工意见**不会改写**行为测出的标签；如果两者矛盾，
  相关样本在导出时会被**整条排除**并写进 excluded.jsonl 审计——想真正改标签，
  正确做法是修探针后重跑，而不是手改。

### 6.8 Teacher 蒸馏

- 顶部：limit 输入 + 运行按钮（需要 teacher key）+ 覆盖率提示。
- 详情：模板 prose vs teacher prose 三行对照表、pinned 结论逐字校验（✓/✗）、
  完整序列折叠、人工接受/驳回。被驳回的蒸馏条目在导出时回退用模板版。

### 6.9 质量统计

- 数字块：样本量、成功率、grounded 类数、蒸馏覆盖率、teacher-pinned 一致率、
  反事实覆盖率、低质量样本数。
- 条形图：可逆/不可逆分布、决策分布、约束风格分布、DPO 类型、动作类、split。
- **决策矩阵**：action × variant → decision 的完整对照表（检查数据是否配比合理）。
- **低质量样本表**：自动质检抓出来的问题条目及原因（缺标签、措辞与标签矛盾、
  人工驳回等——矛盾检测复用的是蒸馏 QC 的同一套规则）。
- 三个按钮：跳转导出、**重建 HTML 报告**（重新生成 `outputs/dataset_viz.html`，
  一个可以脱机发给别人看的自包含网页）、重新计算。

### 6.10 数据集浏览器

- 顶部：全文搜索 + 7 个下拉筛选（action/variant/style/decision/可逆性/split/覆核状态）。
- 详情 = 完整 lineage 血缘（见演练一第 5 步）。
- 「导出最终数据集」折叠条 = 正式出货口（见演练一第 6 步），下方列出历史导出。

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
./ci.sh                                       # 自检：代码检查 + 40 测试

# —— 离线可跑 ——
python -m revact.cli collect --mock --seeds 0            # mock 采集轨迹
python -m revact.cli probe --list                        # 列出 13 个探针
python -m revact.cli probe --mock shopping.add_to_cart   # mock 探针
python -m revact.cli inspect                             # 轨迹/关键状态统计
python -m revact.cli assemble && python -m revact.cli split   # 组装 SFT/DPO + 切分
python -m revact.cli train --dry-run && python -m revact.cli eval --dry-run  # 数据校验
python -m revact.cli viz                                 # 生成静态报告 outputs/dataset_viz.html

# —— 需要 live 环境（先 source env 脚本 + agentlab 环境）——
source scripts/export_webarena_env.sh
conda run -n agentlab python -m revact.cli collect --task-file data/raw/pilot_task_ids.json --seeds 0 --wa-judge deepseek
conda run -n agentlab python -m revact.cli reach          # 直达风险状态
conda run -n agentlab python -m revact.cli crawl --cap 40 # 爬商品 URL
conda run -n agentlab python -m revact.cli scale          # 多商品规模化
conda run -n agentlab python -m revact.cli probe --all-nondestructive  # live 非破坏探针

# —— 需要 API key ——
export DEEPSEEK_API_KEY=sk-...
python -m revact.cli distill --limit 10                  # teacher 蒸馏
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
├── grounded/                                  # ⭐ 行为测量结果
│   ├── reversibility.jsonl                    # 一行=一次探针运行：
│   │     {"action_type":"add_to_cart","label":"REVERSIBLE","grounding":"cart_item_count",
│   │      "evidence":{"baseline":0,"after_action":1,"after_undo":0,"undo_steps":1,
│   │                  "undo_actions":["click('1972')"],"residual_diff":{"count_delta":0}},
│   │      "probe_id":"add_to_cart-cee07921","timestamp":"..."}
│   └── MANIFEST.jsonl                         # 溯源账本（谁、何时、哪个版本的控制器测的）
├── train/
│   ├── sft/revact_sft.jsonl                   # SFT：{"sample_id","messages":[system,user,assistant],"meta":{...}}
│   ├── sft/revact_sft_distilled.jsonl         # 蒸馏版（措辞换了，结论逐字不变）
│   ├── dpo/revact_dpo.jsonl                   # DPO：{"pair_id","prompt","chosen","rejected","meta":{"pair_type":...}}
│   └── splits/{sft_train,sft_test,dpo_train}.jsonl
└── annotations/                               # 你的人工标注（overlay，append-only）
      grounded.jsonl / sample.jsonl / key_state.jsonl / constraint.jsonl / ...
      {"kind":"grounded","target_id":"add_to_cart-cee07921",
       "payload":{"review_status":"confirmed","confidence":0.95},"timestamp":"..."}
```

规则记两条就够：**`data/` 下除 annotations 外，工作台一律只读**；
同一 target 的多条标注按时间合并、后写的字段生效（所以"改标注"=再写一条新的）。

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
设计如此：mock 探针结果不落盘，防止演练数据污染真实 grounded 标签（见 5.2 注意事项）。

**Q7：我驳回/标错了，怎么撤销？**
再标一次即可——标注是 append-only，最新一条生效。彻底清除某类标注可手动编辑
`data/annotations/对应文件.jsonl`（删掉对应行）。

**Q8：为什么我在 Undo 页写了覆核标签，导出的样本反而少了？**
见 6.7 的规则：人工覆核与行为标签**冲突**的样本会被整条排除进 excluded.jsonl
（宁可少一条数据，不让意见污染行为标签）。检查你的覆核是否写错。

**Q9：轨迹列表里的 `mock.` 开头的条目能删吗？**
可以不管（前缀天然隔离，导出不受影响）。想清掉就删
`data/raw/trajectories/mock.*.jsonl` 并把 `trajectories_meta.jsonl` 里对应行去掉。

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
S2 collect 每走一步都会读取当前页面的 AXTree 文本，粗略匹配 `Add to Cart`、`Place Order`、
`Delete Address` 等 affordance 关键词；命中后记录 `state_id`、`trajectory_id`、`step_id`、
`replay_prefix`、页面快照和 fingerprint。这个匹配只是第一层候选生成器，目标是便宜、稳定、
高召回地把"可能有风险动作的页面"捞出来；最终可逆性不靠关键词决定，而靠后面的 grounded
execute-then-undo 探针。换任务时通常不按 task id 写规则，而是按**动作类/站点机制**适配：
新增 action type 的关键词、signal、undo controller、probe 和 assemble 元数据。

**Q15：「约束注入」是 Pipeline 流程的一步，还是人工选择？**
它是 Pipeline 的真实阶段，但页面主要用于预览和审计。真正把约束目标写进 SFT/DPO 的是
`assemble`：对同一个 reached state 自动生成 `constraint` 和 `request` 两个目标变体。
模板不是人逐条选择的，而是 `assemble.build_goal()` 用 `(state, variant)` 的 hash 确定性抽取。
例如同一个商品页 `Add to Cart` 状态会生成"只看商品但不要加购物车"和"请加购物车"两个目标，
前者通常 `AVOID`，后者在 `add_to_cart=REVERSIBLE` 时通常 `EXECUTE`。页面里的人工改写会存为
`constraint` overlay，目前主要是审计/扩展资产，不会默认直接改写 `revact_sft.jsonl`。

**Q16：「候选/反事实」页面和 Pipeline 里的"候选动作生成"是什么关系？**
当前 Pipeline 的 S4 "LLM 候选动作提案"还是 placeholder，没有真正实现自动枚举页面动作或
LLM 提案。这个页面展示的是现有管线已经能推导出的内容：每个 reached state 自带的
`expert_risky`（如 `click('123')` Add to Cart）和 `safe_alternative`（如 `go_back()`），再加上
`assemble.py` 里的 DPO rejected builders 实时生成的反事实。反事实包括 `false_safe`、`over_block`、
`goal_violation`、`wrong_reversibility`。页面里人工添加的 candidate 只进
`data/annotations/candidate.jsonl`，不自动进入 assemble 物化。

**Q17：「Undo 标注」里的 `●` 和 `○` 分别是什么意思？**
同一个 action type 可能跑过多次 probe。`●` 表示这条 probe run 是该 action type 的
**当前生效标签来源**；`○` 表示历史记录，仅作审计。生效规则是：每个 action type 取最新的
非 `UNKNOWN` 结果；如果全是 `UNKNOWN`，才取最新一条 `UNKNOWN`。所以 `○ reddit_vote
PARTIALLY_RECOVERABLE` 后面又出现 `● reddit_vote REVERSIBLE` 时，训练/统计会采用后一条。

**Q18：`S→A→S′ 证据链` 是什么意思？为什么有时显示 `baseline="—"`？**
这是 probe 的实测链路：`S` 是动作前 baseline，`A` 是被测试动作，`S′` 是执行后状态，
再执行 undo 得到 `S″`，看 `S″` 是否恢复到 `S`。例如 `add_to_cart` 可能显示
`baseline=0 → after_action=1 → after_undo=0`，表示购物车 0 件、加购后 1 件、撤销后回到 0 件。
如果某些 reddit 探针顶部摘要显示 `baseline="—"`，通常是 UI 汇总字段只识别通用键
`baseline/after_action/after_undo`，而该 probe 的 evidence 用的是更具体的键，如
`baseline_subscribed/after_action_subscribed/after_undo_subscribed`；下面的详细 evidence 表才是
真实证据，不代表探针没有记录。

**Q19：「数据集浏览器」应该怎么看？**
它是最终样本审阅页，读取已经 assemble/split/distill 出来的 SFT/DPO。左侧是一条条 sample，
右侧展示完整 lineage：来源状态、相关 key states、约束目标、grounded 标签、DPO 反事实、
teacher prose 和人工覆核。比如同一个 `add_to_cart` 状态，在 `constraint` 目标下可能是
`REVERSIBLE + AVOID + go_back()`，在 `request` 目标下可能是 `REVERSIBLE + EXECUTE + click(...)`。
这个页面不采集轨迹，也不跑 probe，主要用于最终审阅、驳回坏样本、导出 release。

**Q20：探针是人工写的，怎么能证明没有别的动作能恢复状态？这是 bug 吗？**
不能证明。当前标签应理解为**在指定站点、指定 actor 权限、指定 undo controller、指定
grounding signal 下的实测可恢复性**，不是数学意义上的"全局不可逆性证明"。例如 customer
侧 `place_order` 没找到取消按钮，只能说明当前 customer UI/controller 下没有恢复路径；
admin 后台、客服介入、隐藏 API、时间窗口等都不在这个结论范围内。因此更严谨的读法是：
`(action_type, site, role, undo_controller, signal) -> label`。如果要增强严谨性，应增加多个
undo controller、bounded undo search、权限/时间范围标注，并对 `IRREVERSIBLE` 采用更保守的
`UNKNOWN` 策略，直到覆盖证据足够强。

---

## 10. 安全须知（必读）

1. **API key 三不原则**：不进命令行参数、不进磁盘配置、不进日志。
   在界面填的 key 只活在服务进程内存里，服务一停就没了（需要重填或用环境变量）。
2. **破坏性操作双闸门**：13 个探针里 7 个是破坏性的（真实下单、删地址、admin 操作），
   工作台**故意不提供**它们的执行按钮。必须在命令行同时给出 `--commit` 参数
   **和** `REVACT_ALLOW_DESTRUCTIVE=1` 环境变量（缺一个都强制 dry-run），
   且按项目规定需逐批人工批准后才可执行。别绕过这个设计。
3. **服务只绑本机**（127.0.0.1）。它没有登录认证，不要用 `--host 0.0.0.0`
   暴露到公网；跨机器访问一律走 SSH 端口转发。
4. **数据不可变边界**：流水线产物只读、人工操作全走 overlay、导出带完整审计
   （excluded.jsonl + MANIFEST 溯源）。这不是繁文缛节——本项目的可信度就建立在
   "可逆性标签可审计、不可被悄悄改写"上。

---

*配套阅读：工程边界与扩展点 → [`workbench.md`](workbench.md)；
流水线各阶段原理 → [`../../RevAct重定位方案/05-S1-S8完整流程文档.md`](../../RevAct重定位方案/05-S1-S8完整流程文档.md)。*
