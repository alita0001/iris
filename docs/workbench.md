# IRIS 数据集构建工作台（workbench）

> 🐣 完全没接触过本项目？先读零基础教程 [`tutorial.md`](tutorial.md)（含三个手把手演练）；
> 本文是面向工程师的架构与边界说明。

浏览器里的可视化数据集构建台：覆盖 配置 → 成功轨迹采集 → 关键状态 → 约束注入 →
候选/反事实 → grounded undo 标注 → teacher 蒸馏 → 质量校验 → 导出/可视化 全链路。
它是现有 CLI pipeline 的**驾驶舱**，不是替代品：所有生成逻辑仍在
`revact/{data,grounding,train}`，工作台只负责编排（子进程 job）、浏览（只读 API）
与人工覆核（overlay 标注）。

## 启动

```bash
cd /workspace/iris
python -m revact.cli serve            # 或 python -m revact.server
# -> http://127.0.0.1:7788   （--host/--port 可改；默认只绑本机）
```

零第三方依赖（stdlib `http.server`），离线数据浏览无需任何环境；live 采集/探针
仍需先 `source scripts/export_webarena_env.sh`
再启动服务（或在「全局配置」里填 WA_SHOPPING）。

## 架构与工程边界

```text
revact/server/            stdlib HTTP API bridge
  datasets.py             只读加载 data/ 产物 + lineage join（自然键：state name /
                          sample_id / pair_id / probe_id / action_type）
  adapters.py             10 阶段注册表：每个按钮映射到真实 CLI 子命令 / 纯函数 /
                          显式 placeholder；live/key 前置条件先检查、不静默 mock
  jobs.py                 子进程 job：日志落 outputs/workbench/jobs/，key 只进子进程
                          env、日志按值打码
  annotations.py          人工覆核 overlay：data/annotations/<kind>.jsonl，append-only，
                          绝不改写 pipeline 原始产物
  quality.py              质量统计（复用 distill.qc_check 的矛盾规则）
  export.py               应用 overlay 的导出：outputs/workbench/exports/<stamp>__<name>/
  app.py / __main__.py    路由 + 配置持久化（剥离 key）
revact/ui/                原生 HTML/CSS/JS 前端（无框架、无外部资源）
configs/workbench.example.json   配置模板；本地保存为 workbench.local.json（gitignore）
```

**数据不变式**：`data/` 下的 pipeline 产物对工作台只读；人工编辑一律进
`data/annotations/`；导出时才合并，且与 grounded 标签冲突的人工覆核**排除样本并
记录审计**，永不改写行为标签（06 号文档铁律）。旧的 `revact viz` /
`outputs/dataset_viz.html` 静态报告完全保留，工作台「质量统计」页可一键重建。

## 密钥安全

- api_key 在 UI 输入后只存**服务进程内存**（`RUNTIME.secrets`），随进程消失；
- 「保存到本地」写 `configs/workbench.local.json` 时剥离 key 值，只保留 env 变量名；
- 运行 job 时 key 经环境变量注入子进程（不在命令行、不入 job 索引）；
- job 日志展示前按本会话见过的 secret 值打码；
- 也可以完全不在 UI 填 key，沿用 shell `export DEEPSEEK_API_KEY=...`。

## 各页与真实度（哪些真实接入 / 哪些是 adapter/placeholder）

| 页 | 后端 | 真实度 |
|---|---|---|
| Pipeline 总览 | adapters + jobs | ✅ 状态由产物文件+最近 job 推导；单步/从阶段继续/全流程链（mock 链离线可跑，live 链需环境+key） |
| 全局配置 | /api/config | ✅ policy/teacher/judge 三角色 provider/base_url/model/key/温度等，经 env（REVACT_LLM_*、REVACT_DISTILL_*、REVACT_WA_JUDGE_BASE_URL）透传给现有 CLI |
| **Prompt 管理** | /api/prompts（`revact/prompts.py` registry） | ✅ 全部 LLM prompt 可编辑：agent 系统提示词（训练+部署同一条）、采集策略模型提示词、teacher 蒸馏模板、显式/隐式约束与请求目标模板池。覆盖存 `configs/prompts.local.json`（gitignore），**调用时读取**（子进程 job 即时生效）；占位符校验；对比默认/恢复默认；样本 meta.prompts_fp + 导出 prompts.json 快照做溯源。改 agent_system/模板池后需重跑 assemble(+multiturn+split) |
| 成功轨迹 | data/raw/trajectories | ✅ step trace + axtree + 截图；成功/失败/异常筛选；「选入下一阶段」为 trajectory overlay |
| 关键状态 | shopping_key_states.jsonl | ✅ 浏览+确认/驳回/类型标签/置信度（overlay）。"为什么关键"=S2 关键词规则，如实展示 |
| 约束注入 | assemble.build_goal | ✅ 预览即真实注入逻辑（确定性模板抽取）；人工改写为 overlay，物化仍需 assemble；六类约束类型是标注维度 |
| 候选/反事实 | reach 产物 + assemble DPO builders | ✅ expert/safe/四类 rejected 实时计算；⚠️ 生成式候选提案（S4）为 **placeholder**，人工添加走 overlay |
| Undo 标注 | grounded/reversibility.jsonl | ✅ S→A→S′ 证据、undo 轨迹、residual_diff、截图、训练字段派生；人工覆核 overlay；mock/live 探针可跑；⛔ 破坏性探针不开放（CLI 双闸门+批准） |
| Teacher 蒸馏 | train/distill.py | ✅ 运行（需 key）；模板 vs teacher prose 对比 + pinned 结论逐字校验 |
| 质量统计 | server/quality.py | ✅ 量/率/分布/反事实覆盖/teacher一致率/低质量清单（复用 distill QC 规则） |
| 训练数据物化 | qc 阶段 actions | ✅ assemble（单步，P0 起 user 含 `<history>`）+ assemble-multiturn（P1 轨迹→多轮 chat 样本）+ train/train-dpo/train-grpo 的 `--dry-run` 校验按钮；真实训练在 CLI（qwen-vllm 环境 + GPU）：`train` / `train-dpo [--adapter]` / `train-grpo [--gspo] [--adapter]`（--adapter = SFT LoRA 热启） |
| rollout FSR 评测 | eval/rollout.py（CLI `eval-rollout`） | ✅ 守护式真实 rollout：模型经 vLLM 部署进 agent 循环，held-out 状态 × constraint/request 变体实测 FSR/over-block/confirm；**破坏性风险动作只记录不执行**（guarded，双闸门不破），可逆动作真执行并事后清理；产物 outputs/rollout_eval/<tag>.jsonl + summary |
| 数据集浏览器 | lineage join + export.py | ✅ 全链 lineage、accept/reject/needs-review、导出 train/val/test + dataset card + excluded 审计；顶部「Dataset Card」面板（HF 风格：单步样本粒度说明、messages 三段解剖、SFT/meta/DPO 字段 schema、长度统计），样本详情底部「完整样本形态」展示未截断原始 JSONL（`/api/dataset_card`、`/api/sample_raw`） |

## 验证

```bash
./ci.sh                                   # ruff + pytest（含 workbench 离线测试）
python -m revact.cli serve &              # 起服务
curl -s localhost:7788/api/health | python -m json.tool
curl -s localhost:7788/api/pipeline | python -m json.tool | head
# 浏览器打开 http://127.0.0.1:7788 ，Pipeline 页选 mock 链「全流程运行」
```

## 多站点（sites）

工作台按 WebArena **站点**扩展，已接入 `shopping` / `shopping_admin`（Magento）与
`reddit`（Postmill）。站点注册表在 `revact/config.py` 的 `SITES`（每站点声明 base-url
环境变量、会话任务、路径表）。grounded 页的探针注册表**按站点分组**展示，并提供
「live 非破坏（reddit）」按钮；配置区新增 `WA_REDDIT`。live 门控是站点感知的
（`live:<site>`）：只有对应 `WA_*` 就绪该站点的 live 探针才可跑。

**新增一个站点镜像**（如 gitlab / wikipedia）：① 部署镜像并 export `WA_<SITE>`；
② `config.SITES` 加一行 `SiteSpec` + 对应 `*_PATHS`；③ 新增
`revact/grounding/probes/<site>.py`（信号 + undo + 探针，参照 `reddit.py`）并在
`probes/__init__.py` 导入；④（可选）在 `envs/mock_env.py` 加 `Mock<Site>Env` 做离线单测。
CLI（`--site`）、工作台站点分面、`splits.py` 的 `cross_site` 全部自动适配。
扩展 reddit 时的实验发现见 [`findings-multisite.md`](findings-multisite.md)。

## 扩展点（接真实执行逻辑的下一步）

1. **S4 生成式候选**：新增 `revact/data/candidates.py`（页面元素枚举 + 诱饵类 +
   合法性过滤，见 04 号文档 §5），在 `adapters.py` 的 `candidates.propose` 把
   `kind="placeholder"` 换成 `cli`，UI 无需改动。
2. **逐 (state,action) 探测**：probe 上量后 `probe_named` 已支持指定探针名；按
   30–80 状态点/类扩容时给 CLI 加 `--state` 参数即可透传。
3. **意见标签并行采集**（分歧矩阵）：作为新 stage 挂进 `STAGES`，产物落
   `data/grounded/opinions.jsonl`，quality.py 加分歧率统计。
4. **cross-action-class split**：`splits.py` 落地后，导出面板加 split 策略选项。
5. **标注协同**：`annotations.add(author=...)` 已带作者字段，多人标注只需在 UI
   加登录名输入。
