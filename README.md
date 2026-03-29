# AI Behavioral Error（AI行为误差）

这是一个 Python 代码库，用于开展一项重建式试验：研究在 travel behavior stated-preference 场景中，LLM 作为受访者时的行为与误差结构。

## 研究范围

当前仓库服务于本项目的第一篇文章：

- 只聚焦短距离 stated-preference 任务。
- 将 Krauss、Krail 和 Axhausen（2022）的问卷视为一个重建式 pilot，而不是对原始 Ngene 设计的精确复刻。
- 保留完整的 AI respondent 日志，便于后续识别 prompt 效应、persona 效应和算法随机性。

当前的基线 pipeline 做四件事：

1. 加载一个包含 6 张短距离题卡的 pilot block。
2. 运行 synthetic respondents 完成问卷。
3. 保存原始交互日志和 respondent-level 的选择结果。
4. 构造 long-format choice data，并估计一个基础 conditional logit 模型。

## 仓库结构

- `configs/`：smoke test 和 pilot run 的实验配置。
- `data/designs/`：短距离属性水平和 6 张 pilot 题卡的机器可读转写。
- `docs/`：研究 framing 和方法说明。
- `experiments/`：按论文和实验轮次组织的完整实验产物目录。
- `src/ai_behavioral_error/`：pipeline 代码。

## 实验流程

这一节描述当前仓库推荐的实验组织方式，尤其适用于 `Krauss_Krail_Axhausen` 这一条主线。

### 1. 实验总流程

每一轮正式实验建议按下面顺序推进：

1. 冻结题卡设计。
2. 冻结 persona 采样规则。
3. 冻结 prompt 模板和输出格式。
4. 先做小样本 smoke test，检查解析率和选择分布。
5. 再做正式采集。
6. 采集完成后整理 `parsed_choices.csv` 和 canonical `pooled_choices_long.csv`（旧实验若需要，可额外保留 `ai_choices_wide.csv` 作为 legacy artifact）。
7. 估计 mixed logit 或其他离散选择模型；当前 Krauss 主线推荐使用 `Biogeme`。
8. 与人类论文结果做参数方向和量级比较。
9. 写实验摘要，记录边界、异常和后续改动建议。

### 2. 文件夹组织

推荐结构如下：

- `experiments/<paper_name>/`
  存放某一篇论文的全部复现实验。
- `experiments/<paper_name>/<experiment_name>/`
  存放某一轮具体实验。
- `experiments/<paper_name>/<experiment_name>/data/`
  存放题卡、prompt 模板、persona 采样规则、人类参数转写表、实验配置。
- `experiments/<paper_name>/<experiment_name>/scripts/`
  存放采集、估计、对比、汇总脚本。
- `experiments/<paper_name>/<experiment_name>/outputs/`
  存放 persona 样本、原始模型回复、结构化 choice、估计结果、图和 summary。
- `experiments/<paper_name>/<experiment_name>/docs/`
  存放该轮实验的中文说明文档。

当前 `Krauss_Krail_Axhausen` 主线已经有三类典型实验：

- `v1`：小样本、5 张卡、Biogeme MNL 基线。
- `v2`：小样本、5 张卡、paper-aligned mixed logit 子模型。
- `v3`：大样本、1445 persona、6 张卡、单 session prompt。
- `v7`：公开材料最高保真 pooled `SD + MD`、`7` 个备选项、`Biogeme` panel mixed logit 主线。

现在仓库里也有第二条论文线：

- `Swissmetro/20260329_public_reverse_engineered_panel_mnl_biogeme_v1`：Swissmetro 公开材料逆向工程、pylogit 对齐 `MNL` benchmark、三轮 AI respondent 复现实验。

### 3. 每轮实验至少需要准备的文件

- `data/experiment_config.json`
  定义样本量、模型、temperature、draw 数、任务数等。
- `data/persona_sampling_rules.json`
  定义 demographic 和 mobility resource 的采样规则。
- `data/survey_instrument_en.md`
  记录公开材料问卷 framing 和答题约束。
- `data/*cards*.csv`
  记录题卡本体或 pooled choice-set 设计文件。
- `data/human_table4_*.csv`
  记录从原文转写的人类基线参数。
- `scripts/run_ai_collection.py`
  负责 persona 生成、问卷投喂、原始日志和结构化输出。
- `scripts/estimate_*.py`
  负责模型估计。
- `scripts/build_comparison.py`
  负责 AI-human 参数对比。
- `scripts/summarize_experiment.py`
  负责生成中文摘要。

### 4. Prompt 设计框架

当前仓库推荐把 prompt 设计拆成四层，并尽量保持可审计。

第一层：system prompt

- 固定 respondent 身份。
- 明确“只根据 profile 和题卡属性作答”。
- 禁止引入题外假设。
- 明确要求结构化输出。

第二层：persona profile

- 只放可审计的人口学和 mobility resources。
- 当前建议字段包括：
  `gender`、`age_band`、`age_years`、`income_band`、`city_type`、`household_cars`、`accessible_bikes`、`pt_pass`、`maas_subscription`。
- 除非另有实验目的，不要加入“环保倾向”“风险偏好”“热爱运动”这类主观行为描述。

第三层：task/session prompt

- 明确场景：
  `leisure`、`intra-urban`、`no luggage`、`ignore membership constraints`。
- 明确 alternatives 顺序。
- 明确属性名和单位。
- 如果是多题问卷，必须明确 task id。

第四层：output format

- 单题模式：`{"choice":"B"}`
- 多题 session 模式：`{"choices":{"SD01":"A","SD02":"B",...}}`
- 输出格式必须尽量短、尽量硬约束，方便自动解析。

### 5. 当前建议优先做的实验

在 `Krauss_Krail_Axhausen` 这条线上，当前最重要的实验顺序是：

1. `reconstruction pilot`
   先验证 pipeline 跑通，不追求人类参数贴合。
2. `small-sample mixed logit`
   先看 paper-aligned 规格能否在小样本上正常工作。
3. `human-scale replication`
   把样本量扩到与原文同量级，判断差异究竟来自样本量还是行为机制。
4. `multi-turn vs single-session comparison`
   比较“逐题多轮问答”和“一次 session 完成整份问卷”的差异。
5. `error decomposition`
   在基线稳定后，再去拆 prompt variance、persona variance、sampling variance。

### 6. 每轮实验必须保存的原始产物

- `persona_samples.csv`
- `raw_interactions.jsonl`
- `respondent_transcripts.json`
- `parsed_choices.csv`
- `pooled_choices_long.csv` 或其他 canonical estimation input
- `biogeme_mixed_estimates.csv` 或其他模型结果文件
- `ai_vs_human_comparison.csv`
- `experiment_summary.md`

### 7. 当前主线实验的实际差异

为了避免混淆，需要特别说明：

- `v1` 和 `v2` 更接近“一个 persona 多次问答”的对话式实验。
- `v3` 更接近“一个 persona 一次性完成 6 张卡”的问卷式实验。
- 两者的 `raw_interactions.jsonl` 行定义不同，后续分析时不能直接混为一类。

## 统一环境

当前仓库默认只使用根目录虚拟环境 `./.venv`。

- 推荐 Python 版本：`3.11`
- 当前仓库的统一解释器入口：
  - Windows：`.\.venv\Scripts\python.exe`
  - Unix：`./.venv/bin/python`
- 当前主线实验不依赖 `torch`，Krauss `v7` 的参数估计统一使用 `Biogeme`

如果需要重建这个环境，建议按下面顺序操作：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

如果要运行 Krauss_Krail_Axhausen 的实验估计脚本，建议补齐实验依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[experiments]"
```

## 快速开始

运行 mock smoke test：

```bash
./.venv/bin/python -m ai_behavioral_error run-pipeline --config configs/sd_smoke_mock.json
```

运行更大的 mock pilot：

```bash
./.venv/bin/python -m ai_behavioral_error run-pipeline --config configs/sd_pilot_mock.json
```

运行 `qwen3.5:4b` 的 Ollama live smoke test：

```bash
./.venv/bin/python -m ai_behavioral_error run-pipeline --config configs/sd_smoke_ollama_qwen35_4b.json
```

运行关闭 thinking 的同一组 Ollama smoke test：

```bash
./.venv/bin/python -m ai_behavioral_error run-pipeline --config configs/sd_smoke_ollama_qwen35_4b_no_think.json
```

如果你之后想调用 OpenAI-compatible API，可以从下面这个模板开始：

- `configs/sd_pilot_openai_compat_template.json`

这个模板默认目标端点兼容 `chat/completions`，并且 API key 从 `OPENAI_API_KEY` 读取。

## 输出文件

每次实验都会把结果写到 `results/` 下各自的输出目录中：

- `raw_interactions.jsonl`：完整 prompt、模型原始响应和元数据。
- `response_composition_summary.json`：thinking 与 final response 的拆分摘要。
- `response_composition_rows.csv`：逐条交互的 thinking/response 指标。
- `choices.csv`：每个 respondent-task 一行。
- `long_format.csv`：每个 respondent-task-alternative 一行。
- `diagnostics.json`：解析率和简单稳定性指标。
- `mnl_coefficients.csv` 与 `mnl_summary.txt`：基础 conditional logit 结果。
- `choice_shares.png`、`position_bias.png`、`mnl_coefficients.png`：基础图形输出。

## 当前边界

这个代码库刻意保持学术研究可用，而不是工业级实现：

- 错误处理尽量精简。
- 可视化只用简单的 `matplotlib.pyplot`。
- 提供一个 mock backend 供本地 smoke test 使用。
- 提供一个轻量的 OpenAI-compatible backend 模板，便于后续 live run。
