# Krauss_Krail_Axhausen 短距离 AI 复现实验

## 实验范围

这个实验是基于 Krauss、Krail 和 Axhausen（2022）的短距离子实验所做的一次重建式 AI replication，使用本地 Ollama 上的 `qwen3.5:4b`，并关闭 thinking 模式。

这个实验的目标不是精确复原原始 Ngene blocks。原论文在主文中报告了属性空间，并给出了一张短距离图形化示例卡，但没有公开完整题卡矩阵。因此本实验采用：

- 论文中的短距离 mode set。
- Table 2 中的短距离属性水平。
- 从仓库现有 pilot 设计中抽取并整理的 5 张重建式题卡。
- 基于 Table 1 分布或基于均值做 moment-matching 近似采样的 30 个 synthetic personas。
- 每个 respondent 连续完成 5 个 choice tasks，并在对话上下文中保留 persona 信息。

## 为什么先做短距离

用户要求先做一个 `30 respondent rounds × 5 cards` 的 AI replication。短距离设计最适合作为第一轮落地，因为：

- 备选项直观：e-scooter sharing、bike sharing、walking、private car。
- 原论文给出了短距离的图形化示例卡。
- 当前仓库里已经有一份可机器读取的短距离重建设计。

## 人类基线对比

人类对比系数来自论文 Table 4 中与短距离备选项和可重叠属性对应的部分。AI 模型使用 Biogeme 估计的是一个 MNL，因此这个对比是方向性、系数层面的比较，而不是对论文 mixed logit with scale heterogeneity 的一比一复刻。

## 环境

### 推理环境

- 本地 Ollama 服务地址：`127.0.0.1:11434`
- Ollama 版本：`0.18.3`
- 模型：`qwen3.5:4b`
- 解码模式：`think=false`
- temperature：`0.1`
- top_p：`0.95`
- top_k：`20`
- num_predict：`128`

### 估计环境

- 当前仓库统一虚拟环境：根目录 `./.venv`
- Python：`3.9.7`
- Biogeme：`3.2.13`
- NumPy：`1.26.4`
- pandas：`2.3.3`
- SciPy：`1.13.1`
- 当前 `./.venv` 已设置 `include-system-site-packages = true`

### Venv 说明

- 原先实验目录中的 `.venv39` 已经移除。
- 当前请统一使用仓库根目录的 `./.venv`。
- 建议的默认入口：`./.venv/bin/python`

## 运行结果

### 采集摘要

- synthetic respondents：`30`
- 每个 respondent 的题卡数：`5`
- AI choices 总数：`150`
- valid choice rate：`1.0`
- 平均每题 Ollama 调用时长：约 `4.05` 秒

### 选择分布

- Private Car：`108`
- Walking：`30`
- Bike Sharing：`12`
- E-Scooter Sharing：`0`

### Persona 集中度

- `21` 个 personas 在 5 题里全部选择 `Private Car`
- `6` 个 personas 在 5 题里全部选择 `Walking`
- `2` 个 personas 在 5 题里全部选择 `Bike Sharing`
- `1` 个 persona 在 `Bike Sharing` 和 `Private Car` 之间切换

### AI 与人类参数对比

- 可比较参数数：`16`
- 符号一致数：`7`
- sign match rate：`0.4375`

这轮实验已经可以作为一份完整的端到端 AI respondent replication artifact，但它还不能很好地恢复人类论文中的行为结构。最主要的问题是 `Private Car` 高度集中、persona 内切换很弱，而且 `E-Scooter` 为零选择，这使得若干 paper-style 参数的识别非常弱。

## 文件说明

- `data/krauss_sd_5cards_reconstruction.csv`：本次实验实际使用的 5 张短距离题卡。
- `data/persona_sampling_rules.json`：基于论文的 persona 采样规则。
- `data/human_table4_sd_subset.csv`：从 Table 4 转写的人类基线系数。
- `data/prompt_template.md`：respondent prompt 模板。
- `scripts/run_ai_collection.py`：本地 Ollama 采集脚本。
- `scripts/estimate_biogeme_model.py`：Biogeme 估计脚本。
- `scripts/build_comparison.py`：AI 与 human 对比脚本。

## 预期原始输出

- `outputs/questionnaire_manifest.json`
- `outputs/persona_samples.csv`
- `outputs/raw_interactions.jsonl`
- `outputs/respondent_transcripts.json`
- `outputs/parsed_choices.csv`
- `outputs/ai_choices_wide.csv`
- `outputs/biogeme_ai_estimates.csv`
- `outputs/biogeme_model_summary.json`
- `outputs/ai_vs_human_comparison.csv`
- `outputs/ai_vs_human_coefficients.png`
- `outputs/experiment_summary.md`
