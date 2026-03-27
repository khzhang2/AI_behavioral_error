# Krauss_Krail_Axhausen 第二轮短距离 AI 复现实验

## 实验定位

这个实验是 `20260326_sd_persona_nonthink_qwen35_4b_v2`，目标是在保留 `5` 张短距离重建题卡的前提下，尽量向 Krauss、Krail 和 Axhausen（2022）原文的 `mixed logit` 规格对齐。

这一轮不再使用 v1 的简化 MNL，而是改成：

- 仍然只使用 short-distance 的 `4` 个备选项：`e-scooter sharing`、`bike sharing`、`walking`、`private car`
- 仍然保留 `5` 张重建式题卡
- 每个 synthetic respondent 在同一上下文内先完成 `2` 次 warm-up，再完成 `5` 次正式 choice
- 估计阶段采用 Python 自写的 panel mixed logit 子模型
- random draws 按原文方法改为 `5,000 Sobol draws`

需要说明的边界是：原论文主模型是 short-distance 与 medium-distance 合并后的 pooled mixed logit。当前这轮只保留了 `5` 张 short-distance 题卡，因此这里复刻的是一个 `paper-aligned short-distance subset`，而不是发表版全文模型的完整可识别复制。

## prompt 设计

这一轮没有额外加入主观“行为习惯”描述，也没有做更强的人设操控。prompt 只保留与原文规格紧密相关、可审计的 respondent 信息：

- 性别
- 年龄
- 收入组
- 城市类型
- household car accessibility
- accessible bikes
- PT pass
- MaaS subscription

这样做是为了尽量减少 prompt manipulation，把变化更多留给题卡属性本身。

### warm-up 设置

正式 choice 前增加了 `2` 次 warm-up：

- `warmup_1`：确认 respondent profile
- `warmup_2`：确认 mobility resources

warm-up 只要求模型返回一行 JSON，用于激活和核对 persona，不进入 choice model 估计。

## 估计规格

### 与原文对齐的部分

- 模型类型：panel mixed logit 子模型
- draw 方法：`Sobol`
- draw 数量：`5,000`
- random dimensions：`cost`、`ASC_ES`、`ASC_BS`、`ASC_WALK`
- `cost` 采用 log-normal 处理
- 变量命名和进入方式尽量与原文 Table 4 对齐

### 当前 short-distance 子模型的规范化

由于这轮只保留 short-distance 的 `4` 个备选项，当前估计采用 `private car` 作为基准备选项，估计的是：

- `ASC_ES_REL_CAR`
- `ASC_BS_REL_CAR`
- `ASC_WALK_REL_CAR`

以及与 `car` 基准相对的 age / bike accessibility / car accessibility / PT pass / MaaS 效应。

## 环境

### 推理环境

- 本地 Ollama 服务地址：`127.0.0.1:11434`
- 模型：`qwen3.5:4b`
- 解码模式：`think=false`
- temperature：`0.1`
- top_p：`0.95`
- top_k：`20`
- num_predict：`128`

### 估计环境

- 当前仓库统一虚拟环境：根目录 `./.venv`
- Python：`3.9.7`
- 主要依赖：`torch`、`biogeme`、`numpy`、`pandas`、`scipy`
- Sobol draws 实现：`scipy.stats.qmc.Sobol`
- 当前 `./.venv` 已设置 `include-system-site-packages = true`

这一轮没有使用 Biogeme 来估计 mixed logit，原因是当前环境下的 Biogeme 版本没有直接提供与原文一致的 Sobol draw 接口，所以改为 Python 自写模拟极大似然来复刻原文的 random draw 方法。

如果运行包含 `torch` 的脚本，建议用 `OMP_NUM_THREADS=1 ./.venv/bin/python ...` 作为默认入口。

## 实际运行结果

### 采集摘要

- synthetic respondents：`30`
- warm-up 总数：`60`
- 正式 choices：`150`
- `valid_choice_rate = 1.0`
- 全部 `210` 次调用的 `done_reason` 都是 `stop`

### 调用时长

- 平均 warm-up 时长：`7.18` 秒
- 平均 choice 时长：`10.03` 秒
- 最长 warm-up 时长：`12.68` 秒
- 最长 choice 时长：`19.48` 秒

### choice 分布

- `E-Scooter Sharing`：`85`
- `Private Car`：`55`
- `Walking`：`10`
- `Bike Sharing`：`0`

和 v1 相比，这一轮最明显的变化是：

- `E-Scooter` 不再是零选择
- `Private Car` 不再压倒性主导
- `28` 个 persona 在 `E-Scooter` 与 `Private Car` 之间切换

这说明题卡属性已经比 v1 更明显地进入 choice 过程。

## mixed logit 结果

估计结果文件：

- `outputs/mixed_choice_estimates.csv`
- `outputs/mixed_choice_model_summary.json`

本轮 mixed logit 摘要：

- `n_respondents = 30`
- `n_observations = 150`
- `n_parameters = 36`
- `n_draws = 5000`
- `final_loglikelihood = -23.419`
- `rho_square = 0.887`

但是，这一轮完整 paper-aligned 规格在 `5` 张题卡、`150` 个 choice 的数据规模上仍然明显过度参数化。虽然 log-likelihood 很高，但很多参数绝对值非常大，说明当前不是“高质量复现”，而是“在小样本上把完整规格硬拟合了出来”。

典型表现包括：

- `ASC_ES_REL_CAR = 10.565`
- `CARACC_ES_REL_CAR = -151.770`
- `PTPASS_ES_REL_CAR = 133.686`
- `B_TIME_ES = -87.924`
- `B_TIME_CAR = 52.079`
- `B_ACCESS_SHARED = 42.806`

因此，这一轮 mixed logit 的主要意义是：

- 证明 paper-aligned 的 Python mixed logit pipeline 已经跑通
- 证明 `5,000 Sobol draws` 的原文式模拟估计可以在当前框架内复现
- 说明当前 `5 cards + 150 choices` 对完整论文规格来说仍然识别不足

## AI 与人类参数对比

对比文件：

- `outputs/ai_vs_human_comparison.csv`
- `outputs/ai_vs_human_coefficients.png`
- `outputs/ai_vs_human_summary.json`

摘要：

- 可比较参数数：`36`
- 符号一致数：`22`
- `sign_match_rate = 0.6111`

这个 sign match rate 比 v1 更高，但不能简单解读成“已经复现了人类参数”。原因是当前 mixed logit 的系数幅度明显失真，很多参数虽然符号对了，但数值已经被小样本过拟合拉得很极端。

## 文件说明

- `data/krauss_sd_5cards_reconstruction.csv`：本轮使用的 `5` 张 short-distance 题卡
- `data/persona_sampling_rules.json`：persona 采样规则
- `data/human_table4_sd_subset.csv`：根据原文 Table 4 整理的 short-distance 子模型对比系数
- `data/prompt_template.md`：英文 prompt 模板
- `scripts/run_ai_collection.py`：本地 Ollama 采集脚本
- `scripts/estimate_mixed_choice_model.py`：Python mixed logit 估计脚本
- `scripts/build_comparison.py`：AI 与 human 对比脚本

## 当前结论

这轮 v2 实验已经实现了你要求的几个关键点：

- 保留 `5` 张题卡
- 加入 `2` 次 warm-up
- 继续使用 non-thinking
- mixed choice model 与原文变量体系基本对齐
- random draws 方法改为原文式 `5,000 Sobol draws`

但从结果上看，当前数据量还不足以稳定识别完整论文规格。下一步如果目标是“让参数更接近人类 Table 4”，更合理的方向不是继续往 prompt 里塞额外行为习惯，而是：

- 扩大题卡数
- 扩大 respondent 数
- 或者先临时缩减 mixed logit 规格，再逐步加回复杂项
