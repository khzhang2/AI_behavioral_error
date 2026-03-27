# Krauss_Krail_Axhausen 短距离逐题版 AI 复现实验

## 实验定位

这个实验目录是基于 `20260327_sd_1445persona_6cards_session_qwen35_4b_v3` 的更正版。

目标不是改动 v3 的样本规模或 mixed logit 规格，而是修正最关键的 survey 呈现偏差：

- 保留 `1445` 个 persona
- 保留 `6` 张 short-distance 重建题卡
- 保留 `qwen3.5:4b`、`think=false`、`temperature=0.1`
- 保留 `5,000 Sobol draws` 的 paper-aligned mixed logit 子模型
- 改成同一 persona 上下文里逐题展示 `6` 个 choice tasks，而不是一次性要求模型输出 `6` 个答案

## 对论文的核对结果

根据原文 *What drives the utility of shared transport services for urban travellers? A stated preference survey in German cities*：

- SP 设计是每位受访者 `6` 个 choice situations、每题 `4` 个 alternatives
- short-distance 问卷的场景是 `intra-urban`、`leisure`、`no luggage`
- 受访者被告知不用担心自己是否是共享服务会员
- 设计是 `8 blocks × 6 choice situations`
- 为了降低顺序偏差，choice situations 是从一个 block 中随机抽取展示

这轮实现据此做了两项明确修正：

- 把 `v3` 的单条 session prompt 改成逐题多轮问答
- 对每个 respondent 随机化 `6` 张短距离题卡的展示顺序

## 保留的透明近似

这轮仍然不是对论文全文模型的完全等价复现，而是一个透明的 short-distance reconstruction：

- 论文发表版主模型是 `SD + MD` 的 pooled mixed logit
- 当前目录仍然只跑 short-distance 的 `4` 个 alternatives
- 人口学抽样继续沿用 Table 1 的总体 sample marginals
- `1445` 个 respondent 继续沿用 v3 的 human-scale 设定，因此它是“short-distance human-scale replication”，不是“论文原始 SD 子样本大小”的精确复刻

## 题卡与变量说明

- `data/krauss_sd_6cards_reconstruction.csv` 仍然是重建式 pilot block，不是原始 Ngene 设计文件
- 题卡中的 `range` 继续按 Table 2 和 Fig. 3 的展示方式使用 `km`
- `human_table4_sd_subset.csv` 仍然保留 Table 4 的转写系数，用于 AI-human 对比

## 运行方式

### 采集

建议直接使用系统 `python`：

```powershell
python experiments\Krauss_Krail_Axhausen\20260327_sd_1445persona_6cards_multiturn_qwen35_4b_v4\scripts\run_ai_collection.py
```

小样本验证：

```powershell
python experiments\Krauss_Krail_Axhausen\20260327_sd_1445persona_6cards_multiturn_qwen35_4b_v4\scripts\run_ai_collection.py --n-respondents 3
```

断点续跑：

```powershell
python experiments\Krauss_Krail_Axhausen\20260327_sd_1445persona_6cards_multiturn_qwen35_4b_v4\scripts\run_ai_collection.py --resume
```

### 估计与汇总

```powershell
python experiments\Krauss_Krail_Axhausen\20260327_sd_1445persona_6cards_multiturn_qwen35_4b_v4\scripts\estimate_mixed_choice_model.py
python experiments\Krauss_Krail_Axhausen\20260327_sd_1445persona_6cards_multiturn_qwen35_4b_v4\scripts\build_comparison.py
python experiments\Krauss_Krail_Axhausen\20260327_sd_1445persona_6cards_multiturn_qwen35_4b_v4\scripts\summarize_experiment.py
```

### 后处理监听器

`post_collection_runner.py` 已改成同时兼容：

- Windows 的 `.venv\Scripts\python.exe`
- Unix 风格的 `.venv/bin/python`
- 当前解释器 `sys.executable`

当前仓库还额外放置了一个可用的本地 GPU `torch`：

- 优先加载 `.python_packages\cu118`
- 如果后续你想切换到别的 wheel，可以再改成 `.python_packages\cu126` 或新的目录

## 输出差异

和 `v3` 相比，当前目录的原始日志定义发生了变化：

- `raw_interactions.jsonl` 现在是一题一行，而不是一位 respondent 一行
- `parsed_choices.csv` 额外记录 `presented_task_position`
- `questionnaire_manifest.json` 会标记 `survey_mode = multi_turn_panel_prompt`

因此，这一轮的输出不能和 `v3` 的单 session 原始日志直接混合。
