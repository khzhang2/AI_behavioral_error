# Swissmetro Simple AI Survey Replication

这个目录现在只保留一套最小研究流程：

- 用现有 `data/` 里的 Swissmetro 重建数据
- 通过 `ollama` 调 `qwen3.5:9b`
- 像手机问卷一样逐题问 `9` 个 choice task
- 同一个 respondent 在同一段不断增长的对话里完成整份问卷
- 用 `Biogeme` 估计一个和 pylogit notebook 一致的 `4` 参数 `MNL`

## 主要文件

- `data/experiment_config.json`
- `scripts/questionnaire_template.py`
- `scripts/run_ai_collection.py`
- `scripts/estimate_biogeme_mnl.py`
- `scripts/build_comparison.py`
- `scripts/summarize_experiment.py`

## 设计原则

- prompt 不再先列 respondent dossier，而是直接告诉模型“你是一个什么样的人”
- grounding 改成 compact JSON，专门避免被截断
- 每一题都会把前面的问答一起带上，所以上下文会随着题目推进变长
- prompt 会直接解释问卷特有概念，尤其是 `GA`、`headway`、`seat configuration`
- 只保留研究会用到的输出

## 建议运行顺序

1. 采集 AI 问卷

```powershell
python experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\run_ai_collection.py
```

2. 估计人类 benchmark

```powershell
python experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\estimate_biogeme_mnl.py --dataset human
```

3. 估计 AI MNL

```powershell
python experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\estimate_biogeme_mnl.py --dataset ai
```

4. 生成人类与 AI 对比

```powershell
python experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\build_comparison.py
```

5. 生成实验 summary

```powershell
python experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\summarize_experiment.py
```

## 关键输出

- `outputs/persona_samples.csv`
- `outputs/reconstructed_panels_wide.csv`
- `outputs/reconstructed_panels_long.csv`
- `outputs/parsed_choices.csv`
- `outputs/raw_interactions.jsonl`
- `outputs/respondent_transcripts.json`
- `outputs/run_respondents.json`
- `outputs/human_benchmark_biogeme_mnl_estimates.csv`
- `outputs/ai_biogeme_mnl_estimates.csv`
- `outputs/ai_vs_human_comparison.csv`
- `outputs/ai_vs_human_coefficients.png`
- `outputs/experiment_summary.md`
