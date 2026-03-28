# Krauss Public-Materials Pooled Replication (`v7`)

## 定位

这个目录实现的是：

- `public_materials_high_fidelity`
- `pooled SD + MD`
- `7` 个备选项
- `Biogeme` panel mixed logit

它不是原始 Ngene 文件和原始 respondent microdata 的 exact recovery。这里的 choice-set 组合是依据公开论文的 attribute levels 重建出来的，因此：

- `public_attribute_levels_sd.csv` 和 `public_attribute_levels_md.csv` 是 `public_exact`
- `pooled_choice_sets_public_materials.csv` 的 block / task 组合是 `inferred_from_public`

## 目录内容

- `data/public_attribute_levels_sd.csv`
- `data/public_attribute_levels_md.csv`
- `data/pooled_choice_sets_public_materials.csv`
- `data/human_table4_pooled_full.csv`
- `data/survey_instrument_en.md`
- `data/public_replication_assumptions.json`
- `scripts/generate_public_design.py`
- `scripts/run_ai_collection.py`
- `scripts/estimate_biogeme_mixed_logit.py`
- `scripts/build_comparison.py`
- `scripts/summarize_experiment.py`

## 运行环境

当前主线建议直接使用仓库根目录虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[experiments]"
```

`v7` 不使用 `torch`。参数估计只通过 `Biogeme` 完成。

## 运行顺序

1. 生成公开材料 choice-set 设计文件

```powershell
python experiments\Krauss_Krail_Axhausen\20260327_public_materials_pooled_7alts_biogeme_v7\scripts\generate_public_design.py
```

2. 跑 AI survey 采集

```powershell
python experiments\Krauss_Krail_Axhausen\20260327_public_materials_pooled_7alts_biogeme_v7\scripts\run_ai_collection.py
```

3. 用 Biogeme 估计 pooled mixed logit

```powershell
.\.venv\Scripts\python.exe experiments\Krauss_Krail_Axhausen\20260327_public_materials_pooled_7alts_biogeme_v7\scripts\estimate_biogeme_mixed_logit.py
```

4. 生成人机对比与 summary

```powershell
python experiments\Krauss_Krail_Axhausen\20260327_public_materials_pooled_7alts_biogeme_v7\scripts\build_comparison.py
python experiments\Krauss_Krail_Axhausen\20260327_public_materials_pooled_7alts_biogeme_v7\scripts\summarize_experiment.py
```

## 关键输出

- `outputs/persona_samples.csv`
- `outputs/parsed_choices.csv`
- `outputs/pooled_choices_long.csv`
- `outputs/raw_interactions.jsonl`
- `outputs/respondent_transcripts.json`
- `outputs/biogeme_estimation_wide.csv`
- `outputs/biogeme_mixed_estimates.csv`
- `outputs/biogeme_mixed_model_summary.json`
- `outputs/ai_vs_human_comparison.csv`
- `outputs/ai_vs_human_summary.json`
- `outputs/experiment_summary.md`

## 公开材料假设

- respondent 只使用 public `Table 1` observable margins
- `MaaS subscription ~ Bernoulli(0.10)` 是显式外部假设
- respondent 被固定种子下 balanced shuffle 到 `SD` / `MD`
- 每位 respondent 被分配到一个重建 block，并在 block 内随机顺序完成 `6` 题

## 估计结构

`estimate_biogeme_mixed_logit.py` 实现了：

- `PanelLikelihoodTrajectory`
- `MonteCarlo`
- `5000` draws 默认配置
- `8` 个随机项：
  - `z_cost`
  - `z_es`
  - `z_bs`
  - `z_walk`
  - `z_car`
  - `z_cs`
  - `z_rp`
  - `z_pt`

如果当前 `Biogeme` 版本没有原生 Sobol draws，就通过 user-defined generator 在 `Biogeme` 内部生成 scrambled Sobol normal draws，而不是回退到 `torch`。
