# Swissmetro Public Reverse Engineering (`v1`)

## 定位

这个目录实现的是：

- `public_reverse_engineering`
- `Swissmetro`
- `Biogeme` `4` 参数 `MNL`
- `3` 轮独立 AI run

它不是对历史 Swissmetro DOE 文件或原始问卷脚本的 exact recovery。这里的 panel 设计来自对 `swissmetro.dat` 和公开 notebook / 说明材料的逆向工程，因此必须视为公开材料重建。

## 目录内容

- `data/raw/swissmetro.dat`
- `data/data_provenance.json`
- `data/pylogit_benchmark_targets.json`
- `data/human_cleaned_wide.csv`
- `data/human_cleaned_long.csv`
- `data/human_respondent_profiles.csv`
- `data/reconstructed_panel_catalog.csv`
- `data/reconstructed_panel_baselines.csv`
- `data/swissmetro_reverse_engineering.md`
- `data/swissmetro_design_spec.json`
- `data/swissmetro_codebook.json`
- `data/survey_briefing_en.md`
- `data/questionnaire_template_en.md`
- `scripts/fetch_swissmetro_data.py`
- `scripts/prepare_human_benchmark.py`
- `scripts/reverse_engineer_design.py`
- `scripts/generate_ai_panels.py`
- `scripts/run_ai_collection.py`
- `scripts/estimate_biogeme_mnl.py`
- `scripts/build_comparison.py`
- `scripts/summarize_experiment.py`

## 运行环境

推荐直接使用仓库根目录虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[experiments]"
```

## 运行顺序

1. 获取并冻结公开数据

```powershell
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\fetch_swissmetro_data.py
```

2. 准备人类 benchmark 数据

```powershell
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\prepare_human_benchmark.py
```

3. 逆向工程 panel 设计

```powershell
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\reverse_engineer_design.py
```

4. 估计人类 benchmark

```powershell
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\estimate_biogeme_mnl.py --dataset-role human
```

5. 生成并运行三轮 AI survey

```powershell
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\generate_ai_panels.py --run-id 1
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\run_ai_collection.py --run-id 1
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\estimate_biogeme_mnl.py --dataset-role ai --run-id 1
```

对 `run-id = 2` 和 `run-id = 3` 重复同样三步。

6. 汇总对比与报告

```powershell
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\build_comparison.py
.\.venv\Scripts\python.exe experiments\Swissmetro\20260329_public_reverse_engineered_panel_mnl_biogeme_v1\scripts\summarize_experiment.py
```

## 关键输出

- `outputs/human_benchmark/`
- `outputs/ai_run_01/`
- `outputs/ai_run_02/`
- `outputs/ai_run_03/`
- `outputs/aggregate/experiment_report.md`

## 关键约束

- 人类 benchmark 必须复现 pylogit notebook 的 `4` 参数 `MNL`
- AI survey 必须逐题问答，不能一次性要求模型答完 9 题
- 参数估计只允许使用 `Biogeme`
