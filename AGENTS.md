# AI Behavioral Error - Project Guide

## Project Overview

This is an academic research codebase studying whether LLM-generated synthetic respondents can reproduce human behavioral patterns in discrete choice models. The project uses the Optima dataset (Swissmetro travel survey) as the benchmark, collects AI survey responses via LLMs, and compares the estimated Hybrid Choice Models (HCM) and Multinomial Logit (MNL) models between human and AI respondents.

Python runtime: `.\.venv\Scripts\python.exe` (Windows)

## Directory Structure

```
AI_behavioral_error/
├── data/                          # All datasets (human + AI collected)
│   └── Swissmetro/
│       └── demographic_choice_psychometric/
│           ├── raw/               # Original Optima .dat file from Biogeme distribution
│           ├── ai_collection_qwen3.5_9b/    # AI survey outputs (Qwen 3.5 9B)
│           ├── ai_collection_deepseek_r1_8b/ # AI survey outputs (DeepSeek R1 8B)
│           ├── human_cleaned_wide.csv        # Cleaned human data (wide format)
│           ├── human_cleaned_long.csv        # Cleaned human data (long format)
│           ├── human_respondent_profiles.csv # Human respondent profiles
│           ├── shared_sobol_draws_32.npy     # Shared Sobol draws (32 draws)
│           ├── shared_sobol_draws_500.npy    # Shared Sobol draws (500 draws)
│           ├── human_benchmark_sample_summary.json
│           ├── optima_codebook.json
│           └── optima_data_description.md
├── scripts/                       # All runnable scripts (flat structure)
├── experiments/                   # Archived experiment outputs (read-only after generation)
│   └── Swissmetro/
│       ├── 20260329_optima_hybrid_choice_qwen35_9b_v1/
│       ├── 20260330_optima_hybrid_choice_deepseek_r1_8b_v1/
│       ├── 20260330_optima_mnl_deepseek_r1_8b_v1/
│       └── 20260330_optima_mnl_qwen35_9b_v1/
├── docs/                          # Documentation and audit notes
├── experiment_config.json         # Active experiment configuration (single source of truth)
├── biogeme.toml                   # Biogeme default parameter file (gitignored)
├── biogeme_runtime.toml           # Biogeme runtime overrides (written by scripts, gitignored)
├── pyproject.toml                 # Python project metadata and dependencies
└── AGENTS.md                      # This file
```

### `scripts/`

All runnable Python scripts in a flat structure. No sub-packages. Scripts import from each other directly:

### `experiments/`

Archived outputs only. Each experiment run is stored under `Swissmetro/<date>_<model>_<type>_v<version>/`. Contains:

- `experiment_config.json` (and numbered variants `experiment_config_N.json`) - snapshots of the active config at run time.
- `outputs/` - organized by estimation method and draw count:
  - `{dataset}_biogeme_{n_draws}/` - Biogeme HCM outputs
  - `{dataset}_torch_{n_draws}/` - Torch HCM outputs
  - `{dataset}_mnl_{spec}/` - Biogeme MNL outputs (e.g. `ai_basic`, `human_user_no_od`)
  - `aggregate/` - comparison summaries and the experiment summary Markdown


## Key Conventions

- All paths are resolved relative to the project root via `optima_common.ROOT_DIR = Path(__file__).resolve().parents[1]`.
- The active experiment is determined entirely by `experiment_config.json`. Changing `paths.archive_dir` or `llm.model` switches the target experiment.
- AI collection supports resume via `--resume` flag; it replays `raw_interactions.jsonl` to rebuild state.
- Invalid LLM responses trigger a single repair attempt before moving on.
- Sobol draws are pre-generated and shared across all estimation methods to ensure identical Monte Carlo integration.
- Biogeme `*.iter` files and `biogeme.toml` / `biogeme_runtime.toml` are gitignored; the runtime TOML is rewritten on each Biogeme estimation run.
