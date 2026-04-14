---
name: optima-experiment-workflow
description: Run post-AI analysis for a completed Optima intervention-regime experiment in this repository. Use when Codex needs to validate that AI questionnaire collection has finished, inspect whether the collected responses are complete enough for analysis, run intervention metrics or discrete-choice estimation after collection, interpret the five error dimensions, or write the final Chinese experiment summary. Do not use this skill to launch or resume AI questionnaire collection.
---

# Optima Experiment Workflow

## Overview

Use this skill only for post-AI analysis in the current intervention-regime workflow. Start from an experiment folder that should already contain AI collection outputs. Do not use this skill to prepare personas, generate questionnaire tasks, run AI collection, or resume interrupted collection.

Use the current intervention-regime line as the default analysis workflow. Treat older hybrid-choice or legacy multi-output layouts as reference only unless the user explicitly asks for them.

## Define the main objects first

- `task`: one question. It can be one attitude question or one choice card.
- `block template`: one reusable questionnaire-session template.
- `respondent block`: one `block template` under fixed experimental conditions such as model, prompt arm, persona, and temperature.
- `run`: one full execution of one `respondent block`.

In this repository, one full questionnaire run means:

`1 grounding + n_attitudes + total task cards`

## Confirm the target experiment first

Work with one experiment folder and one model at a time.

- Use the naming rule `YYYYMMDD_<keywords>_<version>`.
- Keep one experiment folder for one model only.
- Keep `llm_models` in `experiment_config.json` to one entry only for the active experiment-ready workflow.
- Treat `paths.archive_dir / experiment_name` as the real experiment folder.

## Validate AI collection before any estimation

Always check completion first. Do this before running any post-AI script.

Read at least:

- `outputs/run_respondents.json`
- `parsed_task_responses.csv`
- `parsed_attitudes.csv`

Use this logic.

1. Compare `completed_respondents` and `target_respondents` in `outputs/run_respondents.json`.
2. Check whether the parsed task table is present and nonempty.
3. Check whether respondents expected to be complete actually have the full number of task rows needed for analysis.

If the AI collection is not complete, stop there.

- Report that the experiment is not ready for post-AI analysis.
- State the completion counts or missing files.
- Do not run AI questionnaire collection.
- Do not resume AI questionnaire collection.

This skill must never launch:

```bash
./.venv/bin/python scripts/run_optima_intervention_regime_ai_collection.py ...
```

or any other AI collection command.

## Know where the inputs and outputs live

Use this storage rule consistently.

`outputs/` stores raw AI collection artifacts only:

- `outputs/raw_interactions.jsonl`
- `outputs/respondent_transcripts.json`
- `outputs/run_respondents.json`
- `outputs/ai_collection_summary.json`

The experiment root stores shared derived AI data, shared diagnostics, and the final report:

- `persona_samples.csv`
- `parsed_attitudes.csv`
- `parsed_task_responses.csv`
- `ai_panel_long.csv`
- `ai_panel_block.csv`
- all shared diagnostics and summaries
- `atasoy_2011_replication/`
- `hcm/ai`, `hcm/human`
- `salcm/ai`, `salcm/human`

## Run the post-AI estimation sequence

After the experiment passes the completion check, use this order for the current intervention-regime workflow.

1. Estimate intervention and randomness metrics:

```bash
./.venv/bin/python scripts/estimate_optima_intervention_metrics.py
```

2. Reproduce the paper's human base logit and continuous model:

```bash
./.venv/bin/python scripts/replicate_atasoy_2011_models.py
```

3. Estimate the AI-side Atasoy 2011 base logit:

```bash
./.venv/bin/python scripts/estimate_atasoy_2011_ai_analysis.py --experiment-dirs <experiment_name>
```

4. Estimate the scale-adjusted latent class model:

```bash
./.venv/bin/python scripts/estimate_optima_salcm.py
```

5. Write the short experiment summary:

```bash
./.venv/bin/python scripts/summarize_optima_intervention_regime.py
```

## Read the results using the five error dimensions

Use the following mapping when interpreting one completed experiment.

1. Random instability within one model:
   Read `exact_repeat_randomness.csv`, `intervention_metrics_summary.json`, and `outputs/ai_collection_summary.json`.
   Focus on `exact-repeat flip rate` and `response entropy`.

2. Semantic invariance:
   Read `intervention_sensitivity.csv` and `ai_panel_block.csv`.
   Focus on `paraphrase gap`, `paraphrase flip rate`, and whether the excess gap rises above the randomness baseline.

3. Label or order sensitivity:
   Read `intervention_sensitivity.csv` and `ai_panel_block.csv`.
   Compare `label gap` with `order gap`. Do not collapse them too early; one may be weak while the other is strong.

4. Trade-off fidelity:
   Read `ai_panel_block.csv` and `outputs/ai_collection_summary.json`.
   Focus on `dominance violation rate` and `monotonicity compliance rate`.

5. Human-relative distortion:
   Read `data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/base_logit_summary.json`, `atasoy_2011_replication/ai_atasoy_base_logit_summary.json`, and `salcm/ai/ai_salcm_regime_summaries.csv`.
   Start with `choice share` differences. Treat `VOT/WTP` or elasticity-style interpretation more cautiously when the optimizer reports precision loss or iteration limits.

## Follow the experiment record rules

Keep the experiment archive clean.

- Do not mix multiple models in one experiment folder.
- Do not store diagnostics or estimation results under `outputs/`.
- Keep one root `experiment_summary.md` only.
- Keep `experiment_summary.md` short and decision-oriented.
- Prefer writing the summary after the main analysis scripts finish.

## Write the experiment report in Chinese

Write `experiment_summary.md` in Chinese by default for this repository unless the user explicitly asks for another language.

Use a compact structure.

1. One short opening paragraph stating the experiment target, sample size, and the main overall conclusion.
2. One Markdown table summarizing the five error dimensions.
3. One short closing paragraph stating the main caveat, especially when the optimizer reports precision loss or iteration limits.

Use the five error dimensions as fixed report rows:

- 同一系统的随机不稳定性
- 对语义等价改写是否稳健
- 对标签或顺序是否过敏
- 是否真的在做 trade-off
- 是否只是总体像人

Use a Markdown table with columns of this form:

| 检验对象 | 这次试验怎么概括 | 主要数值 | 解释 |
| --- | --- | --- | --- |

Keep the wording concrete. Report the actual metric values from the current experiment rather than generic placeholders. When possible, use the same metrics already produced by the pipeline, such as `exact-repeat flip rate`, `response entropy`, `paraphrase flip rate`, `label flip rate`, `order flip rate`, `monotonicity compliance rate`, `dominance violation rate`, and `share gap`.
