# Swissmetro Experiment Summary

## Run
- `experiment_name`: `20260329_public_reverse_engineered_panel_mnl_biogeme_v1`
- `model`: `qwen3.5:9b`
- `completed_respondents`: `752` / `752`
- `tasks_per_respondent`: `9`

## Survey Quality
- `valid_choice_rate`: `1.0000`
- `grounding_parse_rate`: `1.0000`
- `avg_choice_duration_sec`: `0.38`
- `avg_respondent_total_duration_sec`: `3.38`

## Choice Shares
Human:
- `SWISSMETRO`: 0.6043
- `CAR`: 0.2615
- `TRAIN`: 0.1342
AI:
- `TRAIN`: 0.6767
- `CAR`: 0.1801
- `SWISSMETRO`: 0.1432

## Human Benchmark
- `final_loglikelihood`: `-5331.252`
- `rho_square`: `0.2345`
- `number_of_threads`: `27`

## AI MNL
- `final_loglikelihood`: `-5189.172`
- `rho_square`: `0.2565`
- `number_of_threads`: `27`

## AI vs Human
- `sign_match_rate`: `0.5000`
- `mean_abs_difference`: `0.9273`
- `human_time_cost_ratio`: `1.1791`
- `ai_time_cost_ratio`: `-0.1817`

## Significant AI Parameters
- none

## Notes
- `questionnaire_template_file`: `scripts/questionnaire_template.py`
- `grounding`: compact JSON to avoid truncation
- `conversation_style`: one growing multi-turn conversation per respondent
- `estimator`: Biogeme 4-parameter MNL
