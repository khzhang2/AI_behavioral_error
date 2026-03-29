# Swissmetro AI vs Human Report

## Experiment Positioning

- `replication_standard = public_reverse_engineering`
- This line is a public reverse engineering of Swissmetro, not an exact historical DOE recovery.
- LLM model: `qwen3.5:9b`
- AI runs completed: `3`

## Human Benchmark Validation

- cleaned observations: `6768`
- cleaned respondents: `752`
- target final loglikelihood: `-5331.252`
- Biogeme final loglikelihood: `-5331.252`
- benchmark LL delta: `-0.000007`

### Human MNL Estimates

| parameter_name | estimate | std_error | robust_std_error |
| --- | --- | --- | --- |
| ASC_CAR | -0.1546 | - | 0.0582 |
| ASC_TRAIN | -0.7012 | - | 0.0826 |
| B_COST | -1.0838 | - | 0.0682 |
| B_TIME | -1.2779 | - | 0.1043 |

## Reverse-Engineered Design

- design type: `other`
- classification: `respondent-specific pivoted stated-preference design with recurring 9-task template families`
- tasks per respondent: `[9]`
- `TRAIN_HE` levels: `[30, 60, 120]`
- `SM_HE` levels: `[10, 20, 30]`
- `SM_SEATS` levels: `[0, 1]`

## AI Run Summary

| run_label | final_loglikelihood | sign_match_rate | ai_time_cost_ratio | ratio_diff |
| --- | --- | --- | --- | --- |
| ai_run_01 | -5814.093 | 1.0000 | 12.9354 | 11.7563 |
| ai_run_02 | -5648.532 | 1.0000 | 10.4149 | 9.2359 |
| ai_run_03 | -5746.913 | 1.0000 | 12.1766 | 10.9975 |

## Coefficient Comparison

| run_label | parameter_name | human_estimate | ai_estimate | difference_ai_minus_human | sign_match |
| --- | --- | --- | --- | --- | --- |
| ai_run_01 | ASC_CAR | -0.1546 | -0.4479 | -0.2932 | 1 |
| ai_run_01 | ASC_TRAIN | -0.7012 | -1.1197 | -0.4185 | 1 |
| ai_run_01 | B_COST | -1.0838 | -0.0339 | 1.0499 | 1 |
| ai_run_01 | B_TIME | -1.2779 | -0.4379 | 0.8399 | 1 |
| ai_run_02 | ASC_CAR | -0.1546 | -0.6407 | -0.4861 | 1 |
| ai_run_02 | ASC_TRAIN | -0.7012 | -1.2208 | -0.5196 | 1 |
| ai_run_02 | B_COST | -1.0838 | -0.0371 | 1.0467 | 1 |
| ai_run_02 | B_TIME | -1.2779 | -0.3866 | 0.8912 | 1 |
| ai_run_03 | ASC_CAR | -0.1546 | -0.5518 | -0.3972 | 1 |
| ai_run_03 | ASC_TRAIN | -0.7012 | -1.1896 | -0.4884 | 1 |
| ai_run_03 | B_COST | -1.0838 | -0.0332 | 1.0506 | 1 |
| ai_run_03 | B_TIME | -1.2779 | -0.4045 | 0.8733 | 1 |

## Overall Choice Shares

| source_label | alternative_name | count | share |
| --- | --- | --- | --- |
| human | TRAIN | 908 | 0.1342 |
| human | SWISSMETRO | 4090 | 0.6043 |
| human | CAR | 1770 | 0.2615 |
| ai_run_01 | TRAIN | 949 | 0.1402 |
| ai_run_01 | SWISSMETRO | 4035 | 0.5962 |
| ai_run_01 | CAR | 1784 | 0.2636 |
| ai_run_02 | TRAIN | 951 | 0.1405 |
| ai_run_02 | SWISSMETRO | 4230 | 0.6250 |
| ai_run_02 | CAR | 1587 | 0.2345 |
| ai_run_03 | TRAIN | 937 | 0.1384 |
| ai_run_03 | SWISSMETRO | 4114 | 0.6079 |
| ai_run_03 | CAR | 1717 | 0.2537 |

## Subgroup Choice Shares

| source_label | group_type | group_value | alternative_name | share |
| --- | --- | --- | --- | --- |
| human | GA | 0 | TRAIN | 0.0833 |
| human | GA | 0 | SWISSMETRO | 0.6213 |
| human | GA | 0 | CAR | 0.2953 |
| human | GA | 1 | TRAIN | 0.4656 |
| human | GA | 1 | SWISSMETRO | 0.4933 |
| human | GA | 1 | CAR | 0.0411 |
| human | CAR_AV | 0 | TRAIN | 0.3842 |
| human | CAR_AV | 0 | SWISSMETRO | 0.6158 |
| human | CAR_AV | 0 | CAR | 0.0000 |
| human | CAR_AV | 1 | TRAIN | 0.0824 |
| human | CAR_AV | 1 | SWISSMETRO | 0.6019 |
| human | CAR_AV | 1 | CAR | 0.3157 |
| ai_run_01 | GA | 0 | TRAIN | 0.1620 |
| ai_run_01 | GA | 0 | SWISSMETRO | 0.5356 |
| ai_run_01 | GA | 0 | CAR | 0.3024 |
| ai_run_01 | GA | 1 | TRAIN | 0.0073 |
| ai_run_01 | GA | 1 | SWISSMETRO | 0.9654 |
| ai_run_01 | GA | 1 | CAR | 0.0273 |
| ai_run_01 | CAR_AV | 0 | TRAIN | 0.3060 |
| ai_run_01 | CAR_AV | 0 | SWISSMETRO | 0.6940 |
| ai_run_01 | CAR_AV | 0 | CAR | 0.0000 |
| ai_run_01 | CAR_AV | 1 | TRAIN | 0.1081 |
| ai_run_01 | CAR_AV | 1 | SWISSMETRO | 0.5772 |
| ai_run_01 | CAR_AV | 1 | CAR | 0.3146 |

## Run-to-Run Stability

| parameter_name | ai_mean_estimate | ai_std_estimate | ai_min_estimate | ai_max_estimate |
| --- | --- | --- | --- | --- |
| ASC_CAR | -0.5468 | 0.0965 | -0.6407 | -0.4479 |
| ASC_TRAIN | -1.1767 | 0.0518 | -1.2208 | -1.1197 |
| B_COST | -0.0347 | 0.0021 | -0.0371 | -0.0332 |
| B_TIME | -0.4097 | 0.0260 | -0.4379 | -0.3866 |

## Main Reading

- The human Biogeme benchmark is expected to match the pylogit notebook closely because it estimates the same four-parameter MNL with the same cleaning and scaling rules.
- The AI side should be read as a behavioral simulation benchmark: synthetic personas, reconstructed panel families, and one-task-at-a-time prompting.
- The key comparison objects are coefficient signs and magnitudes, mode shares, subgroup share structure by `GA` and `CAR_AV`, and the implied time-cost ratio.
