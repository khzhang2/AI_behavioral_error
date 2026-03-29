# Optima Reduced Official-Style HCM Summary

## Run
- `experiment_name`: `20260329_optima_hybrid_choice_qwen35_9b_v1`
- `model`: `qwen3.5:9b`
- `completed_respondents`: `708` / `708`

## AI Survey Quality
- `valid_indicator_rate`: `1.0000`
- `valid_choice_rate`: `1.0000`
- `avg_indicator_duration_sec`: `0.59`
- `avg_choice_duration_sec`: `0.69`

## Biogeme 32
- `human_final_loglikelihood`: `-6959.313`
- `ai_final_loglikelihood`: `-4684.826`

## Torch 500
- `human_final_loglikelihood`: `-6957.891`
- `ai_final_loglikelihood`: `-4673.558`

## Torch32 vs Biogeme32 Alignment
- `human_sign_match_rate`: `1.0000`
- `ai_sign_match_rate`: `0.9024`
- `human_same_point_diff_at_biogeme`: `NA`
- `ai_same_point_diff_at_biogeme`: `NA`
- `alignment_passed`: `False`

## Human vs AI
- `biogeme32_sign_match_rate`: `0.6341`
- `torch500_sign_match_rate`: `0.5854`

## Choice Shares
- `choice_0`: human=`0.3136`, ai=`0.0494`, diff=`-0.2641`
- `choice_1`: human=`0.6257`, ai=`0.9463`, diff=`0.3206`
- `choice_2`: human=`0.0607`, ai=`0.0042`, diff=`-0.0565`

## Indicator Means
- `Envir01`: human_mean=`2.698`, ai_mean=`4.000`, tvd=`0.819`
- `Mobil05`: human_mean=`3.336`, ai_mean=`3.177`, tvd=`0.626`
- `LifSty07`: human_mean=`2.261`, ai_mean=`3.486`, tvd=`0.521`
- `Envir05`: human_mean=`3.606`, ai_mean=`4.059`, tvd=`0.314`
- `Mobil12`: human_mean=`2.062`, ai_mean=`4.168`, tvd=`0.691`
- `LifSty01`: human_mean=`2.610`, ai_mean=`2.945`, tvd=`0.400`

## Notes
- `questionnaire_template_file`: `scripts/questionnaire_template.py`
- `grounding`: compact JSON
- `conversation_style`: one growing conversation per respondent
- `indicators`: Envir01, Mobil05, LifSty07, Envir05, Mobil12, LifSty01
