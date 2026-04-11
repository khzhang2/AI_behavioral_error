# Optima Latent Response Regime Experiment Summary

This archive records the first latent response regime experiment built on the retained Optima benchmark. The experiment is artificial-intelligence-first: the human Optima data are used only to estimate a normative benchmark multinomial logit model and to seed respondent profiles and scenario attributes, whereas the repeated-task panel data are newly collected from the artificial-intelligence survey sessions.

## Data collection quality
### qwen3.5_9b
The completed respondent count is `4` out of `4`. The valid-attitude rate is `1.0000`, and the valid-task rate is `1.0000`. The mean label-flip rate is `0.0000`, the mean order-flip rate is `0.0000`, the mean monotonicity-compliance rate is `1.0000`, and the mean dominance-violation rate is `0.0000`.

### deepseek_r1_8b
The completed respondent count is `4` out of `4`. The valid-attitude rate is `1.0000`, and the valid-task rate is `1.0000`. The mean label-flip rate is `0.0000`, the mean order-flip rate is `1.0000`, the mean monotonicity-compliance rate is `1.0000`, and the mean dominance-violation rate is `1.0000`.

## Human benchmark and pooled artificial-intelligence baseline
The human benchmark multinomial logit model uses `708` respondents and `708` tasks, with a final log likelihood of `-328.048`. The pooled artificial-intelligence panel multinomial logit model uses `8` respondents and `128` tasks, with a final log likelihood of `-1656.833`.

## Scale-adjusted latent class choice model
The pooled artificial-intelligence scale-adjusted latent class choice model is estimated with `3` preference classes and `2` scale classes. The final log likelihood is `-1570.590`. The posterior probabilities sum to one up to the numerical range `1.000000` to `1.000000`.

## Regime interpretation
The regime `human_like_tradeoff` corresponds to preference class `2` and scale class `1`. It has posterior mass `0.1250`, normalized coefficient distance `46.1714`, `6` sign mismatches relative to the human benchmark, mode-share deviation `0.5632`, label-flip rate `0.0000`, order-flip rate `1.0000`, monotonicity-compliance rate `1.0000`, and dominance-violation rate `1.0000`.
The regime `human_like_tradeoff` corresponds to preference class `2` and scale class `2`. It has posterior mass `0.3750`, normalized coefficient distance `46.1714`, `6` sign mismatches relative to the human benchmark, mode-share deviation `0.5840`, label-flip rate `0.0000`, order-flip rate `1.0000`, monotonicity-compliance rate `1.0000`, and dominance-violation rate `1.0000`.
The regime `label_sensitive` corresponds to preference class `3` and scale class `1`. It has posterior mass `0.2496`, normalized coefficient distance `72.0252`, `2` sign mismatches relative to the human benchmark, mode-share deviation `0.1885`, label-flip rate `0.0000`, order-flip rate `0.0000`, monotonicity-compliance rate `1.0000`, and dominance-violation rate `0.0000`.
The regime `label_sensitive` corresponds to preference class `3` and scale class `2`. It has posterior mass `0.0004`, normalized coefficient distance `72.0252`, `2` sign mismatches relative to the human benchmark, mode-share deviation `0.2515`, label-flip rate `0.0000`, order-flip rate `0.0000`, monotonicity-compliance rate `1.0000`, and dominance-violation rate `0.0000`.
The regime `label_sensitive` corresponds to preference class `1` and scale class `1`. It has posterior mass `0.0000`, normalized coefficient distance `84.2734`, `1` sign mismatches relative to the human benchmark, mode-share deviation `0.2672`, label-flip rate `0.0000`, order-flip rate `0.0000`, monotonicity-compliance rate `1.0000`, and dominance-violation rate `0.0000`.
The regime `label_sensitive` corresponds to preference class `1` and scale class `2`. It has posterior mass `0.2500`, normalized coefficient distance `84.2734`, `1` sign mismatches relative to the human benchmark, mode-share deviation `0.3118`, label-flip rate `0.0000`, order-flip rate `0.0000`, monotonicity-compliance rate `1.0000`, and dominance-violation rate `0.0000`.

## Distortion score
The posterior respondent-level distortion score has mean `63.0748`, minimum `47.7346`, and maximum `84.5852`.
