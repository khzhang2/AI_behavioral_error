# Optima Latent Response Regime Experiment Summary

This archive records the current intervention-anchored latent response regime experiment built on the retained Optima benchmark. The analysis is layered. It first asks whether exact-repeat randomness is empirically identified, then asks whether the observed intervention effects exceed that randomness envelope, and only then interprets the scale-adjusted latent class choice model (SALCM) as a summary of latent response regimes rather than as a stand-alone mixture fit.

## Data collection quality
### qwen3.5_9b
The completed respondent count is `2` out of `2`. The valid-attitude rate is `1.0000`, and the valid-task rate is `1.0000`. The mean label-flip rate is `0.0000`, the mean order-flip rate is `0.0000`, the mean monotonicity-compliance rate is `1.0000`, and the mean dominance-violation rate is `0.0000`.

### deepseek_r1_8b
The completed respondent count is `2` out of `2`. The valid-attitude rate is `0.9167`, and the valid-task rate is `1.0000`. The mean label-flip rate is `0.0000`, the mean order-flip rate is `1.0000`, the mean monotonicity-compliance rate is `1.0000`, and the mean dominance-violation rate is `1.0000`.

## Human benchmark and pooled artificial-intelligence baseline
The human benchmark multinomial logit model uses `708` respondents and `708` tasks, with a final log likelihood of `-328.048`. The pooled artificial-intelligence panel multinomial logit model uses `4` respondents and `64` tasks, with a final log likelihood of `-2826.902`.

## Randomness envelope
The pooled data contain `32` exact-repeat choice signatures. The repeat flip rate is `0.0000`, and the repeat response entropy is `0.0000`.

## Intervention effects
The current analysis reports intervention effects for paired paraphrase tasks, paired label-mask tasks, paired order-randomization tasks, the between-block prompt-arm contrast, and the diagnostic monotonicity and dominance tasks.
The pooled `label_mask` intervention yields a total-variation distance of `0.0000` and a paired flip rate of `0.0000`. The mean confidence shift is `0.0000`.
The pooled `order_randomization` intervention yields a total-variation distance of `0.8885` and a paired flip rate of `0.8885`. The mean confidence shift is `0.0000`.
The pooled `paraphrase` intervention yields a total-variation distance of `0.0000` and a paired flip rate of `0.0000`. The mean confidence shift is `0.0000`.
The pooled monotonicity-compliance rate is `0.5000`.
The pooled dominance-violation rate is `0.8885`.

## Scale-adjusted latent class choice model
The pooled artificial-intelligence SALCM is estimated with `3` preference classes and `2` scale classes. The final log likelihood is `-2809.782`, and `6` states have non-negligible posterior mass.

## Regime interpretation under the intervention-anchored framework
The regime `human_like_coherent` corresponds to `C3_S1` with posterior mass `0.4970`. Its normalized coefficient distance from the human benchmark is `12.2586`, and its mode-share deviation is `0.2493`. The label-mask total-variation gap is `0.0000`, the order-randomization total-variation gap is `0.0000`, the prompt-arm total-variation gap is `NA`, the monotonicity-compliance rate is `1.0000`, the dominance-violation rate is `0.0000`, and the composite intervention-signature score is `0.0000`.
The regime `label_sensitive_coherent` corresponds to `C3_S2` with posterior mass `0.0030`. Its normalized coefficient distance from the human benchmark is `12.2586`, and its mode-share deviation is `0.2493`. The label-mask total-variation gap is `0.0000`, the order-randomization total-variation gap is `0.0000`, the prompt-arm total-variation gap is `NA`, the monotonicity-compliance rate is `1.0000`, the dominance-violation rate is `0.0000`, and the composite intervention-signature score is `0.0000`.
The regime `distorted_tradeoff` corresponds to `C2_S1` with posterior mass `0.3224`. Its normalized coefficient distance from the human benchmark is `17.6670`, and its mode-share deviation is `0.5614`. The label-mask total-variation gap is `0.0000`, the order-randomization total-variation gap is `1.0000`, the prompt-arm total-variation gap is `NA`, the monotonicity-compliance rate is `1.0000`, the dominance-violation rate is `1.0000`, and the composite intervention-signature score is `2.0000`.
The regime `distorted_tradeoff` corresponds to `C2_S2` with posterior mass `0.1315`. Its normalized coefficient distance from the human benchmark is `17.6670`, and its mode-share deviation is `0.5614`. The label-mask total-variation gap is `0.0000`, the order-randomization total-variation gap is `1.0000`, the prompt-arm total-variation gap is `NA`, the monotonicity-compliance rate is `1.0000`, the dominance-violation rate is `1.0000`, and the composite intervention-signature score is `2.0000`.
The regime `low_consistency_distorted` corresponds to `C1_S1` with posterior mass `0.0326`. Its normalized coefficient distance from the human benchmark is `22.3037`, and its mode-share deviation is `0.5614`. The label-mask total-variation gap is `0.0000`, the order-randomization total-variation gap is `1.0000`, the prompt-arm total-variation gap is `NA`, the monotonicity-compliance rate is `1.0000`, the dominance-violation rate is `1.0000`, and the composite intervention-signature score is `2.0000`.
The regime `distorted_tradeoff` corresponds to `C1_S2` with posterior mass `0.0135`. Its normalized coefficient distance from the human benchmark is `22.3037`, and its mode-share deviation is `0.5614`. The label-mask total-variation gap is `0.0000`, the order-randomization total-variation gap is `1.0000`, the prompt-arm total-variation gap is `NA`, the monotonicity-compliance rate is `1.0000`, the dominance-violation rate is `1.0000`, and the composite intervention-signature score is `2.0000`.

## Distortion score
The original posterior distortion score has mean `16.0821`, minimum `12.5079`, and maximum `19.6562`. The v2 intervention-anchored distortion score has mean `17.0821`, minimum `12.5079`, and maximum `21.6562`.
