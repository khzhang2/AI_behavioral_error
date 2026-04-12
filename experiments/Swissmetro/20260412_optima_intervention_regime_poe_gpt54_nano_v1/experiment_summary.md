# Experiment Summary: poe_gpt54_nano

This archive records one model only: `poe_gpt54_nano`. The AI collection completed `400` of `400` planned respondent runs. The valid-attitude rate is `1.0000` and the valid-task rate is `1.0000`.

The main conclusion from this experiment is that the largest error is not pure randomness, but distortion relative to the human benchmark. The model shows some repeated-response instability, but the stronger pattern is a persistent over-selection of `CAR` and under-selection of `PT` and `SLOW_MODES`.

For within-model randomness, the mean exact-repeat flip rate is `0.0730` and the mean response entropy is `0.0806`. This indicates low-to-moderate instability under exactly repeated inputs. The model is therefore not fully deterministic, but repeated randomness is not the dominant source of error in this experiment.

For semantic invariance, the error is comparatively weak. The mean paraphrase flip rate is `0.0313`, the mean paraphrase intervention gap is `0.0313`, and the mean paraphrase excess gap is `-0.0583`. This means paraphrase responses do change sometimes, but not more than what would be expected from the repeated-response randomness baseline.

For label and order sensitivity, the two effects separate clearly. Label sensitivity is weak, with mean label flip rate `0.0388` and mean label excess gap `-0.0610`. Order sensitivity is much stronger, with mean order flip rate `0.1313` and mean order excess gap `0.0435`. In this experiment, the model is better described as order-sensitive than label-sensitive.

For trade-off fidelity, the model broadly follows basic consistency rules, but not perfectly. The mean monotonicity compliance rate is `0.9550`, and the mean dominance violation rate is `0.0625`. This suggests that the model usually performs coherent trade-off reasoning, while still showing a nontrivial rate of rule violations.

For validity relative to the human benchmark, distortion is substantial. The AI choice shares are `CAR = 0.8667`, `PT = 0.1239`, and `SLOW_MODES = 0.0094`, whereas the human benchmark shares are `CAR = 0.6257`, `PT = 0.3136`, and `SLOW_MODES = 0.0607`. The total variation share gap is `0.2410`. This is the strongest error dimension in the present experiment.

The overall ranking of the five error dimensions is: human-relative distortion first, order sensitivity second, repeated-response randomness third, trade-off fidelity error fourth, and semantic invariance plus label sensitivity as the weakest errors. The MNL and SALCM outputs are useful for directional interpretation, but they should still be treated cautiously because the AI panel MNL reports precision loss and the SALCM stopped at the iteration limit.

| 检验对象 | 这次试验怎么概括 | 主要数值 | 解释 |
| --- | --- | --- | --- |
| 同一系统的随机不稳定性 | `低到中等` | exact-repeat flip rate = `0.0730`；response entropy = `0.0806` | 完全相同输入下会有一些翻转，但幅度不算高。说明有随机性，但不是主要问题。 |
| 对语义等价改写是否稳健 | `较稳健` | paraphrase flip rate = `0.0313`；paraphrase gap = `0.0313`；paraphrase excess gap = `-0.0583` | 改写措辞后确实会有少量变化，但平均上没有超出重复随机性的基线，所以目前更像轻微 prompt fragility，而不是强语义不变性 failure。 |
| 对标签/顺序是否过敏 | `标签弱，顺序明显` | label flip rate = `0.0388`；label excess gap = `-0.0610`；order flip rate = `0.1313`；order excess gap = `0.0435` | 标签掩码影响不大，但顺序随机化影响明显高得多。这个模型更像是 `order-sensitive`，而不是典型 `label-sensitive regime`。 |
| 是否真的在做 trade-off | `大体在做，但不完全稳定` | monotonicity compliance = `0.9550`；dominance violation = `0.0625` | 大多数时候遵守基本理性规则，所以不是“完全不做 trade-off”。但仍有约 `4.5%` monotonicity failure 和 `6.25%` dominance failure，说明 fidelity 不是满分。 |
| 是否只是“总体像人” | `不像，失真明显` | AI shares: `CAR 0.8667 / PT 0.1239 / SLOW 0.0094`；human shares: `CAR 0.6257 / PT 0.3136 / SLOW 0.0607`；share gap 的总变差距离 = `0.2410` | 这是这次最强的 error。模型显著过度偏向 `CAR`，显著低估 `PT`，几乎不选 `SLOW_MODES`。relative-to-human validity 明显不足。 |
