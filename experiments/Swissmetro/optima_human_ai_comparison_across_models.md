# Comparative Note on Human and Artificial-Intelligence Behavior Across the Optima Hybrid Choice Model and the Multinomial Logit Model

This note compares the retained Optima experiments across two artificial-intelligence (AI) models, `qwen3.5:9b` and `deepseek-r1:8b`, and across two behavioral model classes, namely the reduced official-style hybrid choice model (HCM) and the multinomial logit (MNL) model. The purpose of this comparison is not merely to report model fit, but to identify where human and AI behavior differ, and to determine whether such differences are more visible in aggregate mode shares or in the underlying structural parameters.

The human benchmark is common across all retained Optima exercises, because all experiments use the same cleaned human data package in `data/Swissmetro/demographic_choice_psychometric`. Consequently, the comparison across `qwen3.5:9b` and `deepseek-r1:8b` should be interpreted as a comparison of AI-generated responses rather than a comparison of different human benchmarks. The two AI lines differ only in the large language model used for the questionnaire simulation. All other major survey settings were kept aligned.

## Data quality and effective sample size

The first difference between the two AI lines appears already at the questionnaire stage. The `qwen3.5:9b` line yielded fully valid psychometric and final choice responses, with both the valid-indicator rate and the valid-choice rate equal to `1.0000`. By contrast, the `deepseek-r1:8b` line produced a valid-indicator rate of `0.8044` and a valid-choice rate of `0.8037`. As a result, the final deepseek-based estimation retained only `568` valid AI observations, whereas the qwen-based estimation retained all `708` observations.

This distinction matters for interpretation. In the deepseek line, any direct comparison of final log-likelihood values with the qwen line is confounded by the smaller usable sample. It is therefore more informative to compare the aggregate mode shares and the estimated utility parameters than to compare raw log-likelihood values across the two AI models.

## Aggregate mode shares

The most transparent comparison is the distribution of mode choices. The human benchmark contains approximately `31.36%` public transport (`PT`), `62.57%` car (`CAR`), and `6.07%` slow modes (`SLOW_MODES`). The qwen AI sample is dramatically more car-dominant, with approximately `4.94%` `PT`, `94.63%` `CAR`, and `0.42%` `SLOW_MODES`. The deepseek AI sample is much closer to the human benchmark after the validity filter, with approximately `34.51%` `PT`, `63.03%` `CAR`, and `2.46%` `SLOW_MODES`.

These mode shares imply that deepseek is visibly closer to the human benchmark at the level of aggregate behavior. Qwen produces an extreme car concentration, whereas deepseek reproduces the rough human balance between public transport and car much more closely. However, the aggregate mode shares do not by themselves establish that deepseek has recovered the human utility structure. The structural models show that this conclusion would be premature.

| Data source | `PT` share | `CAR` share | `SLOW_MODES` share |
|---|---:|---:|---:|
| Human benchmark | `0.3136` | `0.6257` | `0.0607` |
| Qwen AI | `0.0494` | `0.9463` | `0.0042` |
| Deepseek AI (valid sample) | `0.3451` | `0.6303` | `0.0246` |

## Multinomial logit comparison

The MNL results show that both AI models diverge from the human benchmark in the basic trade-off structure, even though the degree of divergence differs sharply. In the human four-parameter MNL, the generic cost coefficient and the generic time coefficient are both negative, with `B_COST = -0.6135` and `B_TIME = -0.3391`. This pattern is behaviorally conventional and implies that greater generalized cost reduces utility.

The qwen AI basic MNL reproduces neither the human mode shares nor the human trade-off structure. Its generic cost coefficient is only weakly negative at `-0.0729`, while its generic time coefficient becomes positive at `0.1600`. The resulting fit is almost degenerate once respondent characteristics are added: the final log likelihood of the respondent-characteristic model is `-0.0006`, which is far too close to zero to be read as realistic behavioral recovery. Instead, it indicates that the AI responses are highly deterministic and unusually easy to classify.

The deepseek AI basic MNL is more realistic in the sense that it does not collapse into the same extreme car-dominant pattern. The generic cost coefficient remains negative at `-0.1972`, and the basic-model final log likelihood of `-385.932` is far from the near-perfect qwen outcome. Nevertheless, the deepseek generic time coefficient is still positive at `0.1892`, which remains behaviorally implausible. In other words, deepseek improves the aggregate distribution of choices, but it still does not reproduce the human disutility of travel time.

| MNL basic coefficient | Human | Qwen AI | Deepseek AI |
|---|---:|---:|---:|
| `ASC_CAR` | `3.1205` | `9.0461` | `3.5040` |
| `ASC_PT` | `2.7111` | `2.3823` | `2.7004` |
| `B_COST` | `-0.6135` | `-0.0729` | `-0.1972` |
| `B_TIME` | `-0.3391` | `0.1600` | `0.1892` |

The respondent-characteristic MNL sharpens this interpretation. In the human benchmark, adding the fifteen retained respondent characteristics improves the final log likelihood from `-500.457` to `-391.212`, which is a plausible gain in explanatory power. In the qwen AI line, the same specification drives the final log likelihood essentially to zero. In the deepseek AI line, the same extension improves fit from `-385.932` to `-336.042`, which is substantial but not pathological. Therefore, deepseek is clearly less over-deterministic than qwen in the MNL setting, but it still does not recover the human preference structure.

## Hybrid choice model comparison

The hybrid choice model reveals a different and, in some sense, more demanding diagnostic. On the human side, the `torch 500` estimates are stable across the qwen and deepseek archives, as expected. The human benchmark retains a strongly negative generic cost coefficient, negative travel-time coefficients for both `PT` and `CAR`, and negative waiting-time and distance coefficients. This is the reference behavioral structure.

For qwen, the hybrid choice model confirms the severe distortion already visible in the MNL. The AI mode shares are heavily concentrated on `CAR`, and the AI-side parameter estimates become extreme. The generic cost coefficient becomes positive at `0.9409`, `B_TIME_PT` falls to the lower boundary at `-20.0`, `B_TIME_CAR` becomes `-17.25`, and several latent-variable coefficients enter the choice equation with extremely large positive values. This is not a mild deviation from the human benchmark. It is a qualitatively different behavioral system.

For deepseek, the hybrid choice model yields a subtler but still important result. The aggregate shares are much closer to the human benchmark than in the qwen case, but the structural parameters remain non-human. The generic cost coefficient is negative, yet much weaker in magnitude than the human benchmark, at `-0.3922`. More importantly, both travel-time coefficients become positive, with `B_TIME_PT = 0.1728` and `B_TIME_CAR = 1.6473`. The non-work waiting-time coefficient also becomes positive, and the latent-variable effects entering the `PT` and `CAR` utilities are very large and negative. Hence, deepseek matches the human shares more closely, but it still fails to reproduce the human behavioral mechanism uncovered by the hybrid choice model.

| HCM choice-side coefficient (`torch 500`) | Human | Qwen AI | Deepseek AI |
|---|---:|---:|---:|
| `B_COST` | `-1.6664` | `0.9409` | `-0.3922` |
| `B_TIME_PT` | `-0.7989` | `-20.0000` | `0.1728` |
| `B_TIME_CAR` | `-5.2087` | `-17.2515` | `1.6473` |
| `B_WAIT_WORK` | `-2.5766` | `8.3362` | `-3.7065` |
| `B_WAIT_OTHER` | `-4.2349` | `4.0506` | `3.1308` |
| `B_DIST_WORK` | `-1.1098` | `-9.2419` | `-1.7642` |
| `B_DIST_OTHER` | `-1.2830` | `-2.3509` | `0.3630` |

## What differs across models

The contrast between the MNL and the hybrid choice model is itself informative. The MNL is mainly sensitive to the aggregate choice pattern and the basic cost-time trade-off. Under this lens, qwen appears strongly distorted, whereas deepseek appears closer to the human benchmark in aggregate shares but still problematic in the sign of the time coefficient. The hybrid choice model is more demanding because it must explain not only the final mode choice but also the retained psychometric indicators and the latent behavioral structure linking the indicators to the choice equation. Under this stricter lens, both AI models depart substantially from the human benchmark, although qwen fails more severely.

This pattern implies that aggregate realism and structural realism should be treated separately. Deepseek is better than qwen in matching the observed mode shares. However, once latent attitudes and measurement equations are introduced, the deepseek line still produces implausible time effects and unstable latent-variable loadings. Therefore, deepseek improves the surface distribution of behavior without yet recovering the deeper utility structure that the hybrid choice model is designed to identify.

## Overall conclusion

Across both model classes, the retained Optima experiments support three conclusions. First, the qwen AI data differ markedly from the human benchmark in both aggregate behavior and parameter structure. Second, the deepseek AI data are closer to the human benchmark at the level of aggregate choice shares, especially in the MNL setting. Third, neither AI model reproduces the human trade-off structure once the analysis turns from aggregate shares to structural coefficients, and this limitation is especially visible in the hybrid choice model.

Accordingly, the most defensible substantive conclusion is not that one AI model has solved behavioral replication, but that model comparisons depend strongly on what criterion is used. If the criterion is aggregate choice share, deepseek looks substantially better than qwen. If the criterion is recovery of the human utility structure, neither model is yet satisfactory, and the hybrid choice model makes this failure especially clear.
