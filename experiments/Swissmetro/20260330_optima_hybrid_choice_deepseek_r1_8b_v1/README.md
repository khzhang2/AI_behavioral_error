# Optima Hybrid Choice Model Experiment Report for deepseek-r1:8b

This archive directory records the retained Optima hybrid choice experiment that uses `deepseek-r1:8b` through the local `Ollama` interface. In the current repository organization, development is no longer carried out inside `experiments/`; instead, the maintained code lives in the root `scripts` directory and the maintained data package lives in `data/Swissmetro/demographic_choice_psychometric`. The present folder should therefore be interpreted as the archival record of one completed trial.

The experimental design follows the same reduced official-style two-latent-variable specification that was used in the retained `qwen3.5:9b` archive. The artificial-intelligence (AI) survey reused the same compact grounding, the same psychometric and mode-choice questionnaire structure, and the same respondent count. The only deliberate change is the model used for AI response generation. The resulting questionnaire data were stored outside the archive, in `data/Swissmetro/demographic_choice_psychometric/ai_collection_deepseek_r1_8b`, so that the data package remains independent from the archival results directory.

## Data quality and usable AI sample

The deepseek questionnaire collection completed all `708` intended respondents. However, the model did not return fully valid psychometric and choice answers for every respondent. The mean valid-indicator rate is approximately `0.8044`, and the mean valid-choice rate is approximately `0.8037`. Consequently, the final hybrid choice model estimation used the full `708` human observations but only `568` AI observations that contained valid values for the six retained indicators and the final mode choice.

After this validity filter, the aggregate AI mode shares are not far from the human benchmark. The human benchmark contains approximately `31.36%` public transport (`PT`), `62.57%` car (`CAR`), and `6.07%` slow modes (`SLOW_MODES`). The deepseek AI sample contains approximately `34.51%` `PT`, `63.03%` `CAR`, and `2.46%` `SLOW_MODES`. Thus, the aggregate shares appear superficially close, although the slow-mode share is notably smaller in the AI sample.

## Estimation outputs

The archived outputs are stored in `outputs/human_torch_32`, `outputs/ai_torch_32`, `outputs/human_torch_500`, and `outputs/ai_torch_500`. An initial human-side `Biogeme` run with `32` draws was retained in `outputs/human_biogeme_32` as a benchmark reference, but the final reported hybrid choice estimates in this deepseek archive are based on the `torch` estimator, following the later decision to avoid slow `Biogeme` draw-based estimation for the hybrid choice model.

The main log-likelihood values are as follows. For the human sample, the `torch` estimator produced a final log likelihood of `-6959.312` with `32` draws and `-6957.891` with `500` draws. For the AI sample, the corresponding values are `-776.165` with `32` draws and `-743.233` with `500` draws. The improvement from `32` to `500` draws is therefore limited on the human side and more visible on the AI side, but in both cases the broad parameter pattern is already largely established at `32` draws.

## Behavioral interpretation

The deepseek AI sample does not simply replicate the human Optima behavior, even though the aggregate mode shares appear relatively close. In the human `torch 500` hybrid choice model, the generic cost coefficient is strongly negative at approximately `-1.666`, and both travel-time coefficients are negative, with the car travel-time coefficient much more negative than the public-transport travel-time coefficient. Waiting time and distance also enter with negative coefficients. This pattern is behaviorally coherent and consistent with the interpretation that greater generalized disutility lowers choice probability.

The AI-side `torch 500` estimates are much less stable. The generic cost coefficient remains negative, but its magnitude is much smaller at approximately `-0.392`. More importantly, the public-transport and car travel-time coefficients both become positive, and the coefficient for non-work waiting time is also positive. The latent-variable effects are extremely large and negative for both `PT` and `CAR`, with one parameter reaching the upper numerical boundary of `-20.000`. These patterns suggest that the deepseek AI data do not reproduce the human trade-off structure that underlies the Optima hybrid choice benchmark. Instead, the AI responses appear to generate a much more fragile latent-variable structure, even though the aggregate choice shares are not grossly implausible.

## Overall assessment

This archived deepseek trial leads to a mixed conclusion. At the level of aggregate mode shares, the retained AI sample looks less distorted than the earlier `qwen3.5:9b` Optima results. However, once the full hybrid choice model is estimated, the coefficient structure remains clearly different from the human benchmark. The AI sample produces weaker cost sensitivity, implausible positive time effects, and highly unstable latent-variable loadings into the mode-choice equation. Accordingly, this archive should be read as evidence that matching aggregate shares is not sufficient to claim behavioral replication in a hybrid choice setting.
