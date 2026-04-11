# Experimental Method for Identifying Artificial-Intelligence Response Regimes

This document describes how to design and execute an experiment that identifies artificial-intelligence (AI) response error in the form of latent response regimes rather than pre-labeled error categories. The objective is methodological rather than purely empirical. Instead of beginning from a hand-written taxonomy of mistakes and then testing whether those mistakes exist, the experiment assumes that observed AI responses arise from a mixture of latent decision regimes. These regimes are then inferred from repeated choice behavior, controlled task perturbations, and a small set of structured response indicators.

## 1. Conceptual framing

The core idea is to replace the language of “error types” with the more neutral language of latent response regimes. A latent response regime is a hidden decision state that governs how an AI respondent converts survey inputs into choices. Different regimes may correspond to different behavioral patterns. One regime may resemble a relatively human-like compensatory trade-off process. Another regime may overweight labels or semantic cues. A third regime may be less stable and more affected by scale or consistency differences. The important point is that these patterns are not imposed ex ante as semantic labels. They are inferred from the data through a latent structure.

For this reason, the first model should be a Scale-Adjusted Latent Class Choice Model (SALCM), rather than a fully developed Integrated Choice and Latent Variable (ICLV) model. The SALCM is the most defensible first step because it separates two sources of heterogeneity that are easily confounded. The first source is preference heterogeneity, represented by class-specific utility coefficients. The second source is consistency heterogeneity, represented by scale classes. This distinction is essential because some apparent “errors” are really differences in choice consistency rather than differences in substantive utility trade-offs.

The current repository implements precisely this first-step logic through the Optima latent response regime line.

## 2. Experimental unit

The experimental unit is not a single question. It is one complete AI respondent block. A block is defined by a specific model, a specific prompt arm, one sampled persona, and one full sequence of survey tasks completed within a single growing conversation. This definition is critical because latent regime identification requires repeated observations on the same block. If each block answered only one task, stable regime differences and one-off noise would be observationally confounded.

In the current implementation, a block is indexed by:

- model identity
- sampled respondent profile
- prompt arm
- full sequence of attitude and choice tasks

The repository currently supports pooled multi-model blocks, so that `qwen3.5:9b` and `deepseek-r1:8b` can both enter the same latent-regime estimation.

## 3. Survey design

Each AI respondent block should contain two layers of data. The first layer is a short psychometric layer. The second layer is a repeated choice layer.

### 3.1 Psychometric layer

The psychometric layer retains the six Optima attitude indicators already used in the existing Optima experiments:

- `Envir01`
- `Mobil05`
- `LifSty07`
- `Envir05`
- `Mobil12`
- `LifSty01`

At this stage, these indicators are not yet used as formal measurement equations in an ICLV model. However, they are collected so that the experiment remains ICLV-ready.

### 3.2 Choice layer

The repeated choice layer contains sixteen tasks per respondent block. These tasks are deliberately heterogeneous.

- Eight tasks are core empirical tasks sampled from an Optima scenario bank derived from the retained human benchmark data.
- Two tasks are label-mask twins.
- Two tasks are order-randomization twins.
- Two tasks are monotonicity diagnostics.
- Two tasks are dominance diagnostics.

This structure is the heart of the method. The core tasks identify the main utility structure. The diagnostic tasks create exogenous perturbations that do not merely add noise, but instead reveal whether the AI block is sensitive to labels, ordering, dominance relations, or monotonicity violations.

## 4. Response schema

Each choice task should return a structured response, not a free-form explanation. A structured response is essential because it gives the later measurement-model extension a small set of reliable, machine-readable indicators rather than a long and difficult-to-validate chain of thought.

For each choice task, the current design collects:

- chosen option
- confidence on a `1` to `5` scale
- top two decisive attributes
- whether the respondent believes a clearly dominated option is present

This response format is more useful than narrative self-explanation, because it directly supports the construction of behavioral diagnostics such as label-flip rates, monotonicity-compliance rates, and dominance-violation rates.

## 5. Identification logic

The experiment identifies latent response regimes through three kinds of variation.

### 5.1 Within-block repetition

Repeated tasks within one respondent block allow the researcher to observe whether the same latent mechanism persists across several decisions. Without repeated tasks, there is no clean way to separate stable behavioral structure from idiosyncratic one-task variation.

### 5.2 Controlled exogenous perturbation

The experiment uses controlled task perturbations that alter presentation while preserving the underlying economic structure as much as possible.

- Label-mask twins test semantic attraction to mode names.
- Order-randomization twins test sensitivity to ordering.
- Monotonicity diagnostics test whether the AI respects a basic worsening of an alternative.
- Dominance diagnostics test whether the AI avoids clearly dominated options.

Because these perturbations are experimentally assigned, they can later enter class-membership equations as observed causes of latent regime membership.

### 5.3 Structured behavioral indicators

The structured fields returned after each task provide direct indicators of regime behavior. These indicators are not yet formal measurement equations in the current V1 implementation, but they make the design extendable to ICLV. They also allow immediate post-collection diagnostics.

## 6. Model ladder

The correct estimation sequence is not to jump directly into a complex hybrid model. It is to proceed in layers.

### 6.1 Human benchmark multinomial logit model

The first step is to estimate a human benchmark multinomial logit model on the retained human Optima data. This benchmark is not part of the latent-regime model itself. Instead, it defines the normative reference point against which AI regimes are compared.

In the current repository, this role is performed by:

- `scripts/estimate_optima_panel_mnl.py` with `--dataset human`

### 6.2 Pooled AI panel multinomial logit model

The second step is to estimate a pooled AI panel multinomial logit model on the repeated-task AI data. This model is the non-latent baseline for the new survey design. It tells us how the pooled repeated-task AI data behave before latent regimes are introduced.

In the current repository, this role is performed by:

- `scripts/estimate_optima_panel_mnl.py` with `--dataset ai_pooled`

### 6.3 Scale-Adjusted Latent Class Choice Model

The third step is the pooled AI SALCM. In the current design:

- the unit of classification is the respondent block
- there are three preference classes
- there are two scale classes
- one scale class is normalized to one

Within each preference class, the utility system contains:

- `ASC_PT`
- `ASC_CAR`
- `B_COST`
- `B_TIME_PT`
- `B_TIME_CAR`
- `B_WAIT`
- `B_DIST`

The scale component allows the same underlying utility pattern to be expressed with different degrees of decisiveness or consistency. This is the minimum structure needed to distinguish preference distortion from consistency distortion.

In the current repository, this role is performed by:

- `scripts/estimate_optima_salcm.py`

## 7. Post-estimation diagnostics

The experiment should not stop at coefficient estimation. The whole point of the regime framework is to interpret regimes behaviorally rather than merely statistically.

For each respondent block, the current design computes:

- dominance-violation rate
- monotonicity-compliance rate
- label-flip rate
- order-flip rate
- confidence mean
- top-attribute shares

For each class-scale combination, the current design computes a distortion summary relative to the human benchmark. That summary includes:

- coefficient sign mismatches
- normalized coefficient distance
- mode-share deviation
- average diagnostic-task violation rates

Finally, these regime-level distortions are combined with posterior regime membership probabilities to produce a respondent-level posterior distortion score.

This distortion score is descriptive in V1. It is not yet a formal trustworthiness threshold. However, it creates the bridge to a later trustworthy-artificial-intelligence formulation in which posterior regime probabilities can be mapped into use-case-specific risk.

## 8. Current repository implementation

The current latent-regime experiment line is organized around the following files.

- `scripts/prepare_optima_latent_regime_data.py`
- `scripts/optima_latent_regime_questionnaire.py`
- `scripts/run_optima_latent_regime_ai_collection.py`
- `scripts/estimate_optima_panel_mnl.py`
- `scripts/estimate_optima_salcm.py`
- `scripts/summarize_optima_latent_regime.py`
- `experiment_config.json`

The corresponding archive directory is:

- `experiments/Swissmetro/20260411_optima_salcm_cross_model_v1`

The corresponding data package is:

- `data/Swissmetro/latent_regime_optima_v1`

Within that data package, the experiment writes:

- `scenario_bank.csv`
- `respondent_profile_bank.csv`
- model-specific block assignments
- model-specific panel tasks
- one collection subdirectory per model

Within each model-specific collection subdirectory, the experiment writes:

- `persona_samples.csv`
- `parsed_attitudes.csv`
- `parsed_task_responses.csv`
- `ai_panel_long.csv`
- `ai_panel_block.csv`
- `raw_interactions.jsonl`
- `respondent_transcripts.json`
- `run_respondents.json`

## 9. Recommended workflow

The current practical workflow is as follows.

1. Prepare the latent-regime data package.  
   Use `scripts/prepare_optima_latent_regime_data.py`.

2. Run the repeated-task AI collection for each model.  
   Use `scripts/run_optima_latent_regime_ai_collection.py --model-key ...`.

3. Estimate the human benchmark multinomial logit model.  
   Use `scripts/estimate_optima_panel_mnl.py --dataset human`.

4. Estimate the pooled AI panel multinomial logit model.  
   Use `scripts/estimate_optima_panel_mnl.py --dataset ai_pooled`.

5. Estimate the pooled AI SALCM.  
   Use `scripts/estimate_optima_salcm.py`.

6. Write the archive summary.  
   Use `scripts/summarize_optima_latent_regime.py`.

This order should be preserved because each step depends conceptually on the previous one. The human benchmark defines the normative comparison point, the pooled AI panel multinomial logit model defines the non-latent baseline, and the SALCM is meaningful only relative to those two references.

## 10. How this identifies artificial-intelligence response error

This design identifies AI response error by inference rather than by prior labeling. If one class exhibits coefficients close to the human benchmark, low dominance violations, low label-flip rates, and low mode-share deviation, that class can be interpreted as relatively human-like. If another class exhibits large coefficient distortions, strong label sensitivity, or severe violations of monotonicity or dominance, that class can be interpreted as a more distorted regime. The important methodological point is that these interpretations arise after the model is estimated. They are not inserted into the experiment as ex ante semantic categories.

This is why the approach is methodologically stronger than a simple audit. The experiment does not begin by deciding what the error types are. It begins with a latent mixture hypothesis, supplies repeated tasks and experimental perturbations that make the latent structure identifiable, and then allows the recovered latent structure to determine how the regimes should be described.

## 11. Limitations of the current V1 design

The current V1 implementation should be understood as a first-stage latent-regime design, not as the final methodological model.

- It uses a SALCM rather than a full ICLV model.
- The psychometric items are collected but are not yet formal measurement equations.
- Dynamic regime switching across tasks is not yet modeled.
- The current trustworthiness output is descriptive rather than threshold-based.

For this reason, the current V1 design is best interpreted as the identification backbone. It establishes the repeated-task architecture, the perturbation-based measurement logic, and the block-level latent classification structure. Once these are stable, the natural V2 extension is an ICLV-SALCM in which the structured diagnostic indicators and psychometric variables become explicit measurement equations.
