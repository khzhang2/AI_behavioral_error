# GPT-5.4-nano、DeepSeek-chat 与 human benchmark 对比报告

本文整理当前代码库中两次已经完成的正式实验，并把它们与 human benchmark 做并列对比。目标不是重复解释整个项目，而是回答四个更具体的问题：第一，两次实验的参数与问卷结构是什么；第二，相关数据和结果文件在哪里；第三，`GPT-5.4-nano` 与 `DeepSeek-chat` 相对于 human 的行为偏差有何异同；第四，收敛后的 SALCM 说明了什么。

## 1. 对比对象与实验目录

本报告比较以下三个对象。

| 对象 | 角色 | 实验或数据目录 |
| --- | --- | --- |
| `GPT-5.4-nano` | AI respondent experiment | `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/` |
| `DeepSeek-chat` | AI respondent experiment | `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/` |
| `human benchmark` | 人类基准数据 | `data/Swissmetro/demographic_choice_psychometric/` |

其中 human benchmark 的核心输入文件是：

- `data/Swissmetro/demographic_choice_psychometric/human_cleaned_wide.csv`
- `data/Swissmetro/demographic_choice_psychometric/human_respondent_profiles.csv`

两次 AI 实验各自使用自己的归档配置文件：

- `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/experiment_config.json`
- `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/experiment_config.json`

## 2. 共同问卷结构与运行规模

两次实验的问卷结构相同，差异主要在模型、采样温度以及 `n_block_templates_per_model`。

共同的 `survey_design` 是：

| 字段 | 数值 | 含义 |
| --- | --- | --- |
| `n_attitudes` | `6` | 每个 run 的 attitude questions 数 |
| `n_core_tasks` | `6` | 每个 run 的核心 choice cards 数 |
| `n_paraphrase_twins` | `2` | 每个 run 的 paraphrase cards 数 |
| `n_label_mask_twins` | `2` | 每个 run 的 label-mask cards 数 |
| `n_order_twins` | `2` | 每个 run 的 order-randomization cards 数 |
| `n_monotonicity_tasks` | `2` | 每个 run 的 monotonicity cards 数 |
| `n_dominance_tasks` | `2` | 每个 run 的 dominance cards 数 |
| `total task cards` | `16` | `6 + 2 + 2 + 2 + 2 + 2` |

因此每个 run 的总问答请求数都是：

`1 grounding + 6 attitudes + 16 tasks = 23`

两次实验的总体规模如下。

| 模型 | `n_block_templates_per_model` | `n_repeats_per_template` | respondents / runs | 计划总请求数 |
| --- | --- | --- | --- | --- |
| `GPT-5.4-nano` | `100` | `4` | `400` | `400 × 23 = 9200` |
| `DeepSeek-chat` | `120` | `4` | `480` | `480 × 23 = 11040` |

## 3. 模型设置

两个模型的主要运行参数如下。

| 字段 | `GPT-5.4-nano` | `DeepSeek-chat` |
| --- | --- | --- |
| provider | `poe` | `deepseek` |
| model | `gpt-5.4-nano` | `deepseek-chat` |
| base_url | `https://api.poe.com/v1` | `https://api.deepseek.com` |
| thinking mode | `reasoning_effort = none` | `thinking_mode = non_thinking` |
| temperature | `0.1` | `1.3` |
| top_p | `0.95` | `0.95` |
| top_k | `20` | `20` |
| seed | `20260412` | `20260412` |
| timeout_sec | `240` | `240` |

## 4. 后分析脚本与输出位置

两次实验的 post-AI analysis 使用同一套脚本。顺序如下。

```bash
./.venv/bin/python scripts/estimate_optima_intervention_metrics.py
./.venv/bin/python scripts/estimate_optima_panel_mnl.py --dataset human
./.venv/bin/python scripts/estimate_optima_panel_mnl.py --dataset ai_pooled
./.venv/bin/python scripts/estimate_optima_salcm.py
./.venv/bin/python scripts/summarize_optima_intervention_regime.py
```

每个实验目录中最关键的结果文件包括：

| 文件 | 作用 |
| --- | --- |
| `outputs/ai_collection_summary.json` | AI collection 的完成情况与基础质量统计 |
| `intervention_metrics_summary.json` | exact-repeat 随机性与干预效应总结 |
| `mnl/ai/ai_panel_mnl_estimates.csv` | AI pooled MNL 系数 |
| `mnl/ai/ai_panel_mnl_summary.json` | AI pooled MNL 汇总 |
| `mnl/human/human_baseline_mnl_estimates.csv` | human baseline MNL 系数 |
| `mnl/human/human_baseline_mnl_summary.json` | human baseline MNL 汇总 |
| `salcm/ai/ai_salcm_estimates.csv` | SALCM 参数 |
| `salcm/ai/ai_salcm_summary.json` | SALCM 汇总 |
| `salcm/ai/ai_salcm_regime_summaries.csv` | 各 latent regimes 的解释性统计 |
| `experiment_summary.md` | 中文实验摘要 |

## 5. 关键行为指标对比

下面把两个模型和 human benchmark 最重要的行为指标放在一起。

| 指标 | `GPT-5.4-nano` | `DeepSeek-chat` | human benchmark |
| --- | --- | --- | --- |
| completed respondents | `400 / 400` | `480 / 480` | `708` |
| valid task rate | `1.0000` | `0.9999` | — |
| exact-repeat flip rate | `0.0730` | `0.0639` | — |
| response entropy | `0.0806` | `0.0710` | — |
| label flip rate | `0.0388` | `0.0000` | — |
| order flip rate | `0.1313` | `0.0219` | — |
| monotonicity compliance | `0.9550` | `0.9906` | — |
| dominance violation | `0.0625` | `0.0083` | — |
| choice share: `CAR` | `0.8667` | `0.8743` | `0.6257` |
| choice share: `PT` | `0.1239` | `0.0591` | `0.3136` |
| choice share: `SLOW_MODES` | `0.0094` | `0.0665` | `0.0607` |
| share gap TV vs human | `0.2410` | `0.2544` | `0.0000` |

从这个表可以直接读出三点。

第一，两者都不是“随机乱答”的模型。它们的 exact-repeat flip rate 都不高，而且都没有表现出超出随机性基线的平均干预效应。

第二，`DeepSeek-chat` 在 consistency 和 rule fidelity 上明显强于 `GPT-5.4-nano`。它的 order flip 更低，dominance violation 更低，monotonicity compliance 更高。

第三，`DeepSeek-chat` 并没有因此更接近 human。两者都显著高估 `CAR`。`GPT-5.4-nano` 几乎把 `SLOW_MODES` 压没，而 `DeepSeek-chat` 的 `SLOW_MODES` 更接近 human，但它把 `PT` 压得更狠。

## 6. MNL 系数对比

为了理解“为什么 choice share 会偏”，最直接的方法是比较 MNL 系数。

| 参数 | human | `GPT-5.4-nano` | `DeepSeek-chat` |
| --- | --- | --- | --- |
| `ASC_PT` | `-0.1341` | `4.3931` | `1.1391` |
| `ASC_CAR` | `0.1293` | `5.5798` | `2.9574` |
| `B_COST` | `-0.1588` | `-0.0421` | `-0.0735` |
| `B_TIME_PT` | `-0.0071` | `-0.0231` | `-0.0268` |
| `B_TIME_CAR` | `-0.0271` | `-0.0264` | `-0.0353` |
| `B_WAIT` | `-0.0408` | `0.0044` | `-0.0442` |
| `B_DIST` | `-0.2626` | `-0.0277` | `-0.0679` |

这张表说明，两者与 human 的差异主要来自两类来源。

第一类是 mode-specific default preference，也就是 mode constants 明显过大。尤其 `ASC_CAR`，两种 AI 都远高于 human。

第二类是对成本和距离惩罚的削弱。两种 AI 的 `B_COST` 和 `B_DIST` 绝对值都明显小于 human。`GPT-5.4-nano` 这一点更极端，因此它更容易把 `CAR` 作为默认选项。

`DeepSeek-chat` 在 `B_WAIT` 上反而更接近 human，而 `GPT-5.4-nano` 的 `B_WAIT` 甚至变成轻微正值，这进一步说明 GPT 这边的效用结构更失真。

## 7. 修改后的 SALCM 设定

两边最终用于比较的 SALCM 都不是最初的 `3 × 2` 设定，而是收敛优先的简化版本。原因是原始规格在部分实验上未能稳定收敛。

最终采用的共同设定是：

| 字段 | 数值 |
| --- | --- |
| `n_preference_classes` | `2` |
| `n_scale_classes` | `2` |
| `maxiter` | `3000` |
| `maxfun` | `300000` |
| `ftol` | `1e-8` |
| `membership_covariates` | `semantic_arm`, `prompt_family_naturalistic`, `ScaledIncome`, `CAR_AVAILABLE`, `block_complexity_mean` |

两边在这个规格下都收敛了。

## 8. SALCM 提示了什么

### 8.1 `GPT-5.4-nano`

`GPT-5.4-nano` 的 SALCM 结果在：

- `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/salcm/ai/ai_salcm_summary.json`
- `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/salcm/ai/ai_salcm_regime_summaries.csv`

posterior masses 为：

- `C1_S1 = 0.3521`
- `C1_S2 = 0.0446`
- `C2_S1 = 0.5939`
- `C2_S2 = 0.0093`

这里的主要结构是两个 preference classes。

- `Class 1` 更像“高常数、强 car default、失真更重”的 class。它的 `ASC_CAR` 极大，`B_COST` 甚至转成正值，整体更像一个强行偏车的 distorted regime。
- `Class 2` 相对更接近 human trade-off 的符号方向，但仍然明显 car-biased。它不是“像人”，只是“比 Class 1 相对更像人”。

还要注意：文件中的 `regime_label` 是启发式命名，不能完全按字面读。例如 `human_like_tradeoff` 只能理解成“在 AI 内部相对更像 human 的那一类”，不能理解成“真的接近 human”。

### 8.2 `DeepSeek-chat`

`DeepSeek-chat` 的 SALCM 结果在：

- `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/salcm/ai/ai_salcm_summary.json`
- `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/salcm/ai/ai_salcm_regime_summaries.csv`

posterior masses 为：

- `C1_S1 = 0.3355`
- `C1_S2 = 0.0319`
- `C2_S1 = 0.5120`
- `C2_S2 = 0.1206`

同样，这里也是两个 preference classes 主导。

- `Class 1` 的符号结构更接近 human，且 trade-off fidelity 很强；但它依然高度 car-biased，`ASC_CAR` 很大。
- `Class 2` 更偏向一个稳定的 car-dominant class。它不是“更乱”，而是“更一致地偏”。

这里文件中 `label_sensitive` 这个启发式标签尤其要谨慎，因为这次 DeepSeek 的 `label_flip_rate` 基本是 `0`。因此该标签不能按字面理解成“真的对 label 敏感”，更稳妥的解释是“另一种更失真的偏好类”。

## 9. GPT 与 DeepSeek 相对 human 的共同点与差异

### 9.1 共同点

两者的共同点很明确。

第一，它们的主要问题都不是内部随机性。两者的 exact-repeat flip rate 都不高，而且干预效应平均上都没有超出随机性基线。

第二，它们相对 human 的主要偏差都更像稳定的偏好结构失真，而不是不一致性本身。它们都把 `CAR` 选得过多，把 `PT` 选得过少。

第三，两者都体现出 mode constants 偏大、成本和距离惩罚偏弱的结构性特征。因此它们都更像在执行一种“默认偏车”的内部规则。

### 9.2 差异

差异主要体现在“偏差的形状”和“一致性的强弱”上。

`DeepSeek-chat` 更稳定、更像一个规则一致的决策器。它几乎不表现出 label sensitivity，order sensitivity 也很低，dominance 和 monotonicity 检查基本都能通过。因此，如果只看内部 consistency，DeepSeek 明显优于 GPT。

但 `DeepSeek-chat` 并不因此更像 human。它的总体 share gap 反而略大于 GPT。它更接近 human 的地方是 `SLOW_MODES` 的份额没有被彻底压扁；它更不像 human 的地方是 `PT` 被压得更厉害。

`GPT-5.4-nano` 则更像一个“偏车且更容易受次生因素影响”的模型。它对 order 更敏感，rule fidelity 更弱，且几乎消灭了 `SLOW_MODES`。所以 GPT 的偏差不是单纯的 `CAR > PT`，而是更全面地把非车方式压缩掉了。

## 10. 能否找到 AI 与 human 行为不同的原因

如果这里的“原因”指的是行为机制层面的解释，那么可以找到比较明确的线索；如果指严格因果机制，则当前结果还不够。

当前最有力的机制解释是：

1. 两种 AI 都存在稳定的 `CAR` 默认偏好。
2. 两种 AI 对成本和距离的厌恶程度都弱于 human。
3. `GPT-5.4-nano` 还叠加了更强的顺序敏感性和更差的规则一致性。
4. `DeepSeek-chat` 则更像稳定地执行一个偏车规则，而不是被提示脆弱性拖偏。

因此，更准确的结论不是“AI 会乱答，所以不像人”，而是：

`AI 更像是在稳定地执行一种不同于 human 的行为规则。`

对于 `GPT-5.4-nano`，这条规则表现为：

- 更强的顺序敏感性
- 更弱的 trade-off fidelity
- 几乎把 `SLOW_MODES` 挤出选择集

对于 `DeepSeek-chat`，这条规则表现为：

- 更高的一致性
- 更低的 rule violations
- 但仍稳定地把 `PT` 压得远低于 human

## 11. 重新运行与复查时去哪里找东西

如果以后要复查这两次实验，最有用的路径如下。

### `GPT-5.4-nano`

- 实验目录：`experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/`
- collection 摘要：`outputs/ai_collection_summary.json`
- intervention 指标：`intervention_metrics_summary.json`
- AI MNL：`mnl/ai/ai_panel_mnl_estimates.csv`、`mnl/ai/ai_panel_mnl_summary.json`
- SALCM：`salcm/ai/ai_salcm_estimates.csv`、`salcm/ai/ai_salcm_summary.json`、`salcm/ai/ai_salcm_regime_summaries.csv`
- 中文摘要：`experiment_summary.md`

### `DeepSeek-chat`

- 实验目录：`experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/`
- collection 摘要：`outputs/ai_collection_summary.json`
- intervention 指标：`intervention_metrics_summary.json`
- AI MNL：`mnl/ai/ai_panel_mnl_estimates.csv`、`mnl/ai/ai_panel_mnl_summary.json`
- SALCM：`salcm/ai/ai_salcm_estimates.csv`、`salcm/ai/ai_salcm_summary.json`、`salcm/ai/ai_salcm_regime_summaries.csv`
- 中文摘要：`experiment_summary.md`

### human benchmark

- human source：`data/Swissmetro/demographic_choice_psychometric/`
- 核心文件：
  - `human_cleaned_wide.csv`
  - `human_respondent_profiles.csv`

### 重新做 post-AI analysis

当前脚本默认读取 repo-root 的 `experiment_config.json`。因此如果要严格复现某个归档实验，应先把对应实验目录下的 `experiment_config.json` 临时拷回 repo-root，或者手动把 repo-root 的配置改成与该归档一致，然后再运行：

```bash
./.venv/bin/python scripts/estimate_optima_intervention_metrics.py
./.venv/bin/python scripts/estimate_optima_panel_mnl.py --dataset human
./.venv/bin/python scripts/estimate_optima_panel_mnl.py --dataset ai_pooled
./.venv/bin/python scripts/estimate_optima_salcm.py
./.venv/bin/python scripts/summarize_optima_intervention_regime.py
```

## 12. 最简短结论

`GPT-5.4-nano` 和 `DeepSeek-chat` 都与 human benchmark 有显著差异，但差异的结构不同。两者都稳定地高估 `CAR`，这说明主要问题不是随机性，而是偏好结构失真；其中 `DeepSeek-chat` 更一致、更守规则，但并不更像人，而是更一致地偏离 human，尤其更强地压低 `PT`。`GPT-5.4-nano` 则略微更接近 human 的 aggregate share，但内部更不稳、对顺序更敏感，也更容易违反 trade-off diagnostics。 
