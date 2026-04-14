# GPT-5.4-nano、DeepSeek-chat 与 human benchmark 对比报告

本文整理当前代码库中两次已经完成的正式实验，并把它们与 human benchmark 做并列对比。目标不是重复解释整个项目，而是回答五个更具体的问题：第一，两次实验的参数与问卷结构是什么；第二，相关数据和结果文件在哪里；第三，`GPT-5.4-nano` 与 `DeepSeek-chat` 相对于 human 的行为偏差有何异同；第四，当前 HCM / 有限 ICLV 结果能告诉我们什么；第五，收敛后的 SALCM 说明了什么。

校正说明（2026-04-14）：在这份报告初稿之后，仓库修复了两个人类基准侧错误。第一，旧的人类 panel MNL worker sample 过滤应使用 `OccupStat in {1, 2}`，而不是 `TripPurpose != 3`。第二，原始 `TimePT` 已经包含 waiting time，因此不能再和 `WaitingTimePT` 做重复计入。现在这份报告的结构 baseline 已经统一改成 `atasoy_2011_replication`，也就是 Atasoy 2011 paper-style `base logit`。两次 AI 实验的归档回答仍然来自修复前的 PT 问卷文案，所以 AI 侧 legacy HCM / SALCM 结果在这里仍只适合做有限方向性对照。

## 1. 对比对象与实验目录

本报告比较以下三个对象。

| 对象 | 角色 | 实验或数据目录 |
| --- | --- | --- |
| `GPT-5.4-nano` | AI respondent experiment | `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/` |
| `DeepSeek-chat` | AI respondent experiment | `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/` |
| `human benchmark` | 人类基准数据 | `data/Swissmetro/demographic_choice_psychometric/` |

其中 human benchmark 的核心输入文件是：

- `data/Swissmetro/demographic_choice_psychometric/raw/optima.dat`
- `data/Swissmetro/demographic_choice_psychometric/human_cleaned_wide.csv`
- `data/Swissmetro/demographic_choice_psychometric/human_respondent_profiles.csv`

两次 AI 实验的配置文件：

- `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/experiment_config.json`
- `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/experiment_config.json`

两次 AI 实验报告：

- [../experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/experiment_summary.md](../experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/experiment_summary.md)
- [../experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/experiment_summary.md](../experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/experiment_summary.md)

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
./.venv/bin/python scripts/replicate_atasoy_2011_models.py
./.venv/bin/python scripts/estimate_atasoy_2011_ai_analysis.py --experiment-dirs 20260412_optima_intervention_regime_poe_gpt54_nano_v1 20260412_optima_intervention_regime_deepseek_chat_v1
./.venv/bin/python scripts/estimate_optima_salcm.py
./.venv/bin/python scripts/summarize_optima_intervention_regime.py
```

每个实验目录中最关键的结果文件包括：

| 文件 | 作用 |
| --- | --- |
| `outputs/ai_collection_summary.json` | AI collection 的完成情况与基础质量统计 |
| `intervention_metrics_summary.json` | exact-repeat 随机性与干预效应总结 |
| `atasoy_2011_replication/ai_atasoy_base_logit_estimates.csv` | AI side Atasoy 2011 base logit 系数 |
| `atasoy_2011_replication/ai_atasoy_base_logit_summary.json` | AI side Atasoy 2011 base logit 汇总 |
| `data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/base_logit/base_logit_estimates.csv` | human Atasoy 2011 base logit 系数 |
| `data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/base_logit/base_logit_summary.json` | human Atasoy 2011 base logit 汇总 |
| `data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/hcm/hcm_summary.json` | human Atasoy 2011 exact HCM 汇总 |
| `hcm/ai_atasoy_hcm_summary.json` | AI side Atasoy 2011 exact HCM 汇总（新主线实验） |
| `salcm/ai_salcm_estimates.csv` | SALCM 参数 |
| `salcm/ai_salcm_summary.json` | SALCM 汇总 |
| `salcm/ai_salcm_regime_summaries.csv` | 各 latent regimes 的解释性统计 |
| `experiment_summary.md` | 中文实验摘要 |

## 5. 关键行为指标对比

下面把两个模型和 human benchmark 最重要的行为指标放在一起。

| 指标 | `GPT-5.4-nano` | `DeepSeek-chat` | human benchmark |
| --- | --- | --- | --- |
| completed respondents | `400 / 400` | `480 / 480` | `896` |
| valid task rate | `1.0000` | `0.9999` | — |
| exact-repeat flip rate | `0.0730` | `0.0639` | — |
| response entropy | `0.0806` | `0.0710` | — |
| label flip rate | `0.0388` | `0.0000` | — |
| order flip rate | `0.1313` | `0.0219` | — |
| monotonicity compliance | `0.9550` | `0.9906` | — |
| dominance violation | `0.0625` | `0.0083` | — |
| base-model share: `PMM` | `0.8468` | `0.8818` | `0.6231` |
| base-model share: `PT` | `0.1437` | `0.0546` | `0.3209` |
| base-model share: `SM` | `0.0095` | `0.0636` | `0.0560` |
| base-model share gap TV vs human | `0.2236` | `0.2662` | `0.0000` |

从这个表可以直接读出三点。

第一，两者都不是“随机乱答”的模型。它们的 exact-repeat flip rate 都不高，而且都没有表现出超出随机性基线的平均干预效应。

第二，`DeepSeek-chat` 在 consistency 和 rule fidelity 上明显强于 `GPT-5.4-nano`。它的 order flip 更低，dominance violation 更低，monotonicity compliance 更高。

第三，即使改成 Atasoy 2011 的 base logit 作为统一结构 baseline，两者也都没有变成“接近 human”的模型。两者都显著高估 `PMM`。`GPT-5.4-nano` 几乎把 `SM` 压没，而 `DeepSeek-chat` 的 `SM` 更接近 human，但它把 `PT` 压得更狠。

这里还需要单独说明一点：这一行 human 数值来自 `data/Swissmetro/demographic_choice_psychometric/atasoy_2011_replication/`，对应的是 Atasoy 2011 paper-style `base logit` 在 `1906` 个 public Optima observations 上的复现结果。

## 6. Atasoy 2011 Base Logit 系数对比

为了理解“为什么 base-model share 会偏”，最直接的方法是比较同一套 Atasoy 2011 `base logit` 系数。

这一节现在是三边完全同口径的结构比较：human 列来自 public `optima.dat` 的 paper replication，AI 两列来自两个实验目录现有输出上的 Atasoy-style base logit 估计，不需要重新对 AI 发请求。

| 参数 | human | `GPT-5.4-nano` | `DeepSeek-chat` |
| --- | --- | --- | --- |
| `ASCPMM` | `-0.4134` | `0.5854` | `1.5112` |
| `ASCSM` | `-0.4700` | `-3.6773` | `-1.7185` |
| `beta_cost` | `-0.0592` | `-0.0616` | `-0.0789` |
| `beta_time_pmm` | `-0.0299` | `-0.0339` | `-0.0321` |
| `beta_time_pt` | `-0.0121` | `-0.0285` | `-0.0304` |
| `beta_distance` | `-0.2273` | `-0.0649` | `-0.0597` |
| `beta_ncars` | `1.0010` | `0.0196` | `0.2746` |
| `beta_nchildren` | `0.1535` | `0.1426` | `-0.1098` |
| `beta_language` | `1.0925` | `0.6555` | `0.6282` |
| `beta_work` | `-0.5824` | `0.3010` | `0.0396` |
| `beta_urban` | `0.2862` | `-0.2695` | `-0.2958` |
| `beta_student` | `3.2073` | `0.4543` | `1.1936` |
| `beta_nbikes` | `0.3469` | `-0.3733` | `0.2645` |

这张表说明，两种 AI 与 human 的差异主要来自三类来源。

第一类是 mode-specific default preference，也就是 `ASCPMM` 被大幅抬高，而 `ASCSM` 被大幅压低。两种 AI 都更容易把 private motorized modes 当成默认选项。

第二类是对 `distance` 的惩罚被显著削弱。human 的 `beta_distance` 大约是 `-0.227`，而两种 AI 都只有大约 `-0.06` 左右。这意味着 AI 对 slow modes 的距离负担明显不够敏感。

第三类是 socio-demographic slope 的方向开始偏离 human。最明显的是 `beta_work` 和 `beta_urban`：human 都是朝向 `PT` 的方向，但两种 AI 都反过来变成更偏 `PMM`。`GPT-5.4-nano` 还出现了 `beta_nbikes < 0`，而 `DeepSeek-chat` 则把 `beta_ncars` 压得几乎消失。

## 7. HCM / 有限 ICLV 结果能告诉我们什么

当前仓库的默认 HCM 主线已经切换到 Atasoy 2011 的 fixed-normalization exact HCM。旧的 Biogeme panel HCM 仍然保留为历史脚本，但不再是默认结构基准。

这一节保留的是 AI 侧历史归档 HCM 结果。当前默认 human 结构基准已经改成 `atasoy_2011_replication/base_logit/` 与 `atasoy_2011_replication/hcm/`，因此 experiment 目录中的旧 `hcm/human` 归档已经不再保留。

对应结果文件在：

- `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/hcm/ai_biogeme_hcm_estimates.csv`
- `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/hcm/ai_biogeme_hcm_summary.json`
- `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/hcm/ai_biogeme_hcm_estimates.csv`
- `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/hcm/ai_biogeme_hcm_summary.json`

先看 HCM 进入估计的样本规模：

| HCM 输入样本 | `GPT-5.4-nano` | `DeepSeek-chat` |
| --- | --- | --- |
| respondents | `400` | `479` |
| core tasks | `2400` | `2874` |
| 每个 respondent 的 core tasks | `6` | `6` |

这里有两个需要先定义的点。第一，`HCM input` 指的是进入 HCM 估计脚本的有效样本，而不是原始 collection 总数。第二，`core task` 指的是原始 stated choice 选项卡，不包含 `paraphrase`、`label-mask`、`order-randomization`、`monotonicity`、`dominance` 这些诊断题。

如果只看 HCM 输入样本中的 core-task choice shares，结果与前面的 Atasoy base-model 结论一致：

| 指标 | `GPT-5.4-nano` | `DeepSeek-chat` |
| --- | --- | --- |
| `PT` share | `0.1138` | `0.0484` |
| `CAR` share | `0.8804` | `0.8855` |
| `SLOW_MODES` share | `0.0058` | `0.0661` |

也就是说，即使用 HCM 只看 core tasks，行为模式仍然没变：`GPT-5.4-nano` 继续几乎压掉 `SLOW_MODES`，`DeepSeek-chat` 继续把 `PT` 压得更狠。

但当前这版 HCM 的核心限制非常明确：**AI 侧参数几乎完全停在初始化点**。两边 AI 结果里，真正非零的参数只有四个，而且都正好等于初始化值：

| 参数 | `GPT-5.4-nano` | `DeepSeek-chat` |
| --- | --- | --- |
| `SIGMA_CAR` | `0.5` | `0.5` |
| `SIGMA_ENV` | `0.5` | `0.5` |
| `DELTA_1` | `0.5` | `0.5` |
| `DELTA_2` | `0.5` | `0.5` |
| 其他 37 个参数 | `0.0` | `0.0` |

这意味着什么，需要说清楚。

第一，当前 HCM 脚本已经把数据链路跑通了。也就是说，问卷结构、六个 attitude indicators、人口统计变量和 core-task choice outcomes 的确足以支撑一个“可以运行”的有限 ICLV / HCM。

第二，当前这版 HCM **还没有产生可解释的结构估计**。因为当所有 utility parameters、latent-variable structural coefficients、measurement intercepts 和 loadings 都停在初始值时，就不能据此解释 latent variables 的经济含义，也不能据此解释 AI 与 human 为什么不同。

第三，因此当前 HCM 对三者对比的贡献，主要还停留在“可行性验证”和“样本筛选后行为模式复核”这两层，而不是“识别出新的潜变量机制”。更直白地说：这版 HCM 现在告诉我们，问卷的数据结构是够的，但数值优化和模型识别还不够好。

所以，如果问“当前 HCM 能不能解释 GPT 与 DeepSeek 为什么和 human 不一样”，答案是：

- 目前**不能靠 HCM 参数本身来解释**。
- 目前最多只能说：在 HCM 只保留 core tasks 之后，两种 AI 与 human 的偏差方向并没有消失。
- 真正对机制更有信息量的，当前仍然是前面的 Atasoy 2011 base logit 和后面的 SALCM。

因此，当前最稳妥的结论是：

`HCM / 有限 ICLV 在数据结构上是可行的，但在当前数值实现下仍处于“能跑通、未识别”的阶段；它暂时不能提供比 Atasoy 2011 base logit 和 SALCM 更强的行为机制解释。`

## 8. 修改后的 SALCM 设定

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

## 9. SALCM 提示了什么

### 9.1 `GPT-5.4-nano`

`GPT-5.4-nano` 的 SALCM 结果在：

- `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/salcm/ai_salcm_summary.json`
- `experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/salcm/ai_salcm_regime_summaries.csv`

posterior masses 为：

- `C1_S1 = 0.3521`
- `C1_S2 = 0.0446`
- `C2_S1 = 0.5939`
- `C2_S2 = 0.0093`

这里的主要结构是两个 preference classes。

- `Class 1` 更像“高常数、强 car default、失真更重”的 class。它的 `ASC_CAR` 极大，`B_COST` 甚至转成正值，整体更像一个强行偏车的 distorted regime。
- `Class 2` 相对更接近 human trade-off 的符号方向，但仍然明显 car-biased。它不是“像人”，只是“比 Class 1 相对更像人”。

还要注意：文件中的 `regime_label` 是启发式命名，不能完全按字面读。例如 `human_like_tradeoff` 只能理解成“在 AI 内部相对更像 human 的那一类”，不能理解成“真的接近 human”。

### 9.2 `DeepSeek-chat`

`DeepSeek-chat` 的 SALCM 结果在：

- `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/salcm/ai_salcm_summary.json`
- `experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/salcm/ai_salcm_regime_summaries.csv`

posterior masses 为：

- `C1_S1 = 0.3355`
- `C1_S2 = 0.0319`
- `C2_S1 = 0.5120`
- `C2_S2 = 0.1206`

同样，这里也是两个 preference classes 主导。

- `Class 1` 的符号结构更接近 human，且 trade-off fidelity 很强；但它依然高度 car-biased，`ASC_CAR` 很大。
- `Class 2` 更偏向一个稳定的 car-dominant class。它不是“更乱”，而是“更一致地偏”。

这里文件中 `label_sensitive` 这个启发式标签尤其要谨慎，因为这次 DeepSeek 的 `label_flip_rate` 基本是 `0`。因此该标签不能按字面理解成“真的对 label 敏感”，更稳妥的解释是“另一种更失真的偏好类”。

## 10. GPT 与 DeepSeek 相对 human 的共同点与差异

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

## 11. 能否找到 AI 与 human 行为不同的原因

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

## 12. 重新运行与复查时去哪里找东西

如果以后要复查这两次实验，最有用的路径如下。

### `GPT-5.4-nano`

- 实验目录：`experiments/Swissmetro/20260412_optima_intervention_regime_poe_gpt54_nano_v1/`
- collection 摘要：`outputs/ai_collection_summary.json`
- intervention 指标：`intervention_metrics_summary.json`
- AI base model：`atasoy_2011_replication/ai_atasoy_base_logit_estimates.csv`、`atasoy_2011_replication/ai_atasoy_base_logit_summary.json`
- HCM：`hcm/ai_biogeme_hcm_estimates.csv`、`hcm/ai_biogeme_hcm_summary.json`
- SALCM：`salcm/ai_salcm_estimates.csv`、`salcm/ai_salcm_summary.json`、`salcm/ai_salcm_regime_summaries.csv`
- 中文摘要：`experiment_summary.md`

### `DeepSeek-chat`

- 实验目录：`experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_v1/`
- collection 摘要：`outputs/ai_collection_summary.json`
- intervention 指标：`intervention_metrics_summary.json`
- AI base model：`atasoy_2011_replication/ai_atasoy_base_logit_estimates.csv`、`atasoy_2011_replication/ai_atasoy_base_logit_summary.json`
- HCM：`hcm/ai_biogeme_hcm_estimates.csv`、`hcm/ai_biogeme_hcm_summary.json`
- SALCM：`salcm/ai_salcm_estimates.csv`、`salcm/ai_salcm_summary.json`、`salcm/ai_salcm_regime_summaries.csv`
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
./.venv/bin/python scripts/replicate_atasoy_2011_models.py
./.venv/bin/python scripts/estimate_atasoy_2011_ai_analysis.py --experiment-dirs 20260412_optima_intervention_regime_poe_gpt54_nano_v1 20260412_optima_intervention_regime_deepseek_chat_v1
./.venv/bin/python scripts/estimate_optima_salcm.py
./.venv/bin/python scripts/summarize_optima_intervention_regime.py
```

## 12. 最简短结论

`GPT-5.4-nano` 和 `DeepSeek-chat` 都与 human benchmark 有显著差异，但差异的结构不同。两者都稳定地高估 `CAR`，这说明主要问题不是随机性，而是偏好结构失真；其中 `DeepSeek-chat` 更一致、更守规则，但并不更像人，而是更一致地偏离 human，尤其更强地压低 `PT`。`GPT-5.4-nano` 则略微更接近 human 的 aggregate share，但内部更不稳、对顺序更敏感，也更容易违反 trade-off diagnostics。 
