# QuantTrio/gemma-4-31B-it-AWQ

这个文档由 `20260417_optima_intervention_regime_gemma4_31b_awq_v1` 下的参数对照表自动生成。 当前后端是 `openai_compatible`。 内部模型键是 `gemma4_31b_awq`。

文档读取 `atasoy_2011_replication/parameter_comparison.csv` 与 `hcm/parameter_comparison.csv`，其中 `gap_ai_minus_human` 定义为 AI 参数减 human 参数。

Atasoy base logit 当前差值最大的参数是 `beta_student`，AI 相对 human 更低 `3.2114`。

Exact HCM 当前差值最大的参数是 `alpha_Mobil16`，AI 相对 human 更低 `0.6684`。

## Atasoy Base Logit 最大差值

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| beta_student | 3.2073 | -0.0040 | -3.2114 |
| ASCSM | -0.4700 | -3.1767 | -2.7067 |
| beta_language | 1.0925 | -0.1131 | -1.2056 |
| ASCPMM | -0.4134 | 0.5975 | 1.0109 |
| beta_work | -0.5824 | 0.2158 | 0.7982 |
| beta_ncars | 1.0010 | 0.2964 | -0.7046 |

## Atasoy Base Logit 全部参数

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.4134 | 0.5975 | 1.0109 |
| ASCSM | -0.4700 | -3.1767 | -2.7067 |
| beta_cost | -0.0592 | -0.1113 | -0.0521 |
| beta_distance | -0.2273 | -0.0327 | 0.1946 |
| beta_language | 1.0925 | -0.1131 | -1.2056 |
| beta_nbikes | 0.3469 | 0.2107 | -0.1361 |
| beta_ncars | 1.0010 | 0.2964 | -0.7046 |
| beta_nchildren | 0.1535 | 0.3097 | 0.1561 |
| beta_student | 3.2073 | -0.0040 | -3.2114 |
| beta_time_pmm | -0.0299 | -0.0087 | 0.0213 |
| beta_time_pt | -0.0121 | -0.0217 | -0.0097 |
| beta_urban | 0.2862 | -0.2028 | -0.4890 |
| beta_work | -0.5824 | 0.2158 | 0.7982 |

## Exact HCM 最大差值

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| alpha_Mobil16 | 0.3734 | -0.2950 | -0.6684 |
| alpha_Envir02 | -0.3234 | 0.2923 | 0.6157 |
| beta_nbikes | 0.3850 | -0.2087 | -0.5937 |
| theta_basel_zurich | -0.2560 | -0.5326 | -0.2766 |
| beta_work | -0.5830 | -0.3219 | 0.2611 |
| ASCSM | -0.7720 | -1.0318 | -0.2598 |

## Exact HCM 分块参数

### utility

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.5990 | -0.3446 | 0.2544 |
| ASCSM | -0.7720 | -1.0318 | -0.2598 |
| beta_Acar | -0.5740 | -0.5347 | 0.0393 |
| beta_Aenv | 0.3930 | 0.4214 | 0.0284 |
| beta_cost | -0.0559 | -0.1488 | -0.0929 |
| beta_distance | -0.2240 | -0.0659 | 0.1581 |
| beta_language | 1.0600 | 0.9763 | -0.0837 |
| beta_nbikes | 0.3850 | -0.2087 | -0.5937 |
| beta_ncars | 0.9700 | 1.1421 | 0.1721 |
| beta_nchildren | 0.2150 | 0.2694 | 0.0544 |
| beta_student | 3.2600 | 3.1641 | -0.0959 |
| beta_time_pmm | -0.0294 | -0.0040 | 0.0254 |
| beta_time_pt | -0.0119 | -0.0256 | -0.0137 |
| beta_urban | 0.2830 | 0.2331 | -0.0499 |
| beta_work | -0.5830 | -0.3219 | 0.2611 |

### attitude

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| Acar | 3.0200 | 2.9293 | -0.0907 |
| Aenv | 3.2300 | 3.3681 | 0.1381 |
| theta_age | 0.0044 | 0.0062 | 0.0018 |
| theta_basel_zurich | -0.2560 | -0.5326 | -0.2766 |
| theta_bern | -0.3610 | -0.6160 | -0.2550 |
| theta_east | -0.2280 | -0.0525 | 0.1755 |
| theta_educ | 0.2350 | 0.1777 | -0.0573 |
| theta_graubunden | -0.3030 | -0.2976 | 0.0054 |
| theta_nbikes | 0.0845 | 0.0687 | -0.0158 |
| theta_ncars | 0.1040 | 0.2072 | 0.1032 |
| theta_valais | -0.2230 | -0.0673 | 0.1557 |

### measurement

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| alpha_Envir01 | -0.3991 | -0.4089 | -0.0098 |
| alpha_Envir02 | -0.3234 | 0.2923 | 0.6157 |
| alpha_Envir06 | 0.3717 | 0.3883 | 0.0167 |
| alpha_Mobil11 | 0.3937 | 0.4665 | 0.0728 |
| alpha_Mobil16 | 0.3734 | -0.2950 | -0.6684 |
| loading_Envir01 | 0.8381 | 0.8004 | -0.0377 |
| loading_Envir02 | 1.0114 | 0.8556 | -0.1558 |
| loading_Envir06 | 1.0918 | 1.0753 | -0.0165 |
| loading_Mobil11 | 1.1553 | 1.1353 | -0.0199 |
| loading_Mobil16 | 1.0605 | 1.2451 | 0.1846 |
| sigma_Envir01 | 1.3086 | 1.1655 | -0.1432 |
| sigma_Envir02 | 1.1304 | 1.0218 | -0.1086 |
| sigma_Envir05 | 1.0385 | 0.9019 | -0.1366 |
| sigma_Envir06 | 0.7837 | 0.7230 | -0.0608 |
| sigma_Mobil10 | 1.2063 | 1.0077 | -0.1986 |
| sigma_Mobil11 | 1.0977 | 1.1402 | 0.0425 |
| sigma_Mobil16 | 1.1172 | 1.0542 | -0.0631 |
