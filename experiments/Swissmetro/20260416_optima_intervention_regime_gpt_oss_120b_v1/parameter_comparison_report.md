# openai/gpt-oss-120b

这个文档由 `20260416_optima_intervention_regime_gpt_oss_120b_v1` 下的参数对照表自动生成。 当前后端是 `openai_compatible`。 内部模型键是 `gpt_oss_120b`。

文档读取 `atasoy_2011_replication/parameter_comparison.csv` 与 `hcm/parameter_comparison.csv`，其中 `gap_ai_minus_human` 定义为 AI 参数减 human 参数。

Atasoy base logit 当前差值最大的参数是 `ASCSM`，AI 相对 human 更低 `2.5855`。

Exact HCM 当前差值最大的参数是 `beta_nbikes`，AI 相对 human 更低 `0.7817`。

## Atasoy Base Logit 最大差值

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCSM | -0.4700 | -3.0556 | -2.5855 |
| beta_student | 3.2073 | 1.5945 | -1.6128 |
| ASCPMM | -0.4134 | 0.6020 | 1.0154 |
| beta_work | -0.5824 | 0.4046 | 0.9871 |
| beta_language | 1.0925 | 0.3022 | -0.7904 |
| beta_ncars | 1.0010 | 0.3241 | -0.6769 |

## Atasoy Base Logit 全部参数

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.4134 | 0.6020 | 1.0154 |
| ASCSM | -0.4700 | -3.0556 | -2.5855 |
| beta_cost | -0.0592 | -0.0768 | -0.0177 |
| beta_distance | -0.2273 | -0.0439 | 0.1834 |
| beta_language | 1.0925 | 0.3022 | -0.7904 |
| beta_nbikes | 0.3469 | -0.2160 | -0.5629 |
| beta_ncars | 1.0010 | 0.3241 | -0.6769 |
| beta_nchildren | 0.1535 | 0.0866 | -0.0669 |
| beta_student | 3.2073 | 1.5945 | -1.6128 |
| beta_time_pmm | -0.0299 | -0.0175 | 0.0125 |
| beta_time_pt | -0.0121 | -0.0211 | -0.0091 |
| beta_urban | 0.2862 | 0.0833 | -0.2028 |
| beta_work | -0.5824 | 0.4046 | 0.9871 |

## Exact HCM 最大差值

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| beta_nbikes | 0.3850 | -0.3967 | -0.7817 |
| alpha_Envir02 | -0.3234 | 0.3327 | 0.6561 |
| alpha_Envir06 | 0.3717 | -0.1700 | -0.5417 |
| beta_work | -0.5830 | -0.2037 | 0.3793 |
| ASCPMM | -0.5990 | -0.3132 | 0.2858 |
| ASCSM | -0.7720 | -1.0514 | -0.2794 |

## Exact HCM 分块参数

### utility

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| ASCPMM | -0.5990 | -0.3132 | 0.2858 |
| ASCSM | -0.7720 | -1.0514 | -0.2794 |
| beta_Acar | -0.5740 | -0.5041 | 0.0699 |
| beta_Aenv | 0.3930 | 0.3295 | -0.0635 |
| beta_cost | -0.0559 | -0.0883 | -0.0324 |
| beta_distance | -0.2240 | -0.1405 | 0.0835 |
| beta_language | 1.0600 | 1.0064 | -0.0536 |
| beta_nbikes | 0.3850 | -0.3967 | -0.7817 |
| beta_ncars | 0.9700 | 0.9128 | -0.0572 |
| beta_nchildren | 0.2150 | 0.2243 | 0.0093 |
| beta_student | 3.2600 | 3.1895 | -0.0705 |
| beta_time_pmm | -0.0294 | -0.0173 | 0.0121 |
| beta_time_pt | -0.0119 | -0.0226 | -0.0107 |
| beta_urban | 0.2830 | 0.1484 | -0.1346 |
| beta_work | -0.5830 | -0.2037 | 0.3793 |

### attitude

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| Acar | 3.0200 | 2.9195 | -0.1005 |
| Aenv | 3.2300 | 2.9734 | -0.2566 |
| theta_age | 0.0044 | 0.0098 | 0.0054 |
| theta_basel_zurich | -0.2560 | -0.1583 | 0.0977 |
| theta_bern | -0.3610 | -0.2519 | 0.1091 |
| theta_east | -0.2280 | -0.0337 | 0.1943 |
| theta_educ | 0.2350 | 0.2484 | 0.0134 |
| theta_graubunden | -0.3030 | -0.1696 | 0.1334 |
| theta_nbikes | 0.0845 | 0.1030 | 0.0185 |
| theta_ncars | 0.1040 | 0.1067 | 0.0027 |
| theta_valais | -0.2230 | -0.3055 | -0.0825 |

### measurement

| parameter_name | human_estimate | ai_estimate | gap_ai_minus_human |
| --- | ---: | ---: | ---: |
| alpha_Envir01 | -0.3991 | -0.4642 | -0.0650 |
| alpha_Envir02 | -0.3234 | 0.3327 | 0.6561 |
| alpha_Envir06 | 0.3717 | -0.1700 | -0.5417 |
| alpha_Mobil11 | 0.3937 | 0.4223 | 0.0285 |
| alpha_Mobil16 | 0.3734 | 0.4601 | 0.0867 |
| loading_Envir01 | 0.8381 | 0.9213 | 0.0832 |
| loading_Envir02 | 1.0114 | 0.8630 | -0.1484 |
| loading_Envir06 | 1.0918 | 1.2731 | 0.1813 |
| loading_Mobil11 | 1.1553 | 1.1428 | -0.0124 |
| loading_Mobil16 | 1.0605 | 1.0605 | 0.0000 |
| sigma_Envir01 | 1.3086 | 1.2673 | -0.0414 |
| sigma_Envir02 | 1.1304 | 1.1022 | -0.0282 |
| sigma_Envir05 | 1.0385 | 1.0762 | 0.0377 |
| sigma_Envir06 | 0.7837 | 0.8640 | 0.0803 |
| sigma_Mobil10 | 1.2063 | 1.1030 | -0.1033 |
| sigma_Mobil11 | 1.0977 | 1.1054 | 0.0077 |
| sigma_Mobil16 | 1.1172 | 1.0665 | -0.0507 |
