# 实验结果摘要

## 运行摘要

- synthetic respondents：`752`
- choice tasks per respondent：`9`
- 总 choices：`6768`
- valid choice rate：`1.0000`
- grounding parse rate：`0.0000`
- 平均每题调用时长：`0.28` 秒
- 平均每位 respondent 的 9 题总时长：`2.51` 秒

## 选择分布

- TRAIN：`951`
- SWISSMETRO：`4230`
- CAR：`1587`

## Biogeme MNL 拟合

- final loglikelihood：`-5648.532`
- null loglikelihood：`-6931.820`
- rho_square：`0.1851`
- 参数数：`4`
- 线程数：`27`
- 5% 显著参数数：`4`

## AI 与人类对比

- 可比较参数数：`4`
- 符号一致数：`4`
- sign match rate：`1.0000`
- human time-cost ratio：`1.1791`
- AI time-cost ratio：`10.4149`
- ratio difference：`9.2359`

## 5% 显著参数

- `ASC_CAR`：estimate=`-0.6407`，se=`0.0361`，z=`-17.762`，p=`0`
- `ASC_TRAIN`：estimate=`-1.2208`，se=`0.0456`，z=`-26.778`，p=`0`
- `B_COST`：estimate=`-0.0371`，se=`0.0045`，z=`-8.257`，p=`2.22e-16`
- `B_TIME`：estimate=`-0.3866`，se=`0.0371`，z=`-10.412`，p=`0`

## 复现实验标签

- `replication_standard = public_reverse_engineering`
- `design_provenance = public data reverse engineering`
- `estimation_backend = biogeme`
- `questionnaire_style = multi_turn_one_task_at_a_time`
