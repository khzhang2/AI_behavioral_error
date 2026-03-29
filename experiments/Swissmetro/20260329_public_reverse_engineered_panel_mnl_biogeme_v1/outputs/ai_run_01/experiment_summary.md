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

- TRAIN：`949`
- SWISSMETRO：`4035`
- CAR：`1784`

## Biogeme MNL 拟合

- final loglikelihood：`-5814.093`
- null loglikelihood：`-6990.207`
- rho_square：`0.1683`
- 参数数：`4`
- 线程数：`27`
- 5% 显著参数数：`4`

## AI 与人类对比

- 可比较参数数：`4`
- 符号一致数：`4`
- sign match rate：`1.0000`
- human time-cost ratio：`1.1791`
- AI time-cost ratio：`12.9354`
- ratio difference：`11.7563`

## 5% 显著参数

- `ASC_CAR`：estimate=`-0.4479`，se=`0.0361`，z=`-12.398`，p=`0`
- `ASC_TRAIN`：estimate=`-1.1197`，se=`0.0462`，z=`-24.214`，p=`0`
- `B_COST`：estimate=`-0.0339`，se=`0.0044`，z=`-7.652`，p=`1.976e-14`
- `B_TIME`：estimate=`-0.4379`，se=`0.0374`，z=`-11.705`，p=`0`

## 复现实验标签

- `replication_standard = public_reverse_engineering`
- `design_provenance = public data reverse engineering`
- `estimation_backend = biogeme`
- `questionnaire_style = multi_turn_one_task_at_a_time`
