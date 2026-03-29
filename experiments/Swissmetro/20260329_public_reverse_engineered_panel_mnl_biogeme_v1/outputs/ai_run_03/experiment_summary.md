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

- TRAIN：`937`
- SWISSMETRO：`4114`
- CAR：`1717`

## Biogeme MNL 拟合

- final loglikelihood：`-5746.913`
- null loglikelihood：`-6982.909`
- rho_square：`0.1770`
- 参数数：`4`
- 线程数：`27`
- 5% 显著参数数：`4`

## AI 与人类对比

- 可比较参数数：`4`
- 符号一致数：`4`
- sign match rate：`1.0000`
- human time-cost ratio：`1.1791`
- AI time-cost ratio：`12.1766`
- ratio difference：`10.9975`

## 5% 显著参数

- `ASC_CAR`：estimate=`-0.5518`，se=`0.0360`，z=`-15.307`，p=`0`
- `ASC_TRAIN`：estimate=`-1.1896`，se=`0.0465`，z=`-25.588`，p=`0`
- `B_COST`：estimate=`-0.0332`，se=`0.0040`，z=`-8.387`，p=`0`
- `B_TIME`：estimate=`-0.4045`，se=`0.0381`，z=`-10.623`，p=`0`

## 复现实验标签

- `replication_standard = public_reverse_engineering`
- `design_provenance = public data reverse engineering`
- `estimation_backend = biogeme`
- `questionnaire_style = multi_turn_one_task_at_a_time`
