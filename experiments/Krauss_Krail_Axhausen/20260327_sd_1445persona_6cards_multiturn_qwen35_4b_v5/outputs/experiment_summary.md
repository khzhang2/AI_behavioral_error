# 实验结果摘要

## 运行摘要

- synthetic respondents：`1445`
- choice tasks per respondent：`6`
- 总 choices：`8670`
- valid choice rate：`1.0000`
- 平均每题调用时长：`0.35` 秒
- 平均每位 respondent 的 6 题总时长：`2.08` 秒
- Sobol draws：`5000`

## 选择分布

- Private Car：`6035`
- Bike Sharing：`1439`
- Walking：`819`
- E-Scooter Sharing：`377`

## mixed logit 拟合

- final loglikelihood：`-4680.730`
- rho_square：`0.6106`
- 参数数：`36`
- 5% 显著参数数：`21`

## AI 与人类对比

- 可比较参数数：`36`
- 符号一致数：`25`
- sign match rate：`0.6944`

## 5% 显著参数

- `ASC_WALK_REL_CAR`：estimate=`-10.3140`，se=`3.5302`，z=`-2.922`，p=`0.003482`
- `AGE_ES_REL_CAR`：estimate=`-0.0766`，se=`0.0097`，z=`-7.878`，p=`3.331e-15`
- `AGE_BS_REL_CAR`：estimate=`-0.0253`，se=`0.0040`，z=`-6.330`，p=`2.446e-10`
- `AGE_WALK_REL_CAR`：estimate=`0.1432`，se=`0.0539`，z=`2.657`，p=`0.007892`
- `BIKEACC_ES_REL_CAR`：estimate=`-0.2083`，se=`0.0948`，z=`-2.199`，p=`0.0279`
- `BIKEACC_BS_REL_CAR`：estimate=`0.2141`，se=`0.0423`，z=`5.058`，p=`4.229e-07`
- `CARACC_ES_REL_CAR`：estimate=`-0.2792`，se=`0.1347`，z=`-2.073`，p=`0.03813`
- `CARACC_BS_REL_CAR`：estimate=`-0.9604`，se=`0.0649`，z=`-14.803`，p=`0`
- `CARACC_WALK_REL_CAR`：estimate=`-1.5836`，se=`0.5395`，z=`-2.935`，p=`0.003332`
- `MAAS_ES_REL_CAR`：estimate=`0.9119`，se=`0.3954`，z=`2.306`，p=`0.02108`
- `SIGMA_ES`：estimate=`2.6542`，se=`0.2061`，z=`12.876`，p=`0`
- `SIGMA_BS`：estimate=`1.3754`，se=`0.0697`，z=`19.731`，p=`0`
- `SIGMA_WALK`：estimate=`-3.6412`，se=`1.4617`，z=`-2.491`，p=`0.01274`
- `B_TIME_BS`：estimate=`-0.6679`，se=`0.2171`，z=`-3.076`，p=`0.002097`
- `B_TIME_WALK`：estimate=`-0.6949`，se=`0.1868`，z=`-3.720`，p=`0.0001991`
- `B_ACCESS_CAR`：estimate=`1.0620`，se=`0.4294`，z=`2.473`，p=`0.0134`
- `B_EGRESS_CAR`：estimate=`-0.8482`，se=`0.2768`，z=`-3.064`，p=`0.002184`
- `B_COST_LOGMEAN`：estimate=`-10.9394`，se=`2.3476`，z=`-4.660`，p=`3.164e-06`
- `SIGMA_COST`：estimate=`-9.5984`，se=`1.6621`，z=`-5.775`，p=`7.71e-09`
- `B_SCHEME_SD`：estimate=`8.6718`，se=`2.4126`，z=`3.594`，p=`0.0003251`
- `B_PEDELEC`：estimate=`-7.6431`，se=`2.1747`，z=`-3.515`，p=`0.0004405`
