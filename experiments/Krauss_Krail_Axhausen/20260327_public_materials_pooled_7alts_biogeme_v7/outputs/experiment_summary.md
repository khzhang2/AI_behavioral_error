# 实验结果摘要

## 运行摘要

- synthetic respondents：`1445`
- choice tasks per respondent：`6`
- 总 choices：`8670`
- valid choice rate：`1.0000`
- 平均每题调用时长：`0.53` 秒
- 平均每位 respondent 的 6 题总时长：`3.21` 秒
- Biogeme draws：`8`

## 选择分布

- Private Car：`5335`
- E-Scooter Sharing：`1340`
- Bike Sharing：`1323`
- Walking：`642`
- Public Transport：`18`
- Car Sharing：`12`

## Biogeme mixed logit 拟合

- final loglikelihood：`-2923.274`
- rho_square：`0.5473`
- 参数数：`67`
- 5% 显著参数数：`27`

## AI 与人类对比

- 可比较参数数：`67`
- 符号一致数：`46`
- sign match rate：`0.6866`

## 5% 显著参数

- `B_ACCESS_OWNED`：estimate=`-0.1743`，se=`0.0477`，z=`-3.652`，p=`0.0002602`
- `B_AGE_BS`：estimate=`-0.0607`，se=`0.0132`，z=`-4.594`，p=`4.34e-06`
- `B_AGE_ES`：estimate=`-0.0878`，se=`0.0152`，z=`-5.765`，p=`8.143e-09`
- `B_AGE_PT`：estimate=`0.0840`，se=`0.0251`，z=`3.350`，p=`0.0008069`
- `B_AGE_WALK`：estimate=`0.0536`，se=`0.0188`，z=`2.849`，p=`0.004392`
- `B_AVAILABILITY`：estimate=`0.0362`，se=`0.0037`，z=`9.665`，p=`0`
- `B_CARACC_BS`：estimate=`-0.7588`，se=`0.1481`，z=`-5.125`，p=`2.981e-07`
- `B_CARACC_ES`：estimate=`-0.7603`，se=`0.1795`，z=`-4.236`，p=`2.276e-05`
- `B_CARACC_PT`：estimate=`-3.1638`，se=`1.2793`，z=`-2.473`，p=`0.0134`
- `B_CARACC_WALK`：estimate=`-1.3184`，se=`0.2546`，z=`-5.177`，p=`2.252e-07`
- `B_MAAS_BS`：estimate=`-1.2967`，se=`0.5016`，z=`-2.585`，p=`0.009737`
- `B_MAAS_WALK`：estimate=`-2.6763`，se=`0.7402`，z=`-3.615`，p=`0.0002998`
- `B_PTPASS_PT`：estimate=`-2.1422`，se=`0.6745`，z=`-3.176`，p=`0.001494`
- `B_RANGE`：estimate=`0.1162`，se=`0.0276`，z=`4.205`，p=`2.616e-05`
- `B_TIME_BS`：estimate=`-0.2636`，se=`0.0133`，z=`-19.753`，p=`0`
- `B_TIME_CAR`：estimate=`-0.1881`，se=`0.0380`，z=`-4.947`，p=`7.536e-07`
- `B_TIME_CS`：estimate=`-1.2149`，se=`0.5293`，z=`-2.295`，p=`0.02173`
- `B_TIME_ES`：estimate=`-0.1719`，se=`0.0209`，z=`-8.222`，p=`2.22e-16`
- `B_TIME_PT`：estimate=`-0.3269`，se=`0.0378`，z=`-8.649`，p=`0`
- `B_TIME_RP`：estimate=`-0.6428`，se=`0.1387`，z=`-4.635`，p=`3.568e-06`
- `B_TIME_WALK`：estimate=`-0.2910`，se=`0.0237`，z=`-12.298`，p=`0`
- `PHI_POOL`：estimate=`2.1149`，se=`0.3426`，z=`6.174`，p=`6.666e-10`
- `SIGMA_CAR`：estimate=`3.2199`，se=`0.2734`，z=`11.776`，p=`0`
- `SIGMA_COST`：estimate=`2.9450`，se=`1.4466`，z=`2.036`，p=`0.04177`
- `SIGMA_ES`：estimate=`-1.9979`，se=`0.3207`，z=`-6.230`，p=`4.676e-10`
- `SIGMA_PT`：estimate=`0.8130`，se=`0.3737`，z=`2.176`，p=`0.02958`
- `SIGMA_WALK`：estimate=`-3.6559`，se=`0.5732`，z=`-6.379`，p=`1.787e-10`

## 复现实验标签

- `replication_standard = public_materials_high_fidelity`
- `design_provenance = public_exact attribute levels + inferred_from_public block combinations`
- `estimation_backend = biogeme`
