# Torch 实验结果摘要

## 运行摘要

- synthetic respondents：`1445`
- choice tasks per respondent：`6`
- 总 choices：`8670`
- valid choice rate：`1.0000`
- draws：`32`
- device：`cuda`
- runtime_seconds：`12.38`

## 选择分布

- Private Car：`5335`
- E-Scooter Sharing：`1340`
- Bike Sharing：`1323`
- Walking：`642`
- Public Transport：`18`
- Car Sharing：`12`

## Torch mixed logit 拟合

- final loglikelihood：`-2670.147`
- init loglikelihood：`-5353.964`
- null loglikelihood：`-12019.172`
- rho_square_vs_init：`0.5013`
- rho_square_vs_null：`0.7778`

## Torch vs Biogeme(32 draws)

- 可比较参数数：`67`
- 符号一致数：`57`
- sign match rate：`0.8507`

## Torch vs Human

- 可比较参数数：`67`
- 符号一致数：`55`
- sign match rate：`0.8209`
