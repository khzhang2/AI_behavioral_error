# Torch 实验结果摘要

## 运行摘要

- synthetic respondents：`1445`
- choice tasks per respondent：`6`
- 总 choices：`8670`
- valid choice rate：`1.0000`
- draws：`250`
- device：`cuda`
- runtime_seconds：`399.82`

## 选择分布

- Private Car：`5335`
- E-Scooter Sharing：`1340`
- Bike Sharing：`1323`
- Walking：`642`
- Public Transport：`18`
- Car Sharing：`12`

## Torch mixed logit 拟合

- final loglikelihood：`-2640.950`
- init loglikelihood：`-4991.795`
- null loglikelihood：`-12019.172`
- rho_square_vs_init：`0.4709`
- rho_square_vs_null：`0.7803`

## Torch vs Biogeme(32 draws)

- 可比较参数数：`67`
- 符号一致数：`59`
- sign match rate：`0.8806`

## Torch vs Human

- 可比较参数数：`67`
- 符号一致数：`57`
- sign match rate：`0.8507`
