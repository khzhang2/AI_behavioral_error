# 实验摘要：poe_gpt54_nano

本次实验对应单模型归档 `poe_gpt54_nano`。AI 问卷收集共完成 `400 / 400` 个 planned respondents，态度题有效率为 `1.0000`，任务题有效率为 `1.0000`。总体上，这次实验的主要特征是：模型内部随机性不算高，但相对 human 的最大问题仍然是稳定的结构性偏差，而不是纯随机乱答。

校正说明（2026-04-14）：本归档不再把 `mnl/` 作为默认结构 baseline。当前默认 base model 已改为 `atasoy_2011_replication/`。对 `GPT-5.4-nano` 来说，AI-side Atasoy base logit 预测份额为 `PMM = 0.8468`、`PT = 0.1437`、`SM = 0.0095`，而 human paper replication 的对应值为 `0.6231 / 0.3209 / 0.0560`，总变差距离为 `0.2236`。

| 检验对象 | 这次试验怎么概括 | 主要数值 | 解释 |
| --- | --- | --- | --- |
| 同一系统的随机不稳定性 | `低到中等` | exact-repeat flip rate = `0.0730`；response entropy = `0.0806` | 完全相同输入下会有一些翻转，但幅度不算高。说明有随机性，但不是主要问题。 |
| 对语义等价改写是否稳健 | `较稳健` | paraphrase flip rate = `0.0313`；paraphrase gap = `0.0313`；paraphrase excess gap = `-0.0583` | 改写措辞后会有少量变化，但平均上没有超出重复随机性的基线。 |
| 对标签或顺序是否过敏 | `标签弱，顺序明显` | label flip rate = `0.0388`；label excess gap = `-0.0610`；order flip rate = `0.1313`；order excess gap = `0.0435` | 标签掩码影响不大，但顺序随机化影响明显更强。这个模型更像 `order-sensitive`。 |
| 是否真的在做 trade-off | `大体在做，但不完全稳定` | monotonicity compliance = `0.9550`；dominance violation = `0.0625` | 大多数时候遵守基本理性规则，但仍有一部分明显的 rule violations。 |
| 是否只是“总体像人” | `不像，失真明显` | AI Atasoy shares: `PMM=0.8468, PT=0.1437, SM=0.0095`；human Atasoy shares: `PMM=0.6231, PT=0.3209, SM=0.0560`；share gap TV = `0.2236` | 在统一的 Atasoy 2011 base logit 下，模型显著过度偏向 `PMM`，显著压低 `PT` 和 `SM`。 |

注意事项：human Atasoy base logit 有数值优化警告；AI Atasoy base logit 有数值优化警告；SALCM 未完全收敛。因此本次结果更适合做方向性判断，不宜过度解读精细参数。更具体的结构比较请看 `atasoy_2011_replication/ai_atasoy_analysis.md`。
