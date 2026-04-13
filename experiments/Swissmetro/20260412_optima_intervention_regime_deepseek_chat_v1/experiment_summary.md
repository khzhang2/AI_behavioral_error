# 实验摘要：deepseek_chat

本次实验对应单模型归档 `deepseek_chat`。AI 问卷收集共完成 `480` / `480` 个 planned respondents，态度题有效率为 `1.0000`，任务题有效率为 `0.9999`。总体上，这次 smoke test 的主要特征是：模型内部非常稳定，语义、标签和顺序干预几乎不产生额外波动，但相对 human benchmark 仍表现出明显的出行方式偏移。

| 检验对象 | 这次试验怎么概括 | 主要数值 | 解释 |
| --- | --- | --- | --- |
| 同一系统的随机不稳定性 | 很低 | exact-repeat flip rate = `0.0639`；response entropy = `0.0710` | 完全相同输入下几乎不翻转，within-model randomness 很弱。 |
| 对语义等价改写是否稳健 | 很稳健 | paraphrase flip rate = `0.0000`；paraphrase gap = `0.0000`；paraphrase excess gap = `-0.0990` | 改写措辞后几乎没有系统变化，没有观察到超出随机性基线的 semantic fragility。 |
| 对标签或顺序是否过敏 | 很弱 | label flip rate = `0.0000`；order flip rate = `0.0219`；label excess gap = `-0.0703`；order excess gap = `-0.0667` | 这次 smoke 中 label 与 order 的额外干预效应都非常小，未见明显的 label-sensitive 或 order-sensitive pattern。 |
| 是否真的在做 trade-off | 很强 | monotonicity compliance = `0.9906`；dominance violation = `0.0083` | 基本完全遵守 monotonicity 与 dominance 检查，说明规则性 trade-off fidelity 很强。 |
| 是否只是总体像人 | 不像，仍有明显 distortion | AI shares: `CAR=0.8743, PT=0.0591, SLOW_MODES=0.0665`；human shares: `CAR=0.6257, PT=0.3136, SLOW_MODES=0.0607`；share gap TV = `0.2544` | 模型显著高估 `CAR`，显著低估 `PT`，说明它是稳定地偏离 human benchmark，而不是随机地像人。 |

注意事项：human baseline MNL 有数值优化警告；AI panel MNL 有数值优化警告。因此本次结果更适合做方向性判断，不宜过度解读精细参数。
