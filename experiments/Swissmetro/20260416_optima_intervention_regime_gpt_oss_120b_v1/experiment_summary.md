# 实验摘要：gpt_oss_120b

本次实验对应单模型归档 `gpt_oss_120b`。AI 问卷收集共完成 `480` / `480` 个 planned respondents，态度题有效率为 `1.0000`，任务题有效率为 `1.0000`。总体上，这次实验的主要特征是：模型内部非常稳定，但相对 human benchmark 仍表现出明显的出行方式偏移，而且部分干预与 trade-off 检查并不稳。

| 检验对象 | 这次试验怎么概括 | 主要数值 | 解释 |
| --- | --- | --- | --- |
| 同一系统的随机不稳定性 | 很低 | exact-repeat flip rate = `0.0338`；response entropy = `0.0370` | 完全相同输入下几乎不翻转，within-model randomness 很弱。 |
| 对语义等价改写是否稳健 | 很稳健 | paraphrase flip rate = `0.0115`；paraphrase gap = `0.0115`；paraphrase excess gap = `-0.0241` | 改写措辞后几乎没有系统变化，没有观察到超出随机性基线的 semantic fragility。 |
| 对标签或顺序是否过敏 | 很弱 | label flip rate = `0.0208`；order flip rate = `0.0552`；label excess gap = `-0.0139`；order excess gap = `0.0073` | 当前更明显的是 order sensitivity，而不是 label sensitivity。 |
| 是否真的在做 trade-off | 很强 | monotonicity compliance = `0.9885`；dominance violation = `0.0344` | 模型同时通过 monotonicity 与 dominance 检查，trade-off fidelity 很强。 |
| 是否只是总体像人 | 不像，仍有明显 distortion | AI base-model shares: `PMM=0.8965, PT=0.0985, SM=0.0051`；human base-model shares: `PMM=0.6231, PT=0.3209, SM=0.0560`；share gap TV = `0.2733` | 按 Atasoy 2011 base logit 的结构比较，模型当前最明显的是 `高估` `PMM`，同时 `低估` `PT`。 |

注意事项：human Atasoy base logit 有数值优化警告；AI Atasoy base logit 有数值优化警告；SALCM 未完全收敛。因此本次结果更适合做方向性判断，不宜过度解读精细参数。
