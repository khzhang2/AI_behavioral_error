按当前代码，最重要的是区分 `persona`、`template`、`respondent`、`run` 这四层。

先给结论版：

1. `884 personas` 不是“已经生成好的 884 个 AI persona”，而是 `884` 条人类 profile 记录组成的经验支持集。来源见 [prepare_optima_intervention_regime_data.py](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/scripts/prepare_optima_intervention_regime_data.py)。
2. `n_block_templates_per_model = 100` 的意思是：从这 `884` 条 profile 支持集里，先随机打乱，再取 `100` 条 `selected_profiles`，每一条 profile 绑定一个 `block template`。
3. 所以在当前实现里，`100 templates` 基本就对应 `100` 个被抽中的 persona-profile 实例。
4. 如果 `n_repeats_per_template = 4`，那么每个 template 会展开成 `4` 个 `respondent_id`，也就是 `4` 次完整问卷执行。
5. 因此总 `respondents = 100 × 4 = 400`。
6. 在当前代码语境里，`respondent` 和 `run` 基本是一一对应的。

最容易混淆的是这一点：

- `persona` 更接近“这个 AI 受访者是谁”
- `respondent_id` 更接近“这次完整问卷作答实例是谁”

所以，同一个 `persona` 可以对应多个 `respondent_id`，只要你做了 repeat。

可以用你现在的参数举一个完整例子。

假设：

- `n_block_templates_per_model = 100`
- `n_repeats_per_template = 4`
- 支持集里有 `884` 条 human profiles

那么当前代码做的是：

1. 读取 `884` 条 human profile rows  
这些 rows 构成经验支持集，不是属性全枚举空间。

2. 随机打乱这 `884` 条 rows  
不是按原文件顺序取前 `100` 个。  
是先打乱，再取前 `100` 个。  
所以不是“截断前 100 个人”，而是“随机抽出的 100 个完整 profiles”。

3. 每个被抽中的 profile 绑定一个 `block template`  
例如会得到：

- `DST0001`
- `DST0002`
- ...
- `DST0100`

这里每个 template 都带着：
- 一个固定 persona
- 一个固定 prompt arm
- 一个固定 prompt family
- 一套固定的 core + twin/probe tasks

4. 每个 template 再重复 `4` 次  
所以 `DST0001` 会展开成：

- `DS0001_R1`
- `DS0001_R2`
- `DS0001_R3`
- `DS0001_R4`

这四个 `respondent_id`：
- persona 一样
- template 一样
- prompt 条件一样
- task cards 一样

不同的只是“这是第几次重复执行”。

所以这里就能看到：

- 一个 `template` 对应一个 persona-template 单元
- 一个 `template` 经过 `4` 次 repeats，变成 `4` 个 respondents
- 一个 `respondent` 就是一整次完整问卷执行

如果把当前结构写成公式，就是：

- 支持集大小：`884`
- 抽中的 persona-template 单元数：`100`
- 每个单元重复：`4`
- 总 respondents：`100 × 4 = 400`

然后问答层级再往下分。

当前代码里：

- `raw_interactions.jsonl` 的一行，不一定是一个 `task`
- 更准确地说，是一次模型交互

它有三种 stage：

1. `grounding`
2. `attitude`
3. `task`

所以：

- `jsonl` 一行 = 一次问答交互
- 其中 `task` 行才是你说的 choice task
- `attitude` 行是态度题
- `grounding` 行是开场确认 persona

因此：

- 一组 `tasks` 组成不了完整 `run`
- 一个完整 `run` 是：
  - `1` 次 grounding
  - `6` 次 attitude
  - `16` 次 task
- 合起来总共 `23` 行 interactions

所以你这句可以稍微改准确一点：

- `jsonl` 一行 = 一次 interaction，不一定是 task
- 一组 attitude + tasks + grounding = 一个 run
- 在当前代码里，一个 run 基本对应一个 respondent

这个对应关系还可以看成两张表：

1. [block_assignments.csv](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_smoke_v1/block_assignments.csv)  
一行 = 一个 `respondent_id` = 一个 planned run

2. [panel_tasks.csv](/Users/kaihangzhang/Downloads/GitHub/Research%20codes%20repo/AI_behavioral_error/experiments/Swissmetro/20260412_optima_intervention_regime_deepseek_chat_smoke_v1/panel_tasks.csv)  
一行 = 一个 `task`

所以：

- `block_assignments.csv` 是 run-level
- `panel_tasks.csv` 是 task-level

最后回答你最关键的那个问题：

`respondent` 能不能对标现实问卷中“每次分发问卷的一个受访者”？

答案是：

- `能，对标的是“一次完整问卷作答实例”`
- `但不能简单对标成“一个独立的人”`

原因是现实中一个受访者通常只答一次问卷；而你这里为了测随机性，会让同一个 persona-template 单元重复答 `4` 次。  
所以：

- `DS0001_R1` 到 `DS0001_R4` 像是“同一个合成受访者背景下的四次独立作答实例”
- 它们是四个 `respondents`
- 但背后其实共享同一个 persona

如果只用一句话概括：

`在当前代码里，persona 是“谁在答”，template 是“答哪套问卷”，respondent/run 是“这套问卷被完整答了一次”。`

## 问卷语义影响
你说的“前后对掉、改写”等操作，不发生在 repeat 之间，而发生在一个 template 内部

这是最关键的点。

这些操作不是说：
- 第 1 次 run 用原题
- 第 2 次 run 再把顺序调一下

不是这样。

真正的结构是：

一个 template 本身就包含：
- `6` 个 core tasks
- `2` 个 paraphrase twins
- `2` 个 label-mask twins
- `2` 个 order-randomization twins
- `2` 个 monotonicity tasks
- `2` 个 dominance tasks

也就是总共 `16` 个 tasks。

所以“前后对掉”等操作，是同一份问卷里的某几道题的变体，不是不同 repeat 之间的变化。

按当前代码，大致对应是：

- `task 1-6`: core tasks
- `task 7-8`: paraphrase twins
- `task 9-10`: label-mask twins
- `task 11-12`: order-randomization twins
- `task 13-14`: monotonicity tasks
- `task 15-16`: dominance tasks

更具体地说：

- `paraphrase`：把某个 core task 的措辞改写，但底层内容不变
- `label-mask`：把 mode label 显示方式切换
- `order-randomization`：把选项卡里 A/B/C 的展示顺序改掉
- `monotonicity`：把某个选项变差，看是否还保持一致
- `dominance`：故意造一个明显被支配的选项，看会不会选它

所以你可以这样理解：

- `repeat` 解决的是：同一套问卷反复答，会不会自己变
- `twin/probe` 解决的是：同一底层任务换个说法/顺序/标签后，会不会变

最后一句最直白的总结：

- `persona` = “这个 AI 受访者是谁”
- `template` = “这个 persona 要答哪一整套问卷”
- `respondent/run` = “这整套问卷被完整答了一次”
- `jsonl` 一行 = 一次 interaction，不一定是 task
- 一个完整 `run/respondent` = `1 grounding + 6 attitudes + 16 tasks = 23` 行 interactions

## 具体的问卷 manipulation 设计
1. “某个”基本就是从 core tasks 里按固定位置选出来的，不是每一类都再单独随机抽

当前代码不是“对每种 manipulation 再随机挑一个 core task”，而是先抽出 `n_core_tasks` 个 core tasks，然后按固定位置切片去派生不同 probe。

以你现在的默认设计为例：

- `n_core_tasks = 6`
- `n_paraphrase_twins = 2`
- `n_label_mask_twins = 2`
- `n_order_twins = 2`
- `n_monotonicity_tasks = 2`
- `n_dominance_tasks = 2`

当前代码里的对应关系大致是：

- paraphrase：拿 core 的前 `2` 题
- label-mask：拿接下来的 `2` 题
- order-randomization：再拿接下来的 `2` 题
- monotonicity：再从 core 的前 `2` 题派生
- dominance：再从后面的 `2` 题派生

也就是说：

- 先随机抽 `6` 个 core scenarios
- 然后对这 `6` 个 core scenarios 按位置分配不同 manipulation
- manipulation 本身不是再次随机从 6 个里乱抽

2. `label-mask` 不是换掉所有问题，只换那 `n_label_mask_twins` 个对应的问题

按你现在的配置：

- 只有 `2` 个 label-mask twin tasks

它们只是复制对应的两个 core tasks，然后把 `semantic_labels` 取反。  
所以不是整份问卷都改 label，而是只有那两道 twin tasks 改。

3. `order-randomization` 也不是换掉所有问题，只换那 `n_order_twins` 个对应的问题

按当前配置：

- 只有 `2` 个 order twins

这两道 twin tasks 会在原 core task 的基础上：
- 保持底层属性值不变
- 但把 `option_order` 改成另一个顺序
- 再重新映射成 `display_A_alt / display_B_alt / display_C_alt`

所以不是所有题都换顺序，只是那 `2` 道 twin tasks 换顺序。

4. `dominance` 也不是换掉所有问题，只换那 `n_dominance_tasks` 个对应的问题

按当前配置：

- 只有 `2` 个 dominance tasks

它们是从指定的 core tasks 派生出来的。代码会把其中一个选项故意改成明显更差，构造一个被支配选项。  
所以也不是全问卷都做 dominance test，只是那几道专门的 diagnostic tasks 做。

一句话总结：

- 这些 manipulation 都不是“全问卷统一改”
- 它们都是“只对 template 中的少数指定 core tasks 派生出 twin/probe versions”

如果按你现在的默认参数来理解，一份 `16` 题的 task 问卷结构就是：

- `6` 道 core tasks
- `2` 道 paraphrase tasks
- `2` 道 label-mask tasks
- `2` 道 order-randomization tasks
- `2` 道 monotonicity tasks
- `2` 道 dominance tasks

所以每种 diagnostic 都只作用于其中一小部分题，不是全局替换。
