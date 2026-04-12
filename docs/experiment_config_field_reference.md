# `experiment_config` Field Reference

这份表按照当前 `scripts/` 下的代码整理。目标是回答两个实际问题：哪些字段真的会被脚本读取，以及这些字段应该放在 `experiment_config.json`、`experiment_config_base.json` 还是 `api_credentials.local.json`。

当前推荐原则是：

- `experiment_config.json`：放实验级调参字段。
- `experiment_config_base.json`：放稳定默认值与方法默认值。
- `api_credentials.local.json`：只放本机密钥相关字段。
- experiment-ready 工作流下：每个实验配置只保留一个 `llm_models` 条目，对应一个实验文件夹和一个 model。

## Experiment Folder Layout

在当前 experiment-ready 约定下，每个实验文件夹只对应一个 model，目录结构应理解为：

- experiment root：派生 AI 数据、参数估计、diagnostics、`experiment_summary.md`
- `outputs/`：只保留原始 AI 问答文件，如 `raw_interactions.jsonl`、`respondent_transcripts.json`、`run_respondents.json`、`ai_collection_summary.json`

| 字段路径 | 建议文件 | 类型 | 当前代码用途 | 含义 | 示例 |
|---|---|---:|---|---|---|
| `config_base_file` | `experiment_config.json` | `str` | `scripts/optima_common.py` | 指向基础配置文件 | `"experiment_config_base.json"` |
| `config_overrides` | `experiment_config.json` | `object` | `scripts/optima_common.py` | 对 base 配置的覆盖层；优先级高于 base | `{...}` |
| `config_overrides.experiment_name` | `experiment_config.json` | `str` | 多个脚本归档与输出命名 | 当前实验名 | `"20260412_optima_intervention_regime_v2"` |
| `config_overrides.paths` | `experiment_config.json` | `object` | `scripts/optima_common.py` | 当前实验路径设置 | `{...}` |
| `config_overrides.paths.data_dir` | `experiment_config.json` | `str` | 部分脚本与兼容层 | 当前实验数据目录；在 experiment-ready 工作流中不再是主输出目录 | `"data/Swissmetro/latent_regime_optima_v2"` |
| `config_overrides.paths.archive_dir` | `experiment_config.json` | `str` | 归档输出脚本 | 当前实验 archive 目录 | `"experiments/Swissmetro/20260412_optima_intervention_regime_v2"` |
| `paths.source_data_dir` | `experiment_config_base.json` | `str` | 数据准备脚本 | 人类基准与源数据目录 | `"data/Swissmetro/demographic_choice_psychometric"` |
| `master_seed` | `experiment_config.json` | `int` | 数据准备、bootstrap、抽样 | 主随机种子 | `20260412` |
| `active_llm_key` | `experiment_config.json` | `str` | `llm_config_for()` | 默认使用哪一个 `llm_models[].key` | `"qwen3.5_9b"` |
| `n_block_templates_per_model` | `experiment_config.json` | `int` | `prepare_optima_intervention_regime_data.py` | 每个模型生成多少个问卷块模板 | `236` |
| `n_repeats_per_template` | `experiment_config.json` | `int` | `prepare_optima_intervention_regime_data.py` | 每个 template 做多少次 exact repeat | `3` |
| `llm_models` | `experiment_config.json` | `list[object]` | 所有 collection / summarize / estimate 脚本 | 模型列表；experiment-ready 工作流要求这里只有一个条目 | `[{...}]` |
| `llm_models[].key` | `experiment_config.json` | `str` | `active_llm_key`、`--model-key`、输出目录命名 | 模型条目的内部唯一键名；CLI 应优先使用这个字段，而不是 `model` | `"poe_gpt54_nano"` |
| `llm_models[].respondent_prefix` | `experiment_config.json` | `str` | 数据准备脚本 | `respondent_id` / `block_template_id` 前缀 | `"PO"` |
| `llm_models[].provider` | `experiment_config.json` | `str` | collection 脚本 | 后端类型；决定走 `ollama`、`poe` 或 OpenAI-compatible 分支 | `"poe"` |
| `llm_models[].model` | `experiment_config.json` | `str` | collection 脚本 | 真正发送给后端的模型名 | `"gpt-5.4-nano"` |
| `llm_models[].base_url` | `experiment_config.json` | `str` | collection 脚本 | 后端根地址 | `"https://api.poe.com/v1"` |
| `llm_models[].credentials_file` | `experiment_config.json` | `str` | `optima_common.py` | 本地凭据文件路径 | `"api_credentials.local.json"` |
| `llm_models[].api_key` | `experiment_config.json` 或本地文件 | `str` | `resolve_llm_api_key()` | 直接写死的 API key；不推荐入库 | `""` |
| `llm_models[].api_key_env` | `experiment_config.json` 或本地文件 | `str` | `resolve_llm_api_key()` | 显式指定环境变量名 | `"POE_API_KEY"` |
| `llm_models[].format` | `experiment_config.json` | `str` 或 `object` | collection 脚本 | 输出格式控制；对 OpenAI-compatible 分支会转成 `response_format` | `"json"` |
| `llm_models[].think` | `experiment_config.json` | `bool` | `ollama` 分支 | 是否打开 `ollama` 的 `think` 参数 | `false` |
| `llm_models[].reasoning_effort` | `experiment_config.json` | `str` | Poe / OpenAI-compatible 分支 | 显式传给兼容 OpenAI 的 `reasoning_effort` 字段 | `"none"` |
| `llm_models[].temperature` | `experiment_config.json` | `float` | collection 脚本 | 温度参数 | `0.1` |
| `llm_models[].top_p` | `experiment_config.json` | `float` | collection 脚本 | nucleus sampling 参数 | `0.95` |
| `llm_models[].top_k` | `experiment_config.json` | `int` | `ollama` 分支 | `top_k` 参数 | `20` |
| `llm_models[].seed` | `experiment_config.json` | `int` | collection 脚本 | 请求级随机种子 | `20260412` |
| `llm_models[].grounding_num_predict` | `experiment_config.json` | `int` | collection 脚本 | grounding 阶段生成长度 | `96` |
| `llm_models[].attitude_num_predict` | `experiment_config.json` | `int` | collection 脚本 | attitude 阶段生成长度 | `32` |
| `llm_models[].task_num_predict` | `experiment_config.json` | `int` | collection 脚本 | task 阶段生成长度 | `96` |
| `llm_models[].timeout_sec` | `experiment_config.json` | `int` | collection 脚本 | HTTP 请求超时秒数 | `240` |
| `llm_models[].extra_body` | `experiment_config.json` | `object` | Poe / OpenAI-compatible 分支 | 厂商私有字段透传口 | `{"custom_flag": true}` |
| `llm_models[].response_decoder` | `experiment_config.json` | `object` | `optima_common.py` | 针对单个模型覆盖默认响应解析路径 | `{...}` |
| `llm_models[].response_decoder.response_text_path` | `experiment_config.json` | `str` | `decode_chat_response()` | 正式输出文本路径 | `"choices.0.message.content"` |
| `llm_models[].response_decoder.thinking_text_path` | `experiment_config.json` | `str` | `decode_chat_response()` | thinking / reasoning 文本路径 | `"choices.0.message.reasoning_content"` |
| `llm_models[].response_decoder.done_reason_path` | `experiment_config.json` | `str` | `decode_chat_response()` | finish reason 路径 | `"choices.0.finish_reason"` |
| `llm_models[].response_decoder.total_duration_path` | `experiment_config.json` | `str` | `decode_chat_response()` | duration 路径 | `"total_duration"` |
| `llm_models[].response_decoder.prompt_eval_count_path` | `experiment_config.json` | `str` | `decode_chat_response()` | prompt token 路径 | `"usage.prompt_tokens"` |
| `llm_models[].response_decoder.eval_count_path` | `experiment_config.json` | `str` | `decode_chat_response()` | completion token 路径 | `"usage.completion_tokens"` |
| `survey_design` | `experiment_config.json` | `object` | 数据准备、collection、估计 | 问卷结构参数 | `{...}` |
| `survey_design.n_attitudes` | `experiment_config.json` | `int` | intervention / latent collection 脚本 | 每次 run 采集多少道态度题 | `6` |
| `survey_design.n_core_tasks` | `experiment_config.json` | `int` | 数据准备脚本 | 核心 choice tasks 数 | `6` |
| `survey_design.n_paraphrase_twins` | `experiment_config.json` | `int` | 数据准备脚本 | paraphrase twin 数 | `2` |
| `survey_design.n_label_mask_twins` | `experiment_config.json` | `int` | 数据准备脚本 | label-mask twin 数 | `2` |
| `survey_design.n_order_twins` | `experiment_config.json` | `int` | 数据准备脚本 | order-randomization twin 数 | `2` |
| `survey_design.n_monotonicity_tasks` | `experiment_config.json` | `int` | 数据准备脚本 | monotonicity 诊断题数 | `2` |
| `survey_design.n_dominance_tasks` | `experiment_config.json` | `int` | 数据准备脚本 | dominance 诊断题数 | `2` |
| `survey_design.total_tasks` | `experiment_config.json` | `int` | collection / estimate 脚本 | 问卷总任务数；当前代码优先按组件计数自动合成，仅作为冗余对照 | `16` |
| `survey_design.prompt_families` | `experiment_config.json` | `list[str]` | 数据准备脚本 | 可采样的 prompt family 集合 | `["concise", "naturalistic"]` |
| `survey_design.prompt_arms` | `experiment_config.json` | `list[str]` | 数据准备脚本 | 可采样的 prompt arm 集合 | `["semantic_arm", "neutral_arm"]` |
| `survey_design.option_orders` | `experiment_config.json` | `list[str]` | 数据准备脚本 | 可选展示顺序集合 | `["PT|CAR|SLOW_MODES", "CAR|PT|SLOW_MODES"]` |
| `survey_design.top_attribute_options` | `experiment_config.json` | `list[str]` | 任务解析与有效性校验 | 允许返回的 top-2 attributes 值域 | `["travel_time", "cost", "distance"]` |
| `survey_design.monotonicity_multiplier` | `experiment_config.json` | `float` | 数据准备脚本 | monotonicity 任务中加重目标方案负担的倍率 | `1.25` |
| `survey_design.dominance_time_penalty_min` | `experiment_config.json` | `float` | 数据准备脚本 | dominance 任务的时间惩罚 | `20.0` |
| `survey_design.dominance_wait_penalty_min` | `experiment_config.json` | `float` | 数据准备脚本 | dominance 任务的等待时间惩罚 | `10.0` |
| `survey_design.dominance_cost_penalty_chf` | `experiment_config.json` | `float` | 数据准备脚本 | dominance 任务的成本惩罚 | `6.0` |
| `intervention_tests.repeat_randomness_kappa` | `experiment_config_base.json` | `float` | 指标与分析脚本 | 随机性容忍系数 | `1.25` |
| `intervention_tests.bootstrap_repetitions` | `experiment_config_base.json` | `int` | 指标脚本 | bootstrap 次数 | `200` |
| `intervention_tests.utility_equivalent_manipulations` | `experiment_config_base.json` | `list[str]` | 指标脚本 | 视为 utility-equivalent 的干预类型 | `["paraphrase", "label_mask", "order_randomization"]` |
| `human_baseline_mnl.optimizer` | `experiment_config_base.json` | `str` | benchmark 估计脚本 | 人类基准 MNL 优化器 | `"scipy_bfgs"` |
| `panel_mnl.optimizer` | `experiment_config_base.json` | `str` | panel MNL 脚本 | AI panel MNL 优化器 | `"scipy_bfgs"` |
| `panel_mnl.maxiter` | `experiment_config_base.json` | `int` | panel MNL 脚本 | 最大迭代次数 | `500` |
| `biogeme.optimization_algorithm` | `experiment_config_base.json` | `str` | Biogeme 估计脚本 | Biogeme 优化方法 | `"scipy"` |
| `salcm.n_preference_classes` | `experiment_config_base.json` | `int` | SALCM 脚本 | 偏好类数量 | `3` |
| `salcm.n_scale_classes` | `experiment_config_base.json` | `int` | SALCM 脚本 | 尺度类数量 | `2` |
| `salcm.optimizer` | `experiment_config_base.json` | `str` | SALCM 脚本 | SALCM 优化器 | `"scipy_lbfgsb"` |
| `salcm.maxiter` | `experiment_config_base.json` | `int` | SALCM 脚本 | SALCM 最大迭代次数 | `800` |
| `salcm.membership_covariates` | `experiment_config_base.json` | `list[str]` | SALCM 脚本 | 类别归属协变量列表 | `["model_is_deepseek", "semantic_arm", "block_complexity_mean"]` |
| `analysis_v2.randomness_tolerance_kappa` | `experiment_config_base.json` | `float` | 分析脚本 | V2 分析的随机性容忍系数 | `1.25` |
| `analysis_v2.exact_repeat_signature_fields` | `experiment_config_base.json` | `list[str]` | 分析脚本 | exact repeat signature 的分组字段 | `["model_key", "scenario_id", "option_order"]` |
| `analysis_v2.paired_interventions` | `experiment_config_base.json` | `list[str]` | 分析脚本 | 成对干预列表 | `["paraphrase", "label_mask", "order_randomization"]` |
| `analysis_v2.between_block_interventions` | `experiment_config_base.json` | `list[str]` | 分析脚本 | block 间干预列表 | `["prompt_arm"]` |
| `torch.default_device` | `experiment_config_base.json` | `str` | torch 估计脚本 | 默认设备 | `"cuda_if_available"` |
| `torch.float_dtype` | `experiment_config_base.json` | `str` | torch 估计脚本 | 默认浮点类型 | `"float64"` |
| `n_monte_carlo_draws_biogeme` | `experiment_config_base.json` | `int` | Biogeme 估计脚本 | Biogeme Monte Carlo draw 数 | `32` |
| `n_monte_carlo_draws_torch_compare` | `experiment_config_base.json` | `int` | torch 对比脚本 | torch compare 阶段 draw 数 | `32` |
| `n_monte_carlo_draws_torch_final` | `experiment_config_base.json` | `int` | torch 最终估计脚本 | torch final 阶段 draw 数 | `500` |
| `shared_draw_dims` | `experiment_config_base.json` | `int` | draw 生成脚本 | Sobol draw 维数 | `2` |
| `shared_draw_names` | `experiment_config_base.json` | `list[str]` | draw 生成与估计脚本 | 各维度名字 | `["omega_car", "omega_env"]` |
| `api_credentials.local.json.api_key` | `api_credentials.local.json` | `str` | `resolve_llm_api_key()` | 本机直接存放的 Poe API key；可为空 | `""` |
| `api_credentials.local.json.api_key_env` | `api_credentials.local.json` | `str` | `resolve_llm_api_key()` | 本机显式指定的环境变量名 | `"POE_API_KEY"` |

## 术语对应

| 名词 | 在当前代码中的更准确含义 |
|---|---|
| `task` | 单道题。可能是 attitude item，也可能是一张 choice card 或 diagnostic item。 |
| `block template` | 一套可复用的问卷块模板。当前在数据准备阶段对应 `block_template_id`。 |
| `respondent block` | 在固定模型条件下，一个 block template 的一次实验单元实例。当前数据里通常对应某个 `respondent_id` 所属的整轮任务集合。 |
| `run` | 对一个 respondent block 的一次完整执行。当前代码里 `run_repeat` 表示 exact repeat 的第几次。 |

## 当前最常改的字段

如果你只是想调当前实验，通常优先改这些字段：

- `active_llm_key`
- `llm_models`
- `n_block_templates_per_model`
- `n_repeats_per_template`
- `survey_design.n_attitudes`
- `survey_design.n_core_tasks`
- `survey_design.n_paraphrase_twins`
- `survey_design.n_label_mask_twins`
- `survey_design.n_order_twins`
- `survey_design.n_monotonicity_tasks`
- `survey_design.n_dominance_tasks`
