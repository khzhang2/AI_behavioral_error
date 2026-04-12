# AI Behavioral Error

这是一个面向学术研究的代码库，用于检验大语言模型生成的“虚拟受访者”是否能够复现人类在离散选择模型中的行为模式。代码以 Optima（Swissmetro）数据为基准，收集 AI 回答并比较人类与 AI 在混合选择模型（HCM）和多项式 Logit 模型（MNL）上的表现。

Python 运行环境使用：

```powershell
.\.venv\Scripts\python.exe
```

## 目录结构

```text
AI_behavioral_error/
├── data/
│   └── Swissmetro/
│       └── demographic_choice_psychometric/
│           ├── raw/
│           ├── ai_collection_qwen3.5_9b/
│           ├── ai_collection_deepseek_r1_8b/
│           ├── human_cleaned_wide.csv
│           ├── human_cleaned_long.csv
│           ├── human_respondent_profiles.csv
│           ├── shared_sobol_draws_32.npy
│           ├── shared_sobol_draws_500.npy
│           ├── human_benchmark_sample_summary.json
│           ├── optima_codebook.json
│           └── optima_data_description.md
├── scripts/
├── experiments/
│   └── Swissmetro/
│       └── YYYYMMDD_<keywords>_<model_name>_<version>/
├── docs/
├── experiment_config.json
├── experiment_config_base.json
├── biogeme.toml
├── biogeme_runtime.toml
├── pyproject.toml
└── AGENTS.md
```

### scripts 说明

`scripts/` 下是可直接运行的脚本文件。脚本之间通过直接导入复用共享函数，入口路径约定在根目录下统一。

### experiments 说明

`experiments/` 只用于归档。每次运行的实验会保存在：

```
Swissmetro/YYYYMMDD_<keywords>_<model_name>_<version>/
```

每个实验目录内包含：

- `experiment_config.json`：该次实验最终实际使用的完整配置，只保留一份。
- `outputs/`：只存放 AI 问答原始数据与原始记录摘要，例如 `raw_interactions.jsonl`、`respondent_transcripts.json`、`run_respondents.json`、`ai_collection_summary.json`。
- 实验根目录：存放派生的 AI 面板数据、human / AI 参数估计、diagnostics 结果，以及唯一的 `experiment_summary.md`。

## 关键约定

1. 所有路径以 `optima_common.py` 中的 `ROOT_DIR = Path(__file__).resolve().parents[1]` 为基准。
2. 实验设置目前采用“**两层配置**”：`experiment_config.json` 保留实验级调参字段，`experiment_config_base.json` 只保留稳定默认值与方法默认值。
3. 运行时，`experiment_config.json` 会通过 `config_base_file` 与 `config_overrides` 合并后加载为完整配置。
4. 每个实验文件夹只对应一个 model。当前 experiment-ready 工作流要求 `llm_models` 只包含一个条目。
5. 每次估计/采集脚本中的 `archive_experiment_config(...)` 会把最终完整配置写到对应实验文件夹中的单个 `experiment_config.json`，不再生成编号副本。
6. `outputs/` 只保留 AI 问答原始数据；其他估计、diagnostics 和结果一律放在实验根目录。
7. 关键约定包含 AI 收集支持 `--resume`。当前 intervention questionnaire 收集脚本采用增量落盘：每拿到一次 grounding、attitude 或 task 响应，就立刻追加写入 `outputs/raw_interactions.jsonl`，并同步更新 `parsed_attitudes.csv`、`parsed_task_responses.csv`、`outputs/respondent_transcripts.json` 与 `outputs/run_respondents.json`。因此中途中断后，`--resume` 会先读取这些文件，再从尚未完成的 respondent 和尚未完成的问题继续，而不是默认整个人从头重跑。非法响应尝试一次修复，Sobol 抽样预先生成并共享以保证 Monte Carlo 积分可比；Biogeme 的 `*.iter`、`biogeme.toml`、`biogeme_runtime.toml` 为本地运行文件且默认不提交。

当前更适合放在 `experiment_config.json` 的字段包括：

- `active_llm_key`
- `llm_models`
- `n_block_templates_per_model`
- `n_repeats_per_template`
- `survey_design`

这样总共设计几个问卷模板、每个模板做几次重复、每份问卷有几类题与几道题，都集中在同一个实验调参文件里。

在当前代码里，`total_tasks` 会优先按各组件数量自动合成：

`n_core_tasks + n_paraphrase_twins + n_label_mask_twins + n_order_twins + n_monotonicity_tasks + n_dominance_tasks`

因此更稳妥的做法是主要维护这些组件计数。

完整字段说明见 [docs/experiment_config_field_reference.md](</Users/kaihangzhang/Downloads/GitHub/Research codes repo/AI_behavioral_error/docs/experiment_config_field_reference.md>)。

当前推荐约定是：运行脚本时一律使用 `--model-key` 指定模型，而不是直接传 `model` 名字。`key` 是代码内部的唯一标识；`model` 只是后端真实模型名，不建议拿来当 CLI 主键。

## 当前 experiment-ready 主线脚本

当前与单实验单模型目录约定对齐的主线脚本包括：

- `prepare_optima_data.py`
- `run_optima_ai_collection.py`
- `run_optima_latent_regime_ai_collection.py`
- `prepare_optima_intervention_regime_data.py`
- `estimate_optima_biogeme_hcm.py`
- `estimate_optima_torch_hcm.py`
- `estimate_optima_biogeme_mnl.py`
- `prepare_optima_latent_regime_data.py`
- `optima_latent_regime_questionnaire.py`
- `estimate_optima_panel_mnl.py`
- `estimate_optima_salcm.py`
- `summarize_optima_intervention_regime.py`
- `summarize_optima_latent_regime.py`
- `optima_common.py`

更早的 HCM / 旧 MNL 脚本仍然保留在仓库中，但当前 experiment-ready 目录约定优先以上述主线为准。

## 使用 Poe 作为 AI 回答后端

Poe 仍然通过 `llm_models` 接入。为了保持不同后端的格式统一，更推荐继续在 `llm_models` 里写 `provider`、`model`、`base_url`、采样参数和运行参数。本地凭据文件 `api_credentials.local.json` 只负责保存 `api_key` 或 `api_key_env`。

示例：

```json
{
  "key": "poe_gpt54_nano",
  "respondent_prefix": "PO",
  "provider": "poe",
  "model": "gpt-5.4-nano",
  "base_url": "https://api.poe.com/v1",
  "credentials_file": "api_credentials.local.json",
  "format": "json",
  "reasoning_effort": "none",
  "temperature": 0.1,
  "top_p": 0.95,
  "top_k": 20,
  "seed": 20260412,
  "grounding_num_predict": 96,
  "attitude_num_predict": 32,
  "task_num_predict": 96,
  "timeout_sec": 240
}
```

`api_credentials.local.json` 只用于本机密钥，不会进入版本控制。建议直接写成顶层结构：

```json
{
  "api_key": "",
  "api_key_env": "POE_API_KEY"
}
```

更安全的默认做法是：把 `api_key` 留空，然后在系统环境变量里设置 `POE_API_KEY`。当前代码会在检测到 `api_key` 为空时，自动读取系统环境变量里的 `POE_API_KEY`。如果你想改用别的变量名，也可以在模型配置或本地文件里显式写 `api_key_env`。

### 可选：按模型自定义输出解码器

不同后端对“正式输出”和“thinking / reasoning 输出”的组织方式并不完全相同。现在代码支持在 `llm_models` 中为每个模型单独提供 `response_decoder`，用于覆盖默认解析路径。

示例：

```json
{
  "response_decoder": {
    "response_text_path": "choices.0.message.content",
    "thinking_text_path": "choices.0.message.reasoning_content",
    "done_reason_path": "choices.0.finish_reason",
    "prompt_eval_count_path": "usage.prompt_tokens",
    "eval_count_path": "usage.completion_tokens"
  }
}
```

如果不写 `response_decoder`，代码会按 provider 使用默认解析规则。当前仓库默认仍按 non-thinking 模式运行。

### 可选：显式控制 reasoning

对于 Poe / OpenAI-compatible 分支，当前代码支持在 `llm_models` 中设置 `reasoning_effort`。如果后端兼容 OpenAI 的该字段，可以用：

```json
{
  "reasoning_effort": "none"
}
```

来请求 non-thinking。  
如果某个模型使用别的私有字段，就把它放进 `extra_body`。

### 配置优先级

1. `experiment_config.json` 的 `config_overrides` 优先于 `experiment_config_base.json`。
2. 在单个模型条目内部，实验配置里直接写的字段优先于 `api_credentials.local.json`。
3. 本地凭据文件只会补全实验配置里没有写的字段。

这意味着当前代码里，`experiment_config.json` 确实是更高优先级的调参入口文件。

### API Key 读取顺序

1. 先读模型配置或本地凭据中合并后的 `api_key`。
2. 再读合并后的 `api_key_env`。
3. 如果前两步都为空，且 `provider = "poe"`，自动回退读取系统环境变量 `POE_API_KEY`。

### POE
#### 加入 macOS（zsh） 用户的环境变量设置示例
```bash
echo 'export POE_API_KEY="你的_poe_api_key"' >> ~/.zshrc
source ~/.zshrc
```

确认是否设置成功，可以执行：
```bash
echo $POE_API_KEY
```

#### 基本参数
- `"base_url": "https://api.poe.com/v1"`
- `"provider": "poe"`

#### DeepSeek
#### 加入 macOS（zsh） 用户的环境变量设置示例
```bash
echo 'export DeepSeek_API_KEY="你的_deepseek_api_key"' >> ~/.zshrc
source ~/.zshrc
```

确认是否设置成功，可以执行：
```bash
echo $DeepSeek_API_KEY
```

#### 基本参数
- `"base_url": "https://api.deepseek.com"`
- `"provider": "deepseek"`
- `"temperature": 1.3` (https://api-docs.deepseek.com/quick_start/parameter_settings)


## 本地文件与目录

- `experiment_config.json`：实验入口配置（覆盖层）。
- `experiment_config_base.json`：完整可复现实验基线配置。
- `api_credentials.local.json`：本机 Poe 凭据与本地模型设置（可选）。
- `data/Swissmetro/demographic_choice_psychometric`：保留的人类基准与原始数据。
- `experiments/Swissmetro/...`：所有归档实验结果。
- `biogeme_runtime.toml`：Biogeme 运行时文件。
- `*.iter`：Biogeme 中间迭代快照。

每个 experiment folder 通常包含：

- `outputs/raw_interactions.jsonl`
- `outputs/respondent_transcripts.json`
- `outputs/run_respondents.json`
- `outputs/ai_collection_summary.json`
- 根目录下的 `persona_samples.csv`
- 根目录下的 `parsed_attitudes.csv`
- 根目录下的 `parsed_task_responses.csv`
- 根目录下的 `ai_panel_long.csv`
- 根目录下的 `ai_panel_block.csv`

## 工作流程

1. 准备实验数据包。
2. 运行 questionnaire 收集脚本生成 AI 回答（或继承 `--resume`）。
3. 运行基准与模型估计。
4. 在 `experiments/Swissmetro/...` 下查看归档配置与输出。

### questionnaire 收集脚本的并发与 `--resume` 规则

- 当前 intervention questionnaire 收集脚本的并发单位是 `respondent`，不是单题。
- 同一个 `respondent` 内部始终串行，顺序为：`grounding -> attitudes -> tasks`。
- 不同 `respondent` 之间可以并发，请通过 `--max-workers` 或 `experiment_config.json` 中的 `collection.max_workers` 控制。
- 因此实际并发度是 `min(max_workers, 当前尚未完成的 respondents 数量)`。
- 如果大部分 respondent 已经完成，而只剩 1 个或 2 个 respondent 未完成，那么即使传入 `--max-workers 8`，恢复时也只会表现为 1 线程或 2 线程的有效并发。
- `--resume` 不会把同一个 respondent 内部改成并行；它只是在 respondent 之间继续并发，在 respondent 内部从已保存的位置继续串行执行。
- 当 API 调用是主要瓶颈时，Python 进程通常呈现低 CPU 占用，因为它主要在等待网络响应；这不代表脚本自动退化成全串行。

示例命令：

```bash
./.venv/bin/python scripts/run_optima_intervention_regime_ai_collection.py --model-key poe_gpt54_nano --max-workers 8 --resume
```
