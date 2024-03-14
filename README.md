# mlx-tuning-fork
A very basic framework for parameterized 
Large Language Model (Q)LoRa fine-tuning with MLX.  It uses [mlx](https://github.com/ml-explore/mlx), [mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm), 
and [OgbujiPT](https://github.com/OoriData/OgbujiPT), and is based primarily on the excellent mlx-example libraries
but adds very minimal architecture for systematic running of easily parameterized fine tunes, hyperparameter sweeping,
declarative prompt construction, an equivalent of HF's [train on completions](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only), and other capabilities.  

## Installation

For now, can be installed by cloning the repository and running (in the local working copy)

```bash
$ pip install .
```

Currently just has a single Mistral prompt format (-f/ --prompt-format) module, but with mlx-lm and OgbujiPT you can do something similar with other models:

* Llama
* Mixtral
* Qwen
* [..]

## Command-line options
You can get documentation of the command-line options for fine tuning via:

```commandline
Usage: python -m mlx_tuning_fork.training [OPTIONS] CONFIG_FILE

Options:
  --verbose / --no-verbose
  --summary / --no-summary        Just summarize training data
  --loom-file TEXT                An OgbujiPT word loom file to use for prompt
                                  construction
  --loom-markers TEXT             Loom marker values
  -p, --prompt TEXT               Commandline prompt (overrides) prompt in
                                  YAML configuration
  -t, --temperature FLOAT         Prompt generation temperature
  -nt, --num-tokens INTEGER       Overide number of tokens in config file
  --train-type [completion-only|self-supervised]
  -f, --prompt-format [mistral|chatml]
  -a, --adapter TEXT              Adapter to use instead of the one specified
                                  in the config file
  --wandb-project TEXT            Wandb project name
  --wandb-run TEXT                Wandb run name
  -rp, --repetition-penalty FLOAT
                                  The penalty factor for repeating tokens
                                  (none if not used)
  --repetition-context-size INTEGER
                                  The number of tokens to consider for
                                  repetition penalty
  -tp, --top-p FLOAT              Sampling top-p
  --build-prompt TEXT             Which word loom sections to use in building
                                  the claim (space-separated list of sections)
  --help                          Show this message and exit.
```

The format of the prompts used to train the model is specified via the `-f/--prompt-format` option, which currently
is one of **mistral** or **chatml**.

## Configuration

It uses mlx_lm's [YAML config format](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/examples/lora_config.yaml) 
and adds additional parameters and sections:

* **epochs** (How many epochs, i.e., the number of iterations for a full pass of the data)
* **reporting_interval_proportion** (The proportion of iterations in an epoch to wait between recording training loss - defaults to .01 or 1%)
* **validation_interval_proportion** (Same proportions for interval between validations - defaults to 0.2 or 20%)
* **validations_per_train_item** (The ration of the number of validation per training record seen - defaults to .5 or 1 validation per 2 training records)
* **adapter_save_interval_proportion** (Same proportions for intervals between saving the LoRa adapter - defaults to .1)

## Learning Rate Schedules

Learning rate schedulers can be specified in the configuration file with a section such as the following (
for Cosine annealing):

```yaml
learning_schedule:
  type: "cosine"
  max_lr: 2e-5 #upper bound for learning rate 
  cycle_length: -1 #-1 for the number of steps/iterations in 1 epoch or a specific number otherwise (LR set to min_lr afterwards)
```
The following for Cosine Annealing with proportional warmup:

```yaml
learning_schedule:
  type: "cosine_w_warmup"
  start_lr: 1e-8 #learning rate used at start of the warm-up, which ends at the top-level learning rate
  warmup_proportion: .1 #proportion of steps/iterations in 1 epoch to spend warming up
  min_lr: 1e-7
  cycle_length: -1
```

Otherwise a constant learning rate (specified via **learning_rate** top-level configuration variable) is used throughout

## Prompting
It also provides the ability to dispatch prompts to the model referenced in the config (in conjunction with any
LoRA adapters specified).  The `-p/--prompt` option can be used to provide a prompt, and the `-t/--temperature`, 
`-rp/--repetition-penalty`, `--repetition-context-size`, `-tp/--top-p` can be used to configure the evaluation of the prompt.
There is also an additional *colorize* parameter (specified in the config), which if true, will render the model's 
completion using a coloring scheme that captures the probability of each token using mlx_lm's capability in this regard.

## Declarative Prompts Construction

OgbujiPts [Word Loom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)) can also be used for templated construction of prompts.  

There are 3 command-line options for this:

```commandline
--loom-file TEXT                An OgbujiPT word loom file to use for prompt
                                construction
--build-prompt TEXT             Which word loom sections to use in building
                                 the claim (space-separated list of sections)                                  
--loom-markers TEXT             Loom marker values
```

The ``--loom-file`` option is the location of a word loom file to use for prompt construction, a [TOML](https://toml.io/) 
file.

The loom file provides a system prompt, context, as well as the user prompt.  The system prompt and context are optional, but the user prompt is not.

The ``--build-prompt`` option is a expected to be single or list of [table header names](https://toml.io/en/v1.0.0#table).
If only one is provided, it is assumed to the name of the table with a ``text`` key whose value will be used for 
the user prompt.  If two values are provided, they should be quoted and separated by spaces.  The first refers to 
a table that provides the system prompt and the second refers to the user prompt.  Finally, if three values are provided
they are assumed to be system prompt, context, and user prompt.

If they are not specified via ``--build-prompt``, the system prompt is assumed to be specified in a table named 
**system_prompt**, the context is from a table named **context**, and the user prompt is from a table named **question**.



If any of the text values in the corresponding tables have curly braces, the ``--loom-markers`` option can be used
to provide values for the names specified in between the braces.  It is expected to be a string in the format: 
``name=[.. value ..]``.

So, the following command-line:

```commandline
$ python -m mlx_tuning_fork.training --loom-file=loom.toml \
         --build-prompt "system_prompt_final templated_question_final" -f chatml \
         --loom-markers "medical_problems=[Lymphoid aggregate]" /path/to/loom.toml
```

where the contents of _loom.toml_ are:

```toml
lang = "en"
[system_prompt_final]
text = """You are a medical professional.  If you cannot provide an answer based on the given context, please let me know."""

[context]
text = """Lymphoid aggregates are a collection of B cells, T cells, and supporting cells, present within the stroma of various organs"""

[templated_question_final]
text = """The patient has {Lymphoid aggregate}.  Summarize the patient's problems"""
```

will result in the following [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) prompt being sent to the model:

```
<|im_start|>system
You are a medical professional.  If you cannot provide an answer based on the given context, please let me know.

Lymphoid aggregates are a collection of B cells, T cells, and supporting cells, present within the stroma of various organs
<|im_end|>
<|im_start|>user

The patient has {medical_problems}.  Summarize the patient's problems
<|im_end|>
<|im_start|>assistant
```

## Dataset format

The dataset files are expected to be in this format:

```json
{"input": "[..]", 
 "output": "[..]"}
```

## Learning (completion-only v.s. self-supervised)
By default, mlx_tuning_fork will train on completions only, using the **input** field for the input prompt and **output** for 
the expected output.  However, you can use mlx_lm's default self-supervised
learning using the `--train-type` with a value of _self-supervised_.  In this case, only the value of the output field
in the training data is used. 

## Running Weights and Biases (Wandb) Hyperparameter Sweeps ##

mlx_tuning_fork also allows you to run Wandb hyperparameter sweeps/searches using the mlx_tuning_form.wandb_sweep module.
You can get the command-line options for this via:  

```commandline
$ python -m mlx_tuning_fork.wandb_sweep --help
Usage: python -m mlx_tuning_fork.wandb_sweep [OPTIONS] CONFIG_FILE

Options:
  --verbose / --no-verbose
  --wandb-project TEXT      Wandb project name
  --help                    Show this message and exit.
```

It takes a single argument which is a [Wandb sweep configuration (YAML) file](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)
 .  The `--wandb-project` options refers to a Wandb project where the sweep output is be stored.

