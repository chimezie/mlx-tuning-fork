# mlx-tuning-fork
A very basic framework for parameterized 
Large Language Model (Q)LoRa fine-tuning with MLX.  It uses [mlx](https://github.com/ml-explore/mlx), [mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm), 
and [OgbujiPT](https://github.com/OoriData/OgbujiPT), and is based primarily on the excellent mlx-example libraries
but adds very minimal architecture for systematic running of easily parameterized fine tunes, hyperparameter sweeping,
declarative prompt construction, an equivalent of HF's [train on completions](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only), and other capabilities.  

## Installation

Can be installed via:

```bash
$ pip install mlx-tuning-fork
```

## Command-line options
You can get documentation of the command-line options for the fine-tuning command (**mlx_tuning_fork_training**) via:

```commandline
% mlx_tuning_fork_training --help
Usage: python -m mlx_tuning_fork.training [OPTIONS] CONFIG_FILE

Options:
  --verbose / --no-verbose
  --summary / --no-summary        Just summarize training data
  --train-type [lora-completion-only|dora-completion-only|lora-self-supervised|dora-self-supervised]
  -f, --prompt-format [mistral|chatml|llama3|alpaca|phi|gemma]
  --wandb-project TEXT            Wandb project name
  --wandb-run TEXT                Wandb run name
  --help                          Show this message and exit.
```

The format of the prompts used to train the model is specified via the `-f/--prompt-format` option, which currently
is one of:

- mistra
- chatml
- llama3
- alpaca
- phi
- gemma

## Configuration

It uses mlx_lm's [YAML config format](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/examples/lora_config.yaml) 
and adds additional parameters and sections.

It provides configurations that can be used to automatically determine values for the [mlx_lm fine-tuning parameters](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#fine-tune) that configure
how the training is run:

- the total number of steps/iterations to run (`iters`)
- the number of iterations between validation runs to evaluate the model (`steps_per_eval`, same semantics as [axolotl's](https://axolotl-ai-cloud.github.io/axolotl/docs/config.html) `eval_steps`)
- the number of iterations between the calculation of training loss (`steps_per_report`)
- the number of steps between the writing out of the (D/L)oRA adapter (`save_every`)
- the number of batches of records from the validation set to use for each validation run (`val_batches`)

In particular, the following additional configurations can be used to automatically determine the values for these parameters:

* **epochs** (How many epochs, i.e., the number of iterations for a full pass of the data)
    * If specified, it determines the total number of iterations (`iters`) from the size of the training set, the batch size (`batch_size`), and the requested number of epochs
* **evals_per_epoch** Same as axolotl's hyperparameter of the same name (The number of validations to run for each epoch)
    * If specified, it calculates `steps_per_eval` using `evals_per_epoch`, the size of the training set, the batch size, the number of iterations in an epoch, and the requested number of epochs.  It also calculates `val_batches` such that all validation records are used by the end of the epoch or according to `eval_proportion_of_total`, if specified. 
* **eval_proportion_of_total** (The proportion of the complete set of validation data to use for each validation - defaults to .25 or 25%)
    * Used with `evals_per_epoch` and, if provided, sets `scaled_val_batches` accordingly   
* **reporting_interval_proportion** (The proportion of iterations in an epoch to wait between calculating training loss - defaults to .01 or 1%)
    * Used to determine `steps_per_report`   
* **validation_interval_proportion** (The proportion of iterations in an epoch to wait between validations - defaults to 0.2 or 20%)
    * This is used if `evals_per_epoch` is not specified to determine `steps_per_eval` 
* **validations_per_train_item** (The ratio of the number of validation per training record seen - defaults to .5 or 1 validation per 2 training records)
    * This is used if `evals_per_epoch` is not provided and is used to determine `scaled_val_batches` 
* **saves_per_epoch** Same as axolotl's hyperparameter of the same name (The number of times a LoRa adapter is saved for each epoch - defaults to 2)
    * If provided, it is used to determine `save_every`
* **adapter_save_interval_proportion** (Same proportions for intervals between saving the LoRa adapter - defaults to .1)
    * Used to determine `save_ever` if `saves_per_epoch` is not provided

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

## Generation

mlx-tuning-fork also includes a command for generating from mlx models: **mlx_tuning_fork_generate**

```commandline
% mlx_tuning_fork_generate --help
Usage: python -m mlx_tuning_fork.generate [OPTIONS] MODEL_NAME

Options:
  --loom-file TEXT                An OgbujiPT word loom file to use for prompt
                                  construction
  --loom-markers TEXT             Loom marker values
  -p, --prompt TEXT               Commandline prompt (overrides) prompt in
                                  YAML configuration
  -t, --temperature FLOAT         Prompt generation temperature
  -nt, --num-tokens INTEGER       Overide number of tokens in config file
  -f, --prompt-format [mistral|chatml|llama3|alpaca|phi|gemma]
  -a, --adapter-path TEXT         Adapter to use instead of the one specified
                                  in the config file
  -rp, --repetition-penalty FLOAT
                                  The penalty factor for repeating tokens
                                  (none if not used)
  --repetition-context-size INTEGER
                                  The number of tokens to consider for
                                  repetition penalty
  -tp, --top-p FLOAT              Sampling top-p
  --min-p FLOAT                   Sampling min-p
  --min-p-tokens INTEGER          Sampling min-p
  --build-prompt TEXT             Which word loom sections to use in building
                                  the claim (space-separated list of sections)
  --trust-remote-code / --no-trust-remote-code
  --eos-token TEXT                End of sequence token for tokenizer
  --seed INTEGER                  PRNG seed
  --colorize / --no-colorize      Colorize output based on token probability
  --cot-source TEXT               The name of the file with an apply chat
                                  template structure to use as the basis for a
                                  few-shot prompt construction
  --help                          Show this message and exit.

```

It allows you to generate from a model referenced in the config (in conjunction with any
LoRA adapters specified).  The `-p/--prompt` option can be used to provide a prompt, and the `-t/--temperature`, 
`-rp/--repetition-penalty`, `--repetition-context-size`, `-tp/--top-p`, and `--min-p` can be used to configure the 
various parameters of the prompt evaluation. There is also an additional, boolean `--colorize/--no-colorize` parameter 
(defaults to false), which if true, will render the model's 
completion using a coloring scheme that captures the probability of each token.

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

If any of the table header name of the context is of the form ``[filename.txt]`` the contents of the specified filename are used
for the instead.

If any of the text values in the corresponding tables have curly braces, the ``--loom-markers`` option can be used
to provide values for the names specified in between the braces.  It is expected to be a string in the format: 
``name=[.. value ..]``.

So, the following command-line:

```commandline
$ python -m mlx_tuning_fork.training --loom-file=loom.toml \
         --build-prompt "system_prompt_final context templated_question_final" -f chatml \
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
text = """The patient has {medical_problems}.  Summarize the patient's problems"""
```

will result in the following [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) prompt being sent to the model:

```
<|im_start|>system
You are a medical professional.  If you cannot provide an answer based on the given context, please let me know.

Lymphoid aggregates are a collection of B cells, T cells, and supporting cells, present within the stroma of various organs
<|im_end|>
<|im_start|>user

The patient has Lymphoid aggregate.  Summarize the patient's problems
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
  --wandb-project TEXT            Wandb project name
  --train-type [completion-only|self-supervised]
  -f, --prompt-format [mistral|chatml]
  --help                          Show this message and exit.
```

It takes a single argument which is a [Wandb sweep configuration (YAML) file](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)
 .  The `--wandb-project` options refers to a Wandb project where the sweep output is be stored.

