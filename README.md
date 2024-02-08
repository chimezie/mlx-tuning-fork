# mlx-tuning-fork
Very basic framework for parameterized 
large language model (Q)LoRa fine-tuning with MLX.  It uses [mlx](https://github.com/ml-explore/mlx), [mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm), 
and [OgbujiPT](https://github.com/OoriData/OgbujiPT), and is based primarily on the excellent mlx-example libraries
but adds architecture for systematic running of easily parameterized finetunes as well as
an equivalent of HF's [train on completions](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only). 

It breaks out argument parameters into a YAML file, a configuration file (the only command line argument) expected 
in the following format:

```yaml
parameters:
    model: "..."
    num_tokens: 100
    write_every: 1
    temp: 0.8
    train: true
    [..]
```

* **epochs** (How many epochs, i.e., the number of iterations for a full pass of the data)
* **all_linear_layers** (Whether or not to apply (Q)Lora on all linear layers - no by default)
* **reporting_interval_proportion** (The proportion of iterations in an epoch to wait between recording training loss)
* **validation_interval_proportion** (Same proportions for interval between validations - defaults to 0.2 or 20%)
* **validations_per_iteration** (The ration of an epoch's iterations to total validations run - defaults to 1)
* **adapter_save_interval_proportion** (Same proportions for intervals between saving the LoRa adapter - defaults to .1)

There are a few other tuning parameters, but the rest are directly from 
[mlx-examples](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/trainer.py#L12).

## Learning Rate Schedules

Learning rate schedulers can be specified in the configuration file with a section such as the following (
for Cosine annealing):

```yaml
learning_schedule:
  type: "cosine"
  min_lr: 1e-7 #lower bound for learning rate 
  max_lr: 2e-5 #upper bound for learning rate 
  cycle_length: -1 #-1 for the number of steps/iterations in 1 epoch or a specific number otherwise (LR set to min_lr afterwards)
```
The following for Cosine Annealing with proportional warmup:

```yaml
learning_schedule:
  type: "cosine_w_warmup"
  start_lr: 1e-8 #learning rate used at start of the warm-up
  warmup_proportion: .1 #proportion of steps/iterations in 1 epoch to spend warming up
  min_lr: 1e-7
  max_lr: 2e-5
  cycle_length: -1
```

Otherwise a constant learning rate (specified via **learning_rate** top-level configuration variable) is used throughout

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

## Running Completion-only Supervised Learning

```bash
$ python -m mlx_tuning_fork.completion_only_training  --help
Usage: python -m mlx_tuning_fork.completion_only_training [OPTIONS] CONFIG_FILE

Options:
  --verbose / --no-verbose
  --summary / --no-summary        Just summarize training data
  -p, --prompt TEXT               Commandline prompt (overrides) prompt in
                                  YAML configuration
  -t, --temperature FLOAT         Prompt generation temperature
  -f, --prompt-format [mistral|chatml]
  -a, --adapter TEXT              Adapter to use instead of the one specified
                                  in the config file
  --help                          Show this message and exit.
```

## Dataset format

The dataset files are expected to be in this format:

```json
{"input": "[..]", 
  "output": "[..]"}
```

The prompt template specified is used to construct prompts and reponses to use for training purposes