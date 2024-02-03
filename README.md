# mlx-tuning-fork
Very basic framework for (see: [mlx-examples PR 235](https://github.com/ml-explore/mlx-examples/pull/235)) parameterized 
large language model (Q)LoRa fine-tuning with MLX.  It uses [mlx](https://github.com/ml-explore/mlx), [mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm), 
and [OgbujiPT](https://github.com/OoriData/OgbujiPT), and is based primarily on the excellent mlx-example libraries
but adds much needed architecture (i.e. beyond example code) for systematic running of easily parameterized finetunes, mainly the equivalent
ability of HF's to [train on completions](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only). 

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

* ## **epochs** (How many epochs, i.e., the number of iterations for a full pass of the data)
* ## **all_linear_layers** (Whether or not to apply (Q)Lora on all linear layers - no by default)
* ## **reporting_interval_proportion** (The proportion of iterations in an epoch to wait between recording training loss)
* ## **validation_interval_proportion** (Same proportions for interval between validations - defaults to 0.2 or 20%)
* ## **validations_per_iteration** (The ration of an epoch's iterations to total validations run - defaults to 1)
* ## **adapter_save_interval_proportion** (Same proportions for intervals between saving the LoRa adapter - defaults to .1)

There are a few other parameters, but the rest are directly from 
[mlx-examples](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/trainer.py#L12).

For now, can be installed by cloning the repository and running (in the local working copy)

```bash
$ pip install .
```

