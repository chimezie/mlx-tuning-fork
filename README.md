# mlx-tuning-fork
Very basic framework for (see: [mlx-examples PR 235](https://github.com/ml-explore/mlx-examples/pull/235)) parameterized 
large language model (Q)LoRa fine-tuning with MLX.  It uses [mlx](https://github.com/ml-explore/mlx), [mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm), 
and [OgbujiPT](https://github.com/OoriData/OgbujiPT), and is based primarily on the excellent mlx-example libraries
but adds much needed architecture (i.e. beyond example code) for systematic running of easily parameterized finetunes, mainly the equivalent
ability of HF's to [train on completions](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only). 

It breaks out argument parameters into a YAML file, a configuration file (the only command line argument) expected 
in the following format:

parameters:
    model: "..."
    num_tokens: 100
    write_every: 1
    temp: 0.8
    train: true
    [..]

An epoch parameter determine the number of iterations if provided (the number needed for a
full pass of the data, i.e., an epoch).

For now, can be installed by cloning the repository and running (in the local working copy)

```bash
$ pip install .
```

