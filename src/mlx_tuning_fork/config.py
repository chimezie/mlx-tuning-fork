from mlx_lm import generate
import yaml
import re

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


CONFIG_DEFAULTS = {
    "num_tokens": generate.DEFAULT_MAX_TOKENS,
    "write_every": 1,
    "prompt": None,
    "train": False,
    "data": "data/",
    "temp": generate.DEFAULT_TEMP,
    "top_p": generate.DEFAULT_TOP_P,
    "lora_layers": 16,
    "batch_size": 4,
    "iters": -1,
    "epochs": -1,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_file": "adapters.npz",
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "seed": generate.DEFAULT_SEED,
    "max_tokens": 100,
    "tokens_per_eval": 10,
    "save_every": 100,
    "validation_scale": 5,
    "reporting_interval_proportion": 0.01, #10/1000
    "validation_interval_proportion": 0.2, #200/1000
    "validations_per_train_item": .5,
    "adapter_save_interval_proportion": .1,
    "train_loss_file": None,
    "validation_loss_file": None,
    "ignore_chat_template": False,
    "colorize": False,
    "trust_remote_code": False,
    "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},

}