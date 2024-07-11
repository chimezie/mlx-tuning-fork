import yaml
import re
from mlx_tuning_fork.reporting import WandbCallback
from tqdm import tqdm


def scale_training_parameters(args, model, train_set, training_callback, valid_set, wandb_project=None):
    epoch_num_steps = (len(train_set) + args.batch_size - 1) // args.batch_size
    if args.epochs == -1:
        num_iters = epoch_num_steps if args.iters == -1 else args.iters
    else:
        num_iters = epoch_num_steps * args.epochs
    num_iters = int(num_iters)
    if wandb_project:
        training_callback = WandbCallback(tqdm(total=num_iters))
    print(
        f"{num_iters:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
        f"{len(train_set):,} records, {args.batch_size} at a time and with a validation set of "
        f"{len(valid_set):,} records, training {args.lora_layers} layers out of {len(model.layers)} using qLoRa."
    )
    steps_per_report = int(args.reporting_interval_proportion * num_iters)
    steps_per_eval = int(num_iters * args.validation_interval_proportion)
    val_batches = int(args.validations_per_train_item * args.validation_interval_proportion * num_iters)
    save_every = int(args.adapter_save_interval_proportion * num_iters)
    return num_iters, save_every, steps_per_eval, steps_per_report, val_batches, training_callback

def get_prompt_formatter(prompt_format):
    if prompt_format == 'mistral':
        from mlx_tuning_fork.prompt_templates.mistral import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'chatml':
        from mlx_tuning_fork.prompt_templates.chatml import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'llama3':
        from mlx_tuning_fork.prompt_templates.llama3 import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'alpaca':
        from mlx_tuning_fork.prompt_templates.alpaca import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'phi':
        from mlx_tuning_fork.prompt_templates.phi import TrainingRecordHandler
        return TrainingRecordHandler

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
    "num_tokens": 100,
    "write_every": 1,
    "prompt": None,
    "train": False,
    "data": "data/",
    "temp": 0.6,
    "top_p": 1.0,
    "lora_layers": 16,
    "batch_size": 4,
    "iters": -1,
    "epochs": -1,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "seed": 0,
    "max_tokens": 100,
    "save_every": 100,
    "validation_scale": 5,
    "adapter_path": "adapters",
    "reporting_interval_proportion": 0.01, #10/1000
    "validation_interval_proportion": 0.2, #200/1000
    "validations_per_train_item": .5,
    "adapter_save_interval_proportion": .1,
    "ignore_chat_template": False,
    "colorize": False,
    "trust_remote_code": False,
    "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},
    "grad_checkpoint": False
}
