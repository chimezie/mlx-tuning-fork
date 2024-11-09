import re
try:
    import wandb
except ImportError:
    wandb = None
import click
from tqdm import tqdm
import yaml
from pathlib import Path
import numpy as np
import mlx.optimizers as optim
from types import SimpleNamespace
from mlx_tuning_fork.reporting import WandbCallback
from .config import yaml_loader, get_prompt_formatter, PROMPT_FORMATS
from .training import ALL_TRAIN_TYPES, DORA_TRAIN_TYPES
from mlx_lm.utils import load, save_config
from mlx_tuning_fork.config import CONFIG_DEFAULTS
from mlx_lm.tuner.datasets import Dataset as mlx_lm_dataset, load_dataset
from mlx_lm.tuner.utils import linear_to_lora_layers, build_schedule
from mlx_lm.tuner.trainer import (TrainingArgs, train, default_loss, iterate_batches, input_masked_loss,
                                  iterate_delineated_batches)
from mlx_lm.lora import CONFIG_DEFAULTS

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

class Sweeper:
    def __init__(self, project_name, config, train_type, mask_inputs):
        self.project_name = project_name
        self.config = config
        self.train_type = train_type
        self.mask_inputs = mask_inputs

    def sweep(self):
        if wandb is None:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        wandb.init(project=self.project_name)
        sweep_parameters = self.config["sweep_configuration"]["parameters"]
        wandb_config = wandb.config

        if "learning_rate" in sweep_parameters:
            self.config["learning_rate"] = wandb_config.learning_rate
            print(f"learning rate: {self.config['learning_rate']}")
        if "rank" in sweep_parameters:
            self.config["lora_parameters"]["rank"] = wandb_config.rank
            print(f"LoRA rank: {self.config['lora_parameters']['rank']}")
        if "scale" in sweep_parameters:
            self.config["lora_parameters"]["scale"] = wandb_config.scale
            print(f"LoRA config: {self.config['lora_parameters']['scale']}")
        if "dropout" in sweep_parameters:
            self.config["lora_parameters"]["dropout"] = wandb_config.dropout
            print(f"LoRA dropout: {self.config['lora_parameters']['dropout']}")
        if "batch_size" in sweep_parameters:
            self.config["batch_size"] = wandb_config.batch_size
            print(f"batch size: {self.config['batch_size']}")
        self.config["mask_inputs"] = self.mask_inputs
        self.config["eval_proportion_of_total"] = CONFIG_DEFAULTS["eval_proportion_of_total"]
        self.config["validation_interval_proportion"] = CONFIG_DEFAULTS["validation_interval_proportion"]
        self.config["validations_per_train_item"] = CONFIG_DEFAULTS["validations_per_train_item"]
        self.config["reporting_interval_proportion"] = CONFIG_DEFAULTS["reporting_interval_proportion"]
        if self.mask_inputs:
            print(f"Masking inputs")

        args = SimpleNamespace(**self.config)
        training_callback = WandbCallback(tqdm(total=args.iters))

        np.random.seed(args.seed)

        print("Loading pretrained model")
        model, tokenizer = load(args.model)

        # Freeze all layers
        model.freeze()
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=self.train_type in DORA_TRAIN_TYPES,
        )

        print("Loading datasets")
        train_set, valid_set, test_set = load_dataset(args, tokenizer)


        # Resume training the given adapters.
        if args.resume_adapter_file is not None:
            print(f"Loading pretrained adapters from {args.resume_adapter_file}")
            model.load_weights(args.resume_adapter_file, strict=False)

        adapter_path = Path(args.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        save_config(vars(args), adapter_path / "adapter_config.json")
        adapter_file = adapter_path / "adapters.safetensors"

        print("Training")
        model.train()
        opt = optim.Adam(
            learning_rate=(
                build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
            )
        )

        epoch_num_steps = (len(train_set) + args.batch_size - 1) // args.batch_size
        num_iterations = 400

        print(
            f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
            f"{len(train_set):,} records, {args.batch_size} at a time and with a validation set of "
            f"{len(valid_set):,} records, training {args.num_layers} layers out of {len(model.layers)} using qLoRa."
        )

        if args.evals_per_epoch:
            scaled_steps_per_eval = int(num_iterations / args.evals_per_epoch)
            scaled_val_batches = int(len(valid_set) * args.eval_proportion_of_total / args.batch_size
                                     ) if args.eval_proportion_of_total else (
                int(len(valid_set) / ((args.evals_per_epoch - 1) * args.batch_size))
            )
        else:
            scaled_steps_per_eval = int(num_iterations * args.validation_interval_proportion)
            scaled_val_batches = int(
                args.validations_per_train_item * args.validation_interval_proportion * num_iterations)

        scaled_steps_per_report = int(args.reporting_interval_proportion * num_iterations)
        scaled_save_every = 1000

        print(
            f"Calculating loss every {scaled_steps_per_report:,} steps, reporting validation loss every "
            f"{scaled_steps_per_eval:,} steps, and validating with {scaled_val_batches:,} batches"
        )
        train(
            model=model,
            tokenizer=tokenizer,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            args = TrainingArgs(batch_size=args.batch_size,
                                iters=num_iterations,
                                val_batches=scaled_val_batches,
                                steps_per_report=scaled_steps_per_report,
                                steps_per_eval=scaled_steps_per_eval,
                                steps_per_save=scaled_save_every,
                                adapter_file=adapter_file,
                                max_seq_length=args.max_seq_length,
                                grad_checkpoint=args.grad_checkpoint),
            iterate_batches=(
                iterate_delineated_batches if args.mask_inputs else iterate_batches
            ),
            loss=input_masked_loss if self.mask_inputs else default_loss,
            training_callback=training_callback
        )


@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option('--wandb-project', default=None, type=str, help='Wandb project name')
@click.option('--train-type',
              type=click.Choice(ALL_TRAIN_TYPES, case_sensitive=False),
              default="lora-completion-only")
@click.option('-f', '--prompt-format',
              type=click.Choice(PROMPT_FORMATS, case_sensitive=False))
@click.option('--mask-inputs/--no-mask-inputs', default=False)
@click.argument('config_file', type=click.File('r'))
def main(verbose, wandb_project, train_type, prompt_format, mask_inputs, config_file):
    if wandb is None:
        raise ImportError('wandb module not available.  Install with `pip install wandb`')
    config = yaml.load(config_file, yaml_loader)
    sweep_id = wandb.sweep(sweep=config["sweep_configuration"], project=wandb_project)

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if not config.get(k, None):
            config[k] = v
    global prompt_formatter
    prompt_formatter = get_prompt_formatter(prompt_format)
    wandb.agent(sweep_id, function=Sweeper(wandb_project, config, train_type, mask_inputs).sweep)


if __name__ == '__main__':
    main()
