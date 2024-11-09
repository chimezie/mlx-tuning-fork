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
from mlx_tuning_fork.dataset import Dataset
from mlx_lm.tuner.datasets import Dataset as mlx_lm_dataset
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
    def __init__(self, project_name, config, train_type, mask_input):
        self.project_name = project_name
        self.config = config
        self.train_type = train_type
        self.mask_input = mask_input

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
        names = ("train", "valid", "test")
        if self.train_type in ('lora-completion-only', 'dora-completion-only', 'debug'):
            train_set, valid_set, test_set = (Dataset(Path(args.data) / f"{n}.jsonl") for n in names)
        else:
            train_set, valid_set, test_set = (mlx_lm_dataset(Path(args.data) / f"{n}.jsonl") for n in names)

        if args.train and len(train_set) == 0:
            raise ValueError(
                "Training set not found or empty. Must provide training set for fine-tuning."
            )
        if args.test and len(test_set) == 0:
            raise ValueError(
                "Test set not found or empty. Must provide test_set set for evaluation."
            )

        # Resume training the given adapters.
        if args.resume_adapter_file is not None:
            print(f"Loading pretrained adapters from {args.resume_adapter_file}")
            model.load_weights(args.resume_adapter_file, strict=False)

        adapter_path = Path(args.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        save_config(vars(args), adapter_path / "adapter_config.json")
        adapter_file = adapter_path / "adapters.safetensors"

        trainingArgs = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
        )
        print("Training")
        model.train()
        opt = optim.Adam(
            learning_rate=(
                build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
            )
        )

        train(
            model=model,
            tokenizer=tokenizer,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            args=trainingArgs,
            iterate_batches=(
                iterate_delineated_batches if args.mask_inputs else iterate_batches
            ),
            loss=input_masked_loss if self.mask_input else default_loss,
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
@click.option('--mask-input/--no-mask-input', default=False)
@click.argument('config_file', type=click.File('r'))
def main(verbose, wandb_project, train_type, prompt_format, mask_input, config_file):
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
    wandb.agent(sweep_id, function=Sweeper(wandb_project, config, train_type, mask_input).sweep)


if __name__ == '__main__':
    main()
