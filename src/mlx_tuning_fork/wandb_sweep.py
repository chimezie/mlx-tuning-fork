import re
try:
    import wandb
except ImportError:
    wandb = None
import click
import yaml
import numpy as np
import mlx.optimizers as optim
from types import SimpleNamespace
from mlx_tuning_fork.reporting import WandbCallback
from mlx_lm.utils import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.lora import CONFIG_DEFAULTS, load_dataset

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
    def __init__(self, project_name, config):
        self.project_name = project_name
        self.config = config

    def sweep(self):
        if wandb is None:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        wandb.init(project=self.project_name)
        sweep_parameters = self.config["sweep_configuration"]["parameters"]
        wandb_config = wandb.config

        if "learning_rate" in sweep_parameters:
            self.config["learning_rate"] = wandb_config.learning_rate
            print(f"learning rate: {self.config['learning_rate']}")
        if "lora_rank" in sweep_parameters:
            self.config["lora_parameters"]["lora_rank"] = wandb_config.lora_rank
            print(f"lora rank: {self.config['lora_parameters']['lora_rank']}")
        if "lora_alpha" in sweep_parameters:
            self.config["lora_parameters"]["lora_alpha"] = wandb_config.lora_alpha
            print(f"lora alpha: {self.config['lora_parameters']['lora_alpha']}")
        if "batch_size" in sweep_parameters:
            self.config["batch_size"] = wandb_config.batch_size
            print(f"batch size: {self.config['lora_parameters']['batch_size']}")

        args = SimpleNamespace(**self.config)
        training_callback = WandbCallback()

        np.random.seed(args.seed)

        print("Loading pretrained model")
        model, tokenizer = load(args.model)

        # Freeze all layers
        model.freeze()
        # Convert linear layers to lora layers and unfreeze in the process
        linear_to_lora_layers(model, args.lora_layers, self.config["lora_parameters"])

        train_set, valid_set, test_set = load_dataset(args)
        trainingArgs = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=args.adapter_file,
            max_seq_length=args.max_seq_length,
        )
        print("Training")
        model.train()
        opt = optim.Adam(learning_rate=args.learning_rate)
        train(
            model=model,
            tokenizer=tokenizer,
            args=trainingArgs,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            training_callback=training_callback,
        )


@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option('--wandb-project', default=None, type=str, help='Wandb project name')
@click.argument('config_file', type=click.File('r'))
def main(verbose, wandb_project, config_file):
    if wandb is None:
        raise ImportError('wandb module not available.  Install with `pip install wandb`')
    config = yaml.load(config_file, yaml_loader)
    sweep_id = wandb.sweep(sweep=config["sweep_configuration"], project=wandb_project)

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if not config.get(k, None):
            config[k] = v

    wandb.agent(sweep_id, function=Sweeper(wandb_project, config).sweep)


if __name__ == '__main__':
    main()
