import re
try:
    import wandb
except ImportError:
    wandb = None
import click
from tqdm import tqdm
import yaml
import numpy as np
import mlx.optimizers as optim
from types import SimpleNamespace
from mlx_tuning_fork.reporting import WandbCallback
from .training import ALL_TRAIN_TYPES, DORA_TRAIN_TYPES
from mlx_lm.utils import load
from mlx_tuning_fork.config import CONFIG_DEFAULTS as TF_CONFIG_DEFAULTS
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.utils import linear_to_lora_layers, build_schedule
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.lora import CONFIG_DEFAULTS

class Sweeper:
    def __init__(self, verbose, project_name, config, train_type):
        self.verbose = verbose
        self.project_name = project_name
        self.config = config
        self.train_type = train_type

    def sweep(self):
        if wandb is None:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        wandb.init(project=self.project_name)
        sweep_parameters = self.config["sweep_configuration"]["parameters"]
        wandb_config = wandb.config

        if "learning_rate" in sweep_parameters:
            self.config["learning_rate"] = wandb_config.learning_rate
            if self.verbose:
                print(f"learning rate: {self.config['learning_rate']}")
        if "rank" in sweep_parameters:
            self.config["lora_parameters"]["rank"] = wandb_config.rank
            if self.verbose:
                print(f"LoRA rank: {self.config['lora_parameters']['rank']}")
        if "scale" in sweep_parameters:
            self.config["lora_parameters"]["scale"] = wandb_config.scale
            if self.verbose:
                print(f"LoRA config: {self.config['lora_parameters']['scale']}")
        if "dropout" in sweep_parameters:
            self.config["lora_parameters"]["dropout"] = wandb_config.dropout
            if self.verbose:
                print(f"LoRA dropout: {self.config['lora_parameters']['dropout']}")
        if "batch_size" in sweep_parameters:
            self.config["batch_size"] = wandb_config.batch_size
            if self.verbose:
                print(f"batch size: {self.config['batch_size']}")
        self.config["eval_proportion_of_total"] = TF_CONFIG_DEFAULTS["eval_proportion_of_total"]
        self.config["validation_interval_proportion"] = TF_CONFIG_DEFAULTS["validation_interval_proportion"]
        self.config["validations_per_train_item"] = TF_CONFIG_DEFAULTS["validations_per_train_item"]
        self.config["reporting_interval_proportion"] = TF_CONFIG_DEFAULTS["reporting_interval_proportion"]
        self.config["eval_proportion_of_total"] = TF_CONFIG_DEFAULTS["eval_proportion_of_total"]

        args = SimpleNamespace(**self.config)

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

        print("Training")
        model.train()

        epoch_num_steps = (len(train_set) + args.batch_size - 1) // args.batch_size
        if args.epochs == -1:
            num_iterations = epoch_num_steps if args.iters == -1 else args.iters
        else:
            num_iterations = epoch_num_steps * args.epochs
        num_iterations = int(num_iterations)

        print(
            f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
            f"{len(train_set):,} records, {args.batch_size} at a time and with a validation set of "
            f"{len(valid_set):,} records, training {args.num_layers} layers out of {len(model.layers)} using qLoRa."
        )

        if args.evals_per_epoch:
            scaled_steps_per_eval = int(epoch_num_steps / args.evals_per_epoch)
            scaled_val_batches = int(len(valid_set) * args.eval_proportion_of_total / args.batch_size
                                     ) if args.eval_proportion_of_total else (
                int(len(valid_set) / ((args.evals_per_epoch - 1) * args.batch_size))
            )
        else:
            scaled_steps_per_eval = int(num_iterations * args.validation_interval_proportion)
            scaled_val_batches = int(
                args.validations_per_train_item * args.validation_interval_proportion * num_iterations)

        scaled_steps_per_report = int(args.reporting_interval_proportion * num_iterations)

        if args.saves_per_epoch:
            scaled_save_every = int(epoch_num_steps / args.saves_per_epoch)
        else:
            scaled_save_every = int(args.adapter_save_interval_proportion * num_iterations)

        print(
            f"Calculating loss every {scaled_steps_per_report:,} steps, reporting validation loss every "
            f"{scaled_steps_per_eval:,} steps, validating with {scaled_val_batches:,} batches, "
            f"and saving the adapter every {scaled_save_every:,} steps."
        )
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
            args = TrainingArgs(batch_size=args.batch_size,
                                iters=num_iterations,
                                val_batches=scaled_val_batches,
                                steps_per_report=scaled_steps_per_report,
                                steps_per_eval=scaled_steps_per_eval,
                                steps_per_save=scaled_save_every,
                                max_seq_length=args.max_seq_length),
            training_callback=WandbCallback(tqdm(total=num_iterations))
        )


@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option('--wandb-project', default=None, type=str, help='Wandb project name')
@click.option('--train-type',
              type=click.Choice(ALL_TRAIN_TYPES, case_sensitive=False),
              default="lora-completion-only")
@click.argument('config_file', type=click.File('r'))
def main(verbose, wandb_project, train_type, config_file):
    if wandb is None:
        raise ImportError('wandb module not available.  Install with `pip install wandb`')
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
    config = yaml.load(config_file, yaml_loader)
    sweep_id = wandb.sweep(sweep=config["sweep_configuration"], project=wandb_project)

    # Update defaults for unspecified parameters
    for config_defaults in [{k:v for k,v in CONFIG_DEFAULTS.items() if k not in TF_CONFIG_DEFAULTS},
                            TF_CONFIG_DEFAULTS]:
        for k, v in config_defaults.items():
            if not config.get(k, None):
                config[k] = v
    wandb.agent(sweep_id, function=Sweeper(verbose, wandb_project, config, train_type).sweep)


if __name__ == '__main__':
    main()
