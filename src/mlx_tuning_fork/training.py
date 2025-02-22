import warnings
import click
import yaml
import math
import mlx.optimizers as optim
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train, iterate_batches
from mlx_lm.tuner.utils import linear_to_lora_layers, build_schedule
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.utils import load, save_config
from mlx_lm.lora import print_trainable_parameters
from types import SimpleNamespace
from tqdm import tqdm
from mlx_tuning_fork.config import CONFIG_DEFAULTS, yaml_loader
from mlx_tuning_fork.reporting import WandbCallback
from pathlib import Path
from pprint import pprint

ALL_TRAIN_TYPES = ['lora', 'dora']
DORA_TRAIN_TYPES = ['dora']

@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option("--summary/--no-summary", default=False, help="Just summarize training data")
@click.option('--train-type',
              type=click.Choice(ALL_TRAIN_TYPES, case_sensitive=False),
              default="lora-completion-only")
@click.option('--wandb-project', default=None, type=str,
              help='Wandb project name')
@click.option('--wandb-run', default=None, type=str,
              help='Wandb run name')
@click.argument('config_files', nargs=-1)
def main(verbose, summary, train_type, wandb_project, wandb_run, config_files):
    previous_adapter = None
    for config_file in config_files:
        with open(config_file, "r") as file:
            config = yaml.load(file, yaml_loader)
            param_dict = {k: v for k, v in config.items()}
            if "model" not in param_dict and len(config_files) == 1:
                raise SyntaxError('Missing required "model" parameter')
            for key, default in CONFIG_DEFAULTS.items():
                if key not in param_dict:
                    param_dict[key] = default
            param_dict["verbose"] = verbose
            tokenizer_config = {"trust_remote_code": True if param_dict.get("trust_remote_code") else None}
            param_dict_eos_token = param_dict.get("eos_token")
            if param_dict_eos_token is not None:
                tokenizer_config["eos_token"] = param_dict["eos_token"]
            if previous_adapter is not None:
                param_dict["resume_adapter_file"] = previous_adapter
            args = SimpleNamespace(**param_dict)
            if verbose:
                pprint(param_dict)
        print("Loading pretrained model")
        model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)
        if previous_adapter is not None:
            print(f"Using previous model & adapters ({previous_adapter})")
        model.freeze()

        composably_train(args, config, config_file, model, summary, tokenizer, train_type, wandb_project,
                         wandb_run)
        if len(config_files) > 1:
            previous_adapter = str(Path(args.adapter_path) / "adapters.safetensors")

def composably_train(args, config, config_file, model, summary, tokenizer, train_type, wandb_project,
                     wandb_run):
    linear_to_lora_layers(
        model,
        args.num_layers,
        args.lora_parameters,
        use_dora=train_type in DORA_TRAIN_TYPES,
    )
    print_trainable_parameters(model)
    training_callback = None
    if wandb_project:
        if wandb_run is None:
            raise RuntimeError("Specify the name of a Wandb run to use with --wandb-run ")
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        wandb.init(project=wandb_project, name=wandb_run, config=config)
    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    if args.train and len(train_set) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid_set) == 0:
        warnings.warn(
            "Validation set not found or empty. Should provide validation set for fine-tuning."
        )
    if args.test and len(test_set) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test_set set for evaluation."
        )
    epoch_num_steps = (len(train_set) + args.batch_size - 1) // args.batch_size
    if args.epochs == -1:
        num_iterations = epoch_num_steps if args.iters == -1 else args.iters
    else:
        num_iterations = epoch_num_steps * args.epochs
    num_iterations = int(num_iterations)
    if wandb_project:
        training_callback = WandbCallback(tqdm(total=num_iterations))
    print(
        f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
        f"{len(train_set):,} records, {args.batch_size} at a time and with a validation set of "
        f"{len(valid_set):,} records, training {args.num_layers} layers out of {len(model.layers)} using qLoRa."
    )
    if args.evals_per_epoch:
        scaled_steps_per_eval = int(epoch_num_steps / args.evals_per_epoch)
        scaled_val_batches = int(len(valid_set) * args.eval_proportion_of_total / args.batch_size
                                 ) if args.eval_proportion_of_total else (
            int(len(valid_set) / (min(1, args.evals_per_epoch - 1) * args.batch_size))
        )
    else:
        scaled_steps_per_eval = int(num_iterations * args.validation_interval_proportion)
        scaled_val_batches = int(args.validations_per_train_item * args.validation_interval_proportion * num_iterations)
    scaled_val_batches = max(1, scaled_val_batches)
    scaled_steps_per_report = max(1, int(args.reporting_interval_proportion * num_iterations))
    if args.saves_per_epoch:
        scaled_save_every = int(epoch_num_steps / args.saves_per_epoch)
    else:
        scaled_save_every = int(args.adapter_save_interval_proportion * num_iterations)
    print(
        f"Calculating loss every {scaled_steps_per_report:,} steps, reporting validation loss every "
        f"{scaled_steps_per_eval:,} steps, validating with {scaled_val_batches:,} batches, "
        f"and saving the adapter every {scaled_save_every:,} steps."
    )
    if not summary:
        # Resume training the given adapters.
        if args.resume_adapter_file is not None:
            print(f"Loading pretrained adapters from {args.resume_adapter_file}")
            model.load_weights(args.resume_adapter_file, strict=False)

        adapter_path = Path(args.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        save_config(vars(args), adapter_path / "adapter_config.json")
        adapter_file = adapter_path / "adapters.safetensors"

        if args.train:
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
                args=TrainingArgs(batch_size=args.batch_size,
                                  iters=num_iterations,
                                  val_batches=scaled_val_batches,
                                  steps_per_report=scaled_steps_per_report,
                                  steps_per_eval=scaled_steps_per_eval,
                                  steps_per_save=scaled_save_every,
                                  max_seq_length=args.max_seq_length,
                                  adapter_file=str(adapter_file)),
                training_callback=training_callback
            )

        # Load the LoRA adapter weights which we assume should exist by this point
        if not adapter_file.is_file():
            raise ValueError(
                f"Adapter file {adapter_file} missing. "
                "Use --train to learn and save the adapters"
            )
        model.load_weights(str(adapter_file), strict=False)
        print(f"Loaded weights from {adapter_file}")

        if args.test:
            print(f"Testing ({len(test_set):,} records)")
            model.eval()

            test_loss = evaluate(
                model=model,
                dataset=test_set,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.test_batches
            )

            test_ppl = math.exp(test_loss)

            print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    else:
        total_num_tokens = 0
        max_tokens = 0
        _lengths = []
        for it, info in zip(
                range(1, num_iterations + 1),
                iterate_batches(
                    dataset=train_set,
                    tokenizer=tokenizer,
                    batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length,
                    train=False)
        ):
            lengths = info[-1]
            max_tokens = max(max_tokens, max(lengths))
            _lengths.extend(lengths)
            total_num_tokens += sum(lengths)
        print(f"A total of {total_num_tokens:,} training tokens, {total_num_tokens / num_iterations:.3f} per "
              f"step/iteration, an average of {total_num_tokens / len(_lengths):.3f} tokens per record, with"
              f" the largest having {max_tokens:,} tokens.")
        print(f"mlx_lm.lora --val-batches {scaled_val_batches} \\\n"
              f"            --steps-per-report {scaled_steps_per_report} \\\n"
              f"            --steps-per-eval {scaled_steps_per_eval} \\\n"
              f"            --save-every {scaled_save_every} \\\n"
              f"            --iters {num_iterations} -c {config_file}")


if __name__ == '__main__':
    main()
