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
from mlx_tuning_fork.training import completions_only_loss
from mlx_tuning_fork.tuning.dynamic_learning import SCHEDULE_CONFIGURATION_TYPE_TO_CLASS
from mlx_lm.utils import load, save_config
import mlx.core as mx
from mlx_tuning_fork.dataset import Dataset
from mlx_lm.tuner.datasets import Dataset as mlx_lm_dataset
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import TrainingArgs, train, default_loss, iterate_batches
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

def completions_only_iterate_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    #idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]))
    #See https://github.com/ml-explore/mlx-examples/issues/583
    idx = range(len(dataset))

    # Make the batches:
    batch_idx = [
        idx[i: i + batch_size] for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            input_text = []
            output_text = []
            for j in batch_idx[i]:
                record = dataset[j]
                input_text.append(prompt_formatter.get_input(record))
                output_text.append(prompt_formatter.get_output(record))

            input_batch = [tokenizer.encode(record) for record in input_text]
            output_batch = [tokenizer.encode(record, add_special_tokens=False) +
                            [tokenizer.eos_token_id] for record in output_text]

            input_lengths = [len(x) for x in input_batch]
            output_lengths = [len(x) for x in output_batch]

            full_labels = [input_batch[idx] + output_batch[idx] for idx in range(batch_size)]
            lengths = [len(x) for x in full_labels]

            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )
            pad_to = 8
            max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size, max_length_in_batch), np.int32)
            adjusted_lengths = []
            for j in range(batch_size):
                input_length = input_lengths[j]
                full_ids_end_idx = input_length + min(output_lengths[j], max_length_in_batch - input_length)
                adjusted_lengths.append(full_ids_end_idx)
                batch_arr[j, :full_ids_end_idx] = full_labels[j][:full_ids_end_idx]
            batch = mx.array(batch_arr)
            yield batch, mx.array(input_lengths), mx.array(adjusted_lengths)

        if not train:
            break


class Sweeper:
    def __init__(self, project_name, config, train_type):
        self.project_name = project_name
        self.config = config
        self.train_type = train_type
        self.loss_fn = completions_only_loss if train_type == 'completion-only' else default_loss
        self.iterate_batches_fn = (completions_only_iterate_batches if train_type == 'completion-only'
                                   else iterate_batches)

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
            print(f"lora rank: {self.config['lora_parameters']['rank']}")
        if "alpha" in sweep_parameters:
            self.config["lora_parameters"]["alpha"] = wandb_config.alpha
            print(f"lora alpha: {self.config['lora_parameters']['alpha']}")
        if "dropout" in sweep_parameters:
            self.config["lora_parameters"]["dropout"] = wandb_config.dropout
            print(f"lora dropout: {self.config['lora_parameters']['dropout']}")
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
        # Convert linear layers to lora layers and unfreeze in the process
        linear_to_lora_layers(model, args.lora_layers, self.config["lora_parameters"])

        names = ("train", "valid", "test")
        if self.train_type == 'completion-only':
            dataset = Dataset
        else:
            dataset = mlx_lm_dataset
        train_set, valid_set, test_set = (dataset(Path(args.data) / f"{n}.jsonl") for n in names)


        if "learning_schedule" in self.config:
            scheduler = SCHEDULE_CONFIGURATION_TYPE_TO_CLASS[
                self.config["learning_schedule"]["type"]].from_configuration(args.learning_rate, self.config,
                                                                             args.iters)
        else:
            scheduler = args.learning_rate

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
        opt = optim.Adam(learning_rate=scheduler)
        train(
            model=model,
            tokenizer=tokenizer,
            args=trainingArgs,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            loss=self.loss_fn,
            iterate_batches=self.iterate_batches_fn,
            training_callback=training_callback,
        )


@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option('--wandb-project', default=None, type=str, help='Wandb project name')
@click.option('--train-type',
              type=click.Choice(['completion-only', 'self-supervised'], case_sensitive=False),
              default="completion-only")
@click.option('-f', '--prompt-format',
              type=click.Choice(['mistral', 'chatml'], case_sensitive=False))
@click.argument('config_file', type=click.File('r'))
def main(verbose, wandb_project, train_type, prompt_format, config_file):
    if wandb is None:
        raise ImportError('wandb module not available.  Install with `pip install wandb`')
    config = yaml.load(config_file, yaml_loader)
    sweep_id = wandb.sweep(sweep=config["sweep_configuration"], project=wandb_project)

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if not config.get(k, None):
            config[k] = v
    global prompt_formatter
    if prompt_format == 'mistral':
        from mlx_tuning_fork.prompt_templates.mistral import TrainingRecordHandler
        prompt_formatter = TrainingRecordHandler
    elif prompt_format == 'chatml':
        from mlx_tuning_fork.prompt_templates.chatml import TrainingRecordHandler
        prompt_formatter = TrainingRecordHandler
    wandb.agent(sweep_id, function=Sweeper(wandb_project, config, train_type).sweep)


if __name__ == '__main__':
    main()
