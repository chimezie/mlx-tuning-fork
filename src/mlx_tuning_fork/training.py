import mlx.optimizers as optim
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, evaluate, train, iterate_batches
from mlx_lm.tuner.utils import linear_to_lora_layers, build_schedule
from mlx_lm.utils import load, save_config
from mlx_lm.lora import print_trainable_parameters
from mlx_lm.tuner.datasets import load_dataset, CompletionsDataset, ChatDataset
from mlx_tuning_fork.config import CONFIG_DEFAULTS, yaml_loader
from mlx_tuning_fork.reporting import WandbCallback
from mlx_tuning_fork.tuning import ExtendableCompletionsDataset, ExtendableChatDataset, CompletionsDatasetCollection
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import click
import yaml
import math
import warnings

def no_bos_or_eos(sequence: List, bos: int, eos: int) -> List:
    removed_bos = sequence if sequence[0] != bos else sequence[1:]
    return removed_bos[:-1] if removed_bos[-1] == eos else removed_bos

def contains(small_list: List, big_list: List) -> Tuple[int, int]:
    """
    Returns the beginning and end index of the first occurrence of small_list in big_list.
    """
    small_list_length = len(small_list)
    for ind in (i for i, e in enumerate(big_list) if e == small_list[0]):
        if big_list[ind : ind + small_list_length] == small_list:
            return ind, ind + small_list_length - 1

def input_masked_loss(model, inputs, response_prefix_lengths, lengths):
    shifted_inputs = inputs[:, :-1]
    shifted_labels = inputs[:, 1:]
    logits = model(shifted_inputs)
    logits = logits.astype(mx.float32)

    mask_width = shifted_inputs.shape[1]
    token_indices = mx.arange(mask_width)[None, :]
    mask = mx.logical_and(
        token_indices >= response_prefix_lengths[:, None],
        token_indices < lengths[:, None],
    )

    ce = nn.losses.cross_entropy(logits, shifted_labels) * mask
    ntoks = mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks

def iterate_completion_batches(
    dataset: ExtendableCompletionsDataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_seq_length: int,
    train: bool = False,
    response_generation_tokens: Optional[List[int]] = None,
):
    """
    A version of iterate_batches that works with completion datasets, tracks the boundaries between input/output tokens
    and returns the lengths of input tokens as well as that of the full sequences.
    """
    idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i]))
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    # If running in distributed mode (N machines) then each one should skip N-1
    # samples
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")
    # Make the batches:
    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            response_prefix_lengths = []
            batch = []
            for j in batch_idx[i]:
                full_sequence = dataset.get_item(j, tokenize=True)
                if full_sequence[-1] != tokenizer.eos_token_id:
                    full_sequence.append(tokenizer.eos_token_id)
                batch.append(full_sequence)
                if len(response_generation_tokens) > 1:
                    response_marker_begin, response_marker_end = contains(
                        response_generation_tokens, full_sequence
                    )
                    response_prefix_lengths.append(response_marker_end + 1)
                else:
                    response_marker_begin = full_sequence.index(
                        response_generation_tokens[0]
                    )
                    response_prefix_lengths.append(response_marker_begin + 1)

            lengths = [len(x) for x in batch]

            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the nearest multiple of 8 or the maximum length
            pad_to = 8
            max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)
            for j in range(batch_size // step):
                response_prefix_length = response_prefix_lengths[j]
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, response_prefix_length:truncated_length] = batch[j][
                    response_prefix_length:truncated_length
                ]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )

            yield mx.array(batch_arr), mx.array(response_prefix_lengths), mx.array(
                lengths
            )

        if not train:
            break

ALL_TRAIN_TYPES = ['lora', 'dora']
DORA_TRAIN_TYPES = ['dora']

@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option("--summary/--no-summary", default=False, help="Just summarize training data")
@click.option('--train-type',
              type=click.Choice(ALL_TRAIN_TYPES, case_sensitive=False),
              default="lora")
@click.option('--mask-inputs/--no-mask-inputs', default=False)
@click.option('--wandb-project', default=None, type=str,
              help='Wandb project name')
@click.option('--wandb-run', default=None, type=str,
              help='Wandb run name')
@click.argument('config_file')
def main(verbose, summary, train_type, mask_inputs, wandb_project, wandb_run, config_file):
    tokenizer_config = {}
    with open(config_file, "r") as file:
        config = yaml.load(file, yaml_loader)
        param_dict = {k: v for k, v in config.items()}
        if "model" not in param_dict:
            raise SyntaxError('Missing required "model" parameter')
        for key, default in CONFIG_DEFAULTS.items():
            if key not in param_dict:
                param_dict[key] = default
        param_dict["verbose"] = verbose
        tokenizer_config = {"trust_remote_code": True if param_dict.get("trust_remote_code") else None}
        param_dict_eos_token = param_dict.get("eos_token")
        if param_dict_eos_token is not None:
            tokenizer_config["eos_token"] = param_dict["eos_token"]
        if verbose:
            pprint(param_dict)
        args = SimpleNamespace(**param_dict)

    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)
    model.freeze()
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
            int(len(valid_set) / ((args.evals_per_epoch - 1) * args.batch_size) if args.evals_per_epoch > 1 else 1)
        )
    else:
        scaled_steps_per_eval = int(num_iterations * args.validation_interval_proportion)
        scaled_val_batches = int(args.validations_per_train_item * args.validation_interval_proportion * num_iterations)

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
    response_generation_tokens = []
    if mask_inputs:
        print("Masking inputs")
        if isinstance(args.response_template, str):
            response_generation_tokens = tokenizer.encode(
                args.response_template, add_special_tokens=False
            )
        elif args.response_template == None:
            raise ValueError("Need to specify 'response_template' in order to be able to mask inputs")
        else:
            if not all([item.isinstance(int) for item in args.response_template]):
                raise ValueError(
                    "Response template must be a list of integers if it is not a string."
                )
            response_generation_tokens = args.response_template
        response_generation_tokens = no_bos_or_eos(response_generation_tokens,
                                                   tokenizer.bos_token_id,
                                                   tokenizer.eos_token_id
                                                   ) if len(response_generation_tokens) > 1 else response_generation_tokens
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
                                  response_generation_tokens=response_generation_tokens,
                                  adapter_file=adapter_file),
                iterate_batches=(
                    iterate_completion_batches if mask_inputs else iterate_batches
                ),
                loss=input_masked_loss if mask_inputs else default_loss,
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
                num_batches=args.test_batches,
                loss=input_masked_loss if mask_inputs else default_loss,
                iterate_batches=(
                    iterate_completion_batches if mask_inputs else iterate_batches
                ),
            )

            test_ppl = math.exp(test_loss)

            print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    else:
        total_num_tokens = 0
        max_tokens = 0
        _lengths = []
        for it, info in zip(
                range(1, num_iterations + 1),
                (iterate_completion_batches if mask_inputs else iterate_batches)(
                    dataset=train_set,
                    tokenizer=tokenizer,
                    batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length,
                    train=False,
                    response_generation_tokens=response_generation_tokens
                )
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
