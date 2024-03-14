import warnings

import mlx.optimizers as optim
import numpy as np
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, evaluate, train
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load, generate
from mlx_lm.generate import colorprint_by_t0
from mlx_lm import lora
from types import SimpleNamespace
import mlx.core as mx
from tqdm import tqdm
import mlx.nn as nn
import click
import yaml
import math
from mlx_tuning_fork.dataset import Dataset
from mlx_tuning_fork.config import CONFIG_DEFAULTS, yaml_loader
from mlx_tuning_fork.reporting import WandbCallback
from mlx_tuning_fork.tuning.dynamic_learning import SCHEDULE_CONFIGURATION_TYPE_TO_CLASS
from ogbujipt import word_loom
from ogbujipt.prompting import format
from pathlib import Path
from pprint import pprint


def completions_only_loss(model, inputs, input_lengths, lengths):
    shifted_inputs = inputs[:, :-1]
    shifted_labels = inputs[:, 1:]
    logits, _ = model(shifted_inputs)
    logits = logits.astype(mx.float32)

    mask_width = shifted_inputs.shape[1]
    token_indices = mx.arange(mask_width)[None, :]
    mask = mx.logical_and(token_indices >= input_lengths[:, None], token_indices < lengths[:, None])

    ce = nn.losses.cross_entropy(logits, shifted_labels) * mask
    ntoks = mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def completions_only_iterate_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    while True:
        indices = np.random.permutation(np.arange(len(dataset)))
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            input_text = []
            output_text = []
            for j in range(batch_size):
                record = dataset[indices[i + j]]
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

            max_width = min(max(lengths), max_seq_length)

            batch_arr = np.zeros((batch_size, max_width), np.int32)
            adjusted_lengths = []
            for j in range(batch_size):
                input_length = input_lengths[j]
                full_ids_end_idx = input_length + min(output_lengths[j], max_width - input_length)
                adjusted_lengths.append(full_ids_end_idx)
                batch_arr[j, :full_ids_end_idx] = full_labels[j][:full_ids_end_idx]
            batch = mx.array(batch_arr)
            if train:
                pbar.update(1)
            yield batch, mx.array(input_lengths), mx.array(adjusted_lengths)

        if not train:
            break


def iterate_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    while True:
        # Shuffle indices
        indices = np.arange(len(dataset))
        indices = np.random.permutation(indices)
        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [
                tokenizer.encode(
                    prompt_formatter.get_input(dataset[indices[i + j]]) +
                    prompt_formatter.get_output(dataset[indices[i + j]])
                ) for j in range(batch_size)
            ]
            lengths = [len(x) for x in batch]

            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            max_length_in_batch = min(max(lengths), max_seq_length)
            batch_arr = np.zeros((batch_size, max_length_in_batch), np.int32)

            for j in range(batch_size):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )
            batch = mx.array(batch_arr)

            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def generate_prompt_from_loom(loom_file, loom_markers, prompt_formatter, build_prompt):
    with open(loom_file, mode='rb') as fp:
        loom = word_loom.load(fp)
        if build_prompt is not None:
            loom_sections = build_prompt.split(' ')
            num_loom_sections = len(loom_sections)
            if num_loom_sections not in [1, 2, 3]:
                raise click.BadParameter("Expected: 1-3 loom section names separated by space")
            elif num_loom_sections == 1:
                system_section = 'system_prompt'
                extra_context_section = 'context'
                question_section = loom_sections[0]
            elif num_loom_sections == 2:
                system_section = loom_sections[0]
                extra_context_section = 'context'
                question_section = loom_sections[1]
            else:
                system_section = loom_sections[0]
                extra_context_section = loom_sections[1]
                question_section = loom_sections[2]
        else:
            system_section = 'system_prompt'
            extra_context_section = 'context'
            question_section = 'question'
        question = loom[question_section]
        system = loom.get(system_section, '')
        extra_context = loom.get(extra_context_section, '')
        if loom_markers is not None:
            marker, value = loom_markers.split('=')
            question = question.format(**{marker: value})
        return format(question, preamble=system, contexts=extra_context, delimiters=prompt_formatter.get_delimiters())

@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option("--summary/--no-summary", default=False, help="Just summarize training data")
@click.option("--loom-file", help="An OgbujiPT word loom file to use for prompt construction")
@click.option("--loom-markers", help="Loom marker values", default=None, type=str)
@click.option('-p', '--prompt', default=None, type=str,
              help='Commandline prompt (overrides) prompt in YAML configuration')
@click.option('-t', '--temperature', default=None, type=float,
              help='Prompt generation temperature')
@click.option('-nt', '--num-tokens', default=-1, type=int,
              help='Overide number of tokens in config file')
@click.option('--train-type',
              type=click.Choice(['completion-only', 'self-supervised'], case_sensitive=False),
              default="completion-only")
@click.option('-f', '--prompt-format',
              type=click.Choice(['mistral', 'chatml'], case_sensitive=False))
@click.option('-a', '--adapter', default=None, type=str,
              help='Adapter to use instead of the one specified in the config file')
@click.option('--wandb-project', default=None, type=str,
              help='Wandb project name')
@click.option('--wandb-run', default=None, type=str,
              help='Wandb run name')
@click.option('-rp', '--repetition-penalty', default=0, type=float,
              help='The penalty factor for repeating tokens (none if not used)')
@click.option('--repetition-context-size', default=20, type=int,
              help='The number of tokens to consider for repetition penalty')
@click.option('-tp', '--top-p', default=CONFIG_DEFAULTS["top_p"], type=float,
              help='Sampling top-p')
@click.option('--build-prompt', default=None, type=str,
              help='Which word loom sections to use in building the claim (space-separated list of sections)')
@click.argument('config_file')
def main(verbose, summary, loom_file, loom_markers, prompt, temperature, num_tokens, train_type, prompt_format, adapter,
         wandb_project, wandb_run, repetition_penalty, repetition_context_size, top_p, config_file,
         build_prompt):
    global pbar, prompt_formatter
    if prompt_format == 'mistral':
        from mlx_tuning_fork.prompt_templates.mistral import TrainingRecordHandler
        prompt_formatter = TrainingRecordHandler
    elif prompt_format == 'chatml':
        from mlx_tuning_fork.prompt_templates.chatml import TrainingRecordHandler
        prompt_formatter = TrainingRecordHandler

    lora.Dataset = Dataset

    with open(config_file, "r") as file:
        config = yaml.load(file, yaml_loader)
        param_dict = {k: v for k, v in config.items()}
        if "model" not in param_dict:
            raise SyntaxError('Missing required "model" parameter')
        for key, default in CONFIG_DEFAULTS.items():
            if key not in param_dict:
                param_dict[key] = default
        param_dict["verbose"] = verbose
        if loom_file:
            param_dict["prompt"] = generate_prompt_from_loom(loom_file, loom_markers, prompt_formatter, build_prompt)
            param_dict["test"] = param_dict["train"] = False
            param_dict["ignore_chat_template"] = True
        if prompt:
            param_dict["prompt"] = prompt
            param_dict["test"] = param_dict["train"] = False
        if temperature:
            param_dict["temperature"] = temperature
        if adapter:
            param_dict["adapter_file"] = adapter
        if num_tokens and num_tokens != -1:
            param_dict["num_tokens"] = num_tokens
        pprint(param_dict)
        args = SimpleNamespace(**param_dict)

    print("Loading pretrained model")
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    # if args.eos_token is not None:
    #     tokenizer_config["eos_token"] = args.eos_token
    model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)
    model.freeze()
    # Convert linear layers to lora layers and unfreeze in the process
    linear_to_lora_layers(model, args.lora_layers, args.lora_parameters)

    training_callback = None
    if wandb_project:
        if wandb_run is None:
            raise RuntimeError("Specify the name of a Wandb run to use with --wandb-run ")
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        wandb.init(project=wandb_project, name=wandb_run, config=config)
        training_callback = WandbCallback()

    print("Loading datasets")
    names = ("train", "valid", "test")
    train_set, valid_set, test_set = (lora.Dataset(Path(args.data) / f"{n}.jsonl") for n in names)
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

    print(
        f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
        f"{len(train_set):,} records, {args.batch_size} at a time and with a validation set of "
        f"{len(valid_set):,} records, training {args.lora_layers} layers out of {len(model.model.layers)} using qLoRa."
    )

    scaled_steps_per_report = int(args.reporting_interval_proportion * num_iterations)
    scaled_steps_per_eval = int(num_iterations * args.validation_interval_proportion)
    scaled_val_batches = int(args.validations_per_train_item * args.validation_interval_proportion * num_iterations)
    scaled_save_every = int(args.adapter_save_interval_proportion * num_iterations)

    print(
        f"Calculating loss every {scaled_steps_per_report:,} steps, reporting validation loss every "
        f"{scaled_steps_per_eval:,} steps, validating with {scaled_val_batches:,} batches, and saving the "
        f"adapter every {scaled_save_every:,} steps."
    )

    if not summary:

        if "learning_schedule" in config:
            scheduler = SCHEDULE_CONFIGURATION_TYPE_TO_CLASS[
                config["learning_schedule"]["type"]].from_configuration(args.learning_rate, config, num_iterations)
        else:
            scheduler = args.learning_rate

        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=num_iterations,
            val_batches=scaled_val_batches,
            steps_per_report=scaled_steps_per_report,
            steps_per_eval=scaled_steps_per_eval,
            steps_per_save=scaled_save_every,
            adapter_file=args.adapter_file,
            max_seq_length=args.max_seq_length
        )

        if args.train:
            print("Training")
            model.train()
            opt = optim.Adam(learning_rate=scheduler)
            pbar = tqdm(total=num_iterations)

            train(
                model,
                tokenizer,
                opt,
                train_set,
                valid_set,
                args=training_args,
                loss=completions_only_loss if train_type == 'completion-only' else default_loss,
                iterate_batches=completions_only_iterate_batches if train_type == 'completion-only'
                else iterate_batches,
                training_callback=training_callback
            )

        # Load the LoRA adapter weights which we assume should exist by this point
        if not Path(args.adapter_file).is_file():
            raise ValueError(
                f"Adapter file {args.adapter_file} missing. "
            )
        model.load_weights(args.adapter_file, strict=False)
        print(f"Loaded weights from {args.adapter_file}")

        if args.test:
            print(f"Testing ({len(test_set):,} records)")
            model.eval()

            test_loss = evaluate(
                model=model,
                dataset=test_set,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.test_batches,
                loss=completions_only_loss,
                iterate_batches=completions_only_iterate_batches
            )

            test_ppl = math.exp(test_loss)

            print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

        if args.prompt is not None:
            print("Generating")
            model.eval()

            if not args.ignore_chat_template and (
                    hasattr(tokenizer, "apply_chat_template")
                    and tokenizer.chat_template is not None
            ):
                messages = [{"role": "user", "content": args.prompt}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = args.prompt

            formatter = colorprint_by_t0 if args.colorize else None

            generate(
                model, tokenizer, prompt, args.temp, args.max_tokens, True, formatter=formatter,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                top_p=top_p
            )


if __name__ == '__main__':
    main()
