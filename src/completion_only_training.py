import mlx.optimizers as optim
import numpy as np
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
from mlx_lm.utils import generate, load
from mlx_lm import lora
from types import SimpleNamespace
import mlx.core as mx
from tqdm import tqdm
import mlx.nn as nn
import click
import re
import math
from .dataset import Dataset
from .config import CONFIG_DEFAULTS
import prompt_templates
import yaml
import json
import csv
from pathlib import Path
from pprint import pprint

#https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
All = {'one': 1, 'low': 0.000001}

jAll = json.dumps(All)

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


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

            max_width = max(lengths)
            assert max_width < 2048

            batch_arr = np.zeros((batch_size, max_width), np.int32)
            for j in range(batch_size):
                input_length = input_lengths[j]
                full_ids_end_idx = input_length + output_lengths[j]
                batch_arr[j, :full_ids_end_idx] = full_labels[j][:full_ids_end_idx]
            batch = mx.array(batch_arr)
            if train:
                pbar.update(1)
            yield batch, mx.array(input_lengths), mx.array(lengths)

        if not train:
            break


@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option("--summary/--no-summary", default=False, help="Just summarize training data")
@click.option('--prompt', default=None, type=str,
              help='Commandline prompt (overrides) prompt in YAML configuration')
@click.option('--prompt-format',
              type=click.Choice(['mistral'], case_sensitive=False))
@click.argument('filename', help="The YAML confguration file")
def main(verbose, summary, prompt, prompt_format, filename):
    global pbar, prompt_formatter
    prompt_formatter = getattr(prompt_templates, prompt_format).TrainingRecordHandler

    lora.Dataset = Dataset

    with open(filename, "r") as file:
        config = yaml.load(file, loader)
        param_dict = {k: v for k, v in config["parameters"].items()}
        if "model" not in param_dict:
            raise SyntaxError('Missing required "model" parameter')
        for key, default in CONFIG_DEFAULTS.items():
            if key not in param_dict:
                param_dict[key] = default
        param_dict["dataset_summary"] = summary
        param_dict["verbose"] = verbose
        if prompt:
            param_dict["prompt"] = prompt
        pprint(param_dict)
        args = SimpleNamespace(**param_dict)

    print("Loading pretrained model")
    model, tokenizer = load('/Users/oori/medical_llm/raw_models/mlx/')

    model.freeze()
    for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    print("Loading datasets")
    train_set, valid_set, test_set = lora.load_dataset(args)

    epoch_num_steps = (len(train_set) + args.batch_size - 1) // args.batch_size
    if args.epochs == -1:
        num_iterations = epoch_num_steps if args.iters == -1 else args.iters
    else:
        num_iterations = epoch_num_steps * args.epochs

    pbar = tqdm(total=num_iterations)

    print(
        f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
        f"{len(train_set):,} records, {args.batch_size} at a time and with a validation set of "
        f"{len(valid_set):,} records, training {args.lora_layers} out of {len(model.model.layers)} using qLoRa "
    )

    scaled_steps_per_report = int(10 * (num_iterations) / 1000)
    scaled_steps_per_eval = int(num_iterations * 200 / 1000)
    scaled_val_batches = int((scaled_steps_per_eval * 3 * len(valid_set)) / (2 * args.batch_size * num_iterations))
    scaled_save_every = int(scaled_steps_per_eval / 2)

    print(
        f"Calculating loss every {scaled_steps_per_report:,} steps, reporting validation loss every "
        f"{scaled_steps_per_eval:,} steps, validating with {scaled_val_batches:,} batches, and saving the "
        f"adapter every {scaled_save_every:} steps."
    )

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=num_iterations,
        val_batches=scaled_val_batches,
        steps_per_report=scaled_steps_per_report,
        steps_per_eval=scaled_steps_per_eval,
        steps_per_save=scaled_save_every,
        adapter_file=args.adapter_file,
    )

    if args.train:
        print("Training")
        model.train()
        opt = optim.Adam(learning_rate=args.learning_rate)
        train_loss = []
        validation_loss = []
        train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            loss=completions_only_loss,
            iterate_batches=completions_only_iterate_batches,
            reported_train_loss_data=train_loss,
            validation_loss_data=validation_loss
        )
        if args.train_loss_file:
            with Path(args.train_loss_file).open('a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerows(train_loss)
            print(f"Wrote loss data to {args.train_loss_file}")
        if args.validation_loss_file:
            with Path(args.validation_loss_file).open('a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerows(validation_loss)
            print(f"Wrote loss data to {args.validation_loss_file}")

    # Load the LoRA adapter weights which we assume should exist by this point
    if not Path(args.adapter_file).is_file():
        raise ValueError(
            f"Adapter file {args.adapter_file} missing. "
            "Use --train to learn and save the adapters.npz."
        )
    model.load_weights(args.adapter_file, strict=False)

    if args.test:
        print("Testing")
        model.eval()

        test_loss = evaluate(
            model=model,
            dataset=test_set,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
        )

        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if args.prompt is not None:
        print("Generating")
        model.eval()

        tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
        if args.eos_token is not None:
            tokenizer_config["eos_token"] = args.eos_token

        if not args.ignore_chat_template and (
                hasattr(tokenizer, "apply_chat_template")
                and tokenizer.chat_template is not None
        ):
            messages = [{"role": "user", "content": args.prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = args.prompt

        formatter = generate.colorprint_by_t0 if args.colorize else None

        generate(
            model, tokenizer, prompt, args.temp, args.max_tokens, True, formatter=formatter
        )


if __name__ == '__main__':
    main()
