import yaml
import click
import transformers
import numpy as np

import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn

from copy import deepcopy
from typing import List, Tuple, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path

from mlx_lm.utils import load, save_config, load_adapters
from mlx_lm.tuner.utils import build_schedule
from mlx_lm.tuner.trainer import iterate_batches, TrainingArgs, TrainingCallback, train
from mlx_lm.tuner.datasets import load_dataset, ChatDataset, load_custom_hf_dataset

from mlx_tuning_fork.config import CONFIG_DEFAULTS, yaml_loader

def sigmoid_loss_fn(logits, args):
    return (
            -nn.log_sigmoid(args.beta * logits) * (1 - args.label_smoothing)
            - nn.log_sigmoid(-args.beta * logits) * args.label_smoothing
    )

def ipo_loss_fn(logits, args):
    return (logits - 1 / (2 * args.beta)) ** 2


@dataclass
class DPOArgs:
    label_smoothing: float = field(default=0.0, metadata={"help": "Label smoothing."})
    beta: float = field(default=0.1, metadata={"help": "Beta hyperparameter."})
    dpo_loss_fn: Callable = sigmoid_loss_fn

class PreferenceDataset(ChatDataset):
    """
    Specialization for preference ChatDatasets to provide chat-templated encoding and prompt/completion
    pairs for chosen and rejected preference dataset records
    """
    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)

    def __init__(
        self,
        dataset: ChatDataset,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__(dataset, tokenizer)
        self._tokenizer = tokenizer

    def get_preference_item_with_kwargs(self, idx: int, record_type: str = "chosen", tokenize: bool = False,
                                        add_generation_prompt: bool = True):
        messages = self._data[idx][record_type]
        text = self._tokenizer.apply_chat_template(messages, tokenize=tokenize,
                                                   add_generation_prompt=add_generation_prompt)
        return text

    def get_full_encoding(self, idx: int, record_type: str = "chosen"):
        return self.get_preference_item_with_kwargs(idx, record_type=record_type, tokenize=True)

    def get_prompt_and_completion(self, idx: int, record_type: str = "chosen"):
        prompt = [record for record in self._data[idx][record_type] if record['role'] == 'user'][0]['content']
        completion = [record for record in self._data[idx][record_type] if record['role'] == 'assistant'][0]['content']
        return prompt, completion

    def ordered_indices(self):
        return map(lambda i: i[0],
                   enumerate(sorted(self._data,
                                    key=lambda record: max(len(record["chosen"]), len(record["rejected"])))))

    def __getitem__(self, idx: int):
        return self.get_preference_item_with_kwargs(idx)


def compute_log_probs(model, shifted_inputs, loss_mask, batch_size):
    policy_chosen_logits = model(shifted_inputs).astype(mx.float32)

    #Log probabilities: log-softmax of logits over vocabulary dimension
    per_token_log_probs = mx.log(mx.softmax(policy_chosen_logits, axis=-1))
    # print(per_token_log_probs.shape, loss_mask.shape)

    #Condtional log likelihood of target given the input
    target_token_only_log_probs = mx.where(loss_mask[..., None], per_token_log_probs, 0).sum(axis=-1)
    num_chosen = int(batch_size / 2)
    chosen_log_probs = target_token_only_log_probs[:num_chosen]
    rejected_log_probs = target_token_only_log_probs[num_chosen:]
    return chosen_log_probs, rejected_log_probs


class MLXDirectPreferenceOptimizer:
    def __init__(self, args: DPOArgs, policy: nn.Module, ref_model: nn.Module):
        self.args = args
        self.policy = policy
        self.ref_model = ref_model
        self.metrics = {}

    def input_masked_policy_metrics(self,
                                    model: nn.Module,
                                    inputs: mx.array,
                                    input_lengths: mx.array,
                                    lengths: mx.array):
        """
        Computes policy metrics for input-masked sequences using the specified model
        and compares them to a reference model.

        This function calculates log probabilities for chosen and rejected completions,
        computes rewards and accuracies, and returns losses.

        :param model:
            Neural network model to be used for computing log probabilities.
        :param inputs:
            Input array of sequences.
        :param input_lengths:
            Array containing the lengths of input sequences.
        :param lengths:
            Array containing the full lengths of sequences including completions.
        :return:
            Tuple containing:

            - Mean loss value.
            - Losses normalized by the length of the mask
        """
        shifted_inputs = inputs[:, :-1]
        batch_size = lengths.shape[0]
        completion_mask_width = shifted_inputs.shape[1]
        token_indices = mx.arange(completion_mask_width)[None, :]

        #Mask excluding input (prompt) tokens and suffix padding
        mask = mx.logical_and(token_indices >= input_lengths[:, None], token_indices < lengths[:, None])

        #Get log probabilities of chosen and rejected target tokens using current model
        chosen_log_probs, rejected_log_probs = compute_log_probs(model, shifted_inputs, mask, batch_size)

        # Get log probabilities of chosen and rejected target tokens using reference model
        ref_chosen_log_probs, ref_rejected_log_probs = compute_log_probs(self.ref_model, shifted_inputs, mask,
                                                                         batch_size)
        metrics = {}

        losses = self.args.dpo_loss_fn((chosen_log_probs - rejected_log_probs) -
                                       (ref_chosen_log_probs - ref_rejected_log_probs),
                                       self.args)
        chosen_rewards = self.args.beta * (chosen_log_probs - ref_chosen_log_probs)
        rejected_rewards = self.args.beta * (rejected_log_probs - ref_rejected_log_probs)
        reward_accuracies = chosen_rewards > rejected_rewards

        #@TODO: log/write using training_callback mechanism
        metrics["rewards/chosen"] = chosen_rewards.mean()
        metrics["rewards/rejected"] = rejected_rewards.mean()
        metrics["rewards/accuracies"] = reward_accuracies.mean()
        metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
        metrics["logps/chosen"] = chosen_log_probs.mean()
        metrics["logps/rejected"] = rejected_log_probs.mean()
        print(metrics)
        return losses.mean(), losses.sum() / mask.sum()

    def iterate_input_masked_dpo_batches(self, dataset: PreferenceDataset, tokenizer: transformers.PreTrainedTokenizer,
                                         batch_size: int, max_seq_length: int, train: bool = False,
    ):
        """
        A version of iterate_batches that works with (binarized) preference datasets, managing them as batches of an
        even number of elements, where the first half are 'chat templated', padded encodings of 'chosen' completions
        and the second are of the 'rejected' completions, tracking the boundaries between input/output tokens
        (i.e., prompt/completion, context/continuation, etc.) and returning the lengths of input tokens as well as of
        the full (chosen/rejected) sequences for masking purposes.
        """
        idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i]))

        # If running in distributed mode (N machines) then each one should skip N-1
        # samples
        step = mx.distributed.init().size()
        if batch_size % step != 0:
            raise ValueError("The batch size must be divisible by the number of workers")
        if batch_size % 2 != 0:
            raise ValueError("The batch size must be an even number")

        # Allocate batches at half the size
        batch_idx = [
            idx[i : i + int(batch_size/2) : step]
            for i in range(0, len(idx) - int(batch_size/2) + 1, int(batch_size/2))
        ]
        while True:
            indices = np.random.permutation(len(batch_idx))
            for i in indices:
                prompt_lengths = []
                batch = []
                for j in batch_idx[i]:
                    #Fill in the first and second halves with chosen and rejected completions accordingly
                    for record_type in ["chosen", "rejected"]:
                        prompt, completion = dataset.get_prompt_and_completion(j, record_type=record_type)
                        prompt_lengths.append(input_length(prompt, completion, tokenizer))
                        full_chat_templated_sequence = dataset.get_full_encoding(j, record_type=record_type)
                        if full_chat_templated_sequence[-1] != tokenizer.eos_token_id:
                            full_chat_templated_sequence.append(tokenizer.eos_token_id)
                        batch.append(full_chat_templated_sequence)

                lengths = [len(x) for x in batch]

                # Pad to the nearest multiple of 8 or the maximum length
                pad_to = 8
                max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
                max_length_in_batch = min(max_length_in_batch, max_seq_length)

                batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

                for j in range(batch_size // step):
                    truncated_length = min(lengths[j], max_seq_length)
                    batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                    lengths[j] = (
                        truncated_length  # Update lengths to match truncated lengths
                    )

                yield mx.array(batch_arr), mx.array(prompt_lengths), mx.array(lengths)

            if not train:
                break

DEFAULT_SEED = 0

@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option('--seed', type=int, default=DEFAULT_SEED)
@click.option('-b', '--batch-size', type=int, default=4)
@click.argument('model-name-or-path')
@click.argument('output_dir', default=None)
@click.argument('config_file')
def click_main(verbose, seed, batch_size, model_name_or_path, output_dir, config_file):
    import datasets
    np.random.seed(seed)
    with open(config_file, "r") as file:
        config = yaml.load(file, yaml_loader)
        param_dict = {k: v for k, v in config.items()}
        for key, default in CONFIG_DEFAULTS.items():
            if key not in param_dict:
                param_dict[key] = default
        param_dict["verbose"] = verbose
        tokenizer_config = {"trust_remote_code": True if param_dict.get("trust_remote_code") else None}
        param_dict_eos_token = param_dict.get("eos_token")
        if param_dict_eos_token is not None:
            tokenizer_config["eos_token"] = param_dict["eos_token"]
        args = SimpleNamespace(**param_dict)
    if args.evals_per_epoch % args.batch_size != 0:
        print('WARNING: evals_per_epoch must be divisible by batch_size')
        print('Setting evals_per_epoch to', args.evals_per_epoch - args.evals_per_epoch % args.batch_size)
        args.evals_per_epoch = args.evals_per_epoch - args.evals_per_epoch % args.batch_size

    print('building policy')

    ref_model, tokenizer = load(model_name_or_path, tokenizer_config=tokenizer_config)

    opt = optim.Adam(
        learning_rate=(
            build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
        )
    )
    assert args.lora_parameters["dropout"] == 0.0
    #disable_dropout(policy)

    train_set = PreferenceDataset(datasets.load_dataset(args.hf_dataset["name"],
                                                        split="train",
                                                        **args.hf_dataset.get("config", {})
                                                        ), tokenizer=tokenizer)
    valid_set = PreferenceDataset(datasets.load_dataset(args.hf_dataset["name"],
                                                        split="test",
                                                        **args.hf_dataset.get("config", {})
                                                        ), tokenizer=tokenizer)
    epoch_num_steps = (len(train_set) + args.batch_size - 1) // args.batch_size
    if args.epochs == -1:
        num_iterations = epoch_num_steps if args.iters == -1 else args.iters
    else:
        num_iterations = epoch_num_steps * args.epochs
    num_iterations = int(num_iterations)

    print(
        f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
        f"{len(train_set):,} records, {args.batch_size} at a time and with a validation set of "
        f"{len(valid_set):,} records, training {args.num_layers} layers out of {len(policy.layers)}"
    )

    if args.evals_per_epoch:
        scaled_steps_per_eval = int(epoch_num_steps / args.evals_per_epoch)
        scaled_val_batches = int(len(valid_set) * args.eval_proportion_of_total / args.batch_size
                                 ) if args.eval_proportion_of_total else (
            int(len(valid_set) / ((args.evals_per_epoch - 1) * args.batch_size))
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

    adapter_path = Path(output_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)
    save_config(vars(args), adapter_path / "adapter_config.json")
    adapter_file = adapter_path / "adapters.safetensors"

    policy = load_adapters(deepcopy(ref_model), output_dir)
    policy.freeze()

    dpo_args = DPOArgs()
    training_args = TrainingArgs(
        batch_size=batch_size,
        iters=num_iterations,
        val_batches=scaled_val_batches,
        steps_per_report=scaled_steps_per_report,
        steps_per_eval=scaled_steps_per_eval,
        steps_per_save=scaled_save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length
    )

    dpo_trainer = MLXDirectPreferenceOptimizer(dpo_args, policy, ref_model)

    print("Training")
    policy.train()
    train(
        policy,
        tokenizer,
        opt,
        train_set,
        valid_set,
        args=training_args,
        loss=dpo_trainer.input_masked_policy_metrics,
        iterate_batches=dpo_trainer.iterate_input_masked_dpo_batches,
        # training_callback=training_callback
    )


def contains(small_list: List, big_list: List) -> Tuple[int, int]:
    """
    Returns the beginning and end index of the first occurrence of small_list in big_list.
    """
    small_list_length = len(small_list)
    for ind in (i for i, e in enumerate(big_list) if e == small_list[0]):
        if big_list[ind : ind + small_list_length] == small_list:
            return ind, ind + small_list_length - 1


def no_bos(sequence: List, bos: int) -> List:
    return sequence if sequence[0] != bos else sequence[1:]


def input_length(
    input_text: str, output_text: str, tokenizer: transformers.PreTrainedTokenizer
) -> int:
    """
    Returns the length of the portion of the encoding of the concatenation of input_text and output_text
    that corresponds to the input tokens.
    """
    message = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    output_tokens = no_bos(tokenizer.encode(output_text), tokenizer.bos_token_id)
    full_sequence = tokenizer.apply_chat_template(
        message, add_generation_prompt=True, tokenize=True
    )
    output_begin, output_end = contains(output_tokens, full_sequence)
    return output_begin



if __name__ == '__main__':
    click_main()
