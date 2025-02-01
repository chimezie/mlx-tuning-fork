from typing import List, Optional

import numpy as np
from mlx import core as mx, nn as nn
from mlx_lm.tuner.datasets import CompletionsDataset
from transformers import PreTrainedTokenizer

from .utils import contains


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

class InputMasker:
    """
    A class for producing input_masked batches for training, needed only because
    the MLX training mechanism does not provide a way to pass down additional, custom state information
    to iterate_batches functions to allow them to receive information about how to distinguish
    between input tokens and the rest of the sequence for arbitrary prompt formats and tokenizers.
    """
    def __init__(self, response_generation_tokens: Optional[List[int]] = None, pad_to: Optional[int] = 8):
        self.response_generation_tokens = response_generation_tokens
        self.pad_to = pad_to

    def iterate_completion_batches(self,
                                   dataset: CompletionsDataset,
                                   tokenizer: PreTrainedTokenizer,
                                   batch_size: int,
                                   max_seq_length: int,
                                   train: bool = False):
        """
        A version of mlx_lm.tuner.trainer.iterate_batches that works with completion datasets, tracks the boundaries between
        input/output tokens and returns the lengths of input tokens as well as that of the full sequences.
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
                    full_sequence = dataset[j]
                    batch.append(full_sequence)
                    if len(self.response_generation_tokens) > 1:
                        response_marker_begin, response_marker_end = contains(
                            self.response_generation_tokens, full_sequence
                        )
                        response_prefix_lengths.append(response_marker_end + 1)
                    else:
                        response_marker_begin = full_sequence.index(
                            self.response_generation_tokens[0]
                        )
                        response_prefix_lengths.append(response_marker_begin + 1)

                lengths = [len(x) for x in batch]
                if max(lengths) > max_seq_length:
                    print(
                        f"[WARNING] Some sequences (out of {len(batch)}) are longer than {max_seq_length} tokens. "
                        f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                        "Consider pre-splitting your data to save memory."
                    )

                if max(lengths) > max_seq_length:
                    print(
                        f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                        f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                        "Consider pre-splitting your data to save memory."
                    )

                # Pad to the nearest multiple of 8 or the maximum length
                pad_to = self.pad_to
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

                yield mx.array(batch_arr), mx.array(response_prefix_lengths), mx.array(lengths)

            if not train:
                break
