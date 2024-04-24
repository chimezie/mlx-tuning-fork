import numpy as np
from mlx import core as mx


def create_delineated_batches(input_text, output_text, tokenizer, max_seq_length=2048):
    batch_size = len(input_text)

    input_batch = [tokenizer.encode(record) for record in input_text]
    output_batch = [tokenizer.encode(record) for record in output_text]

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
    input_lengths = mx.array(input_lengths)
    lengths = mx.array(adjusted_lengths)

    return batch, input_lengths, lengths
