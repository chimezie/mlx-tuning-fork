import os
import mlx.core as mx
from mlx_lm.utils import load

from mlx_tuning_fork.tuning.utils import create_delineated_batches


class TestCreateDelineatedBatches:
    def test_basic_no_truncation(self):
        model, tokenizer = load(os.environ.get("MLX_MODEL_PATH"))
        prompt = 'My name is: '
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        tokenized_prompt_length = len(prompt_ids)
        output_text = ['Foo', 'Mandalorian', 'San The country with no name']
        input_text = [prompt] * len(output_text)
        batch, input_lengths, lengths = create_delineated_batches(input_text, output_text, tokenizer, max_seq_length=20)
        assert all([len(i) == 16 for i in batch])
        assert all([i == tokenized_prompt_length for i in input_lengths])
        for idx, output in enumerate(output_text):
            batch_item = batch[idx]
            output_ids = tokenizer.encode(output, add_special_tokens=False)
            tokenized_output_length = len(output_ids)
            non_padding_length = tokenized_output_length + tokenized_prompt_length
            padding = batch_item[non_padding_length:]
            expected_padding = mx.array([0] * (16 - non_padding_length))
            assert mx.array_equal(padding, expected_padding)

    def test_basic_w_truncation(self):
        model, tokenizer = load(os.environ.get("MLX_MODEL_PATH"))
        prompt = 'My name is: '
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        tokenized_prompt_length = len(prompt_ids)
        output_text = ['Foo', 'Mandalorian', 'San The country with no name']
        input_text = [prompt] * len(output_text)
        batch, input_lengths, lengths = create_delineated_batches(input_text, output_text, tokenizer, max_seq_length=10)
        assert all([len(i) == 10 for i in batch])
        assert all([i == tokenized_prompt_length for i in input_lengths])
        for idx, output in enumerate(output_text):
            batch_item = batch[idx]
            output_ids = tokenizer.encode(output, add_special_tokens=False)
            tokenized_output_length = len(output_ids)
            non_padding_length = tokenized_output_length + tokenized_prompt_length
            padding = batch_item[non_padding_length:]
            expected_padding = mx.array([0] * (10 - non_padding_length))
            assert mx.array_equal(padding, expected_padding)

