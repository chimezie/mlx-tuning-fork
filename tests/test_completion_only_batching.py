import pytest
import os
import mlx.core as mx
from mlx_lm.utils import load
import numpy as np

from mlx_tuning_fork.tuning.utils import create_delineated_batches


class TestCreateDelineatedBatches:
    prompt = 'My name is: '
    output_text = ['Foo', 'Mandalorian', 'San the country with no name']

    def setup_method(self):
        _, self.tokenizer = load(os.environ.get("MLX_MODEL_PATH"))
        prompt_ids = self.tokenizer.encode(self.prompt)
        self.tokenized_prompt_length = len(prompt_ids)
        self.input_text = [self.prompt] * len(self.output_text)

    def _run_test(self, max_seq_length, expected_length):
        batch, input_lengths, lengths = create_delineated_batches(self.input_text, self.output_text, self.tokenizer,
                                                                  max_seq_length=max_seq_length)
        assert all([len(i) == expected_length for i in batch])
        assert mx.all(input_lengths == self.tokenized_prompt_length).item()
        for idx, output in enumerate(self.output_text):
            batch_item = batch[idx]
            output_ids = self.tokenizer.encode(output)
            tokenized_output_length = len(output_ids)
            non_padding_length = min(expected_length, tokenized_output_length + self.tokenized_prompt_length)

            # All padding tokens are 0
            assert np.all(np.take(batch_item, np.arange(non_padding_length, len(batch_item))) == 0)

            #All non-padding tokens are nonzero
            assert np.all(np.take(batch_item, np.arange(non_padding_length)) != 0)

    def test_basic_no_truncation(self):
        self._run_test(20, 16)

    def test_basic_w_truncation(self):
        self._run_test(10, 10)
