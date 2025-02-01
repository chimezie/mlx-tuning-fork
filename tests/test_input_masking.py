import pytest
from mlx_lm.tuner.datasets import CompletionsDataset
from mlx_tuning_fork.tuning.input_masking import InputMasker

GEMMA_RESPONSE_GENERATION_TOKENS = [107, 108, 106, 2516]

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 108

class MockDataset(CompletionsDataset):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx: int):
        return self._data[idx]

test_messages = [
    (
        ("Finish the lyrics: It ain't hard to tell, I excel, then prevail",
         "The mic is contacted, I attract clientele"),
        [2, 106, 1645, 108, 30695, 573, 23658, 235292, 1165, 11032, 235303, 235251, 2811, 577, 3337, 235269, 590, 17349,
         235269, 1492, 58444, 107, 108, 106, 2516, 108, 651, 4164, 603, 35725, 235269, 590, 26827, 130786, 107, 108,
         ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 651, 4164, 603, 35725, 235269,
         590, 26827, 130786, 107, 108]
    ),
    (
        ("Finish the lyrics: This rhythmatic explosion", "Is what your frame of mind has chosen"),
        [2, 106, 1645, 108, 30695, 573, 23658, 235292, 1417, 25259, 5560, 31070, 107, 108, 106, 2516, 108, 2437, 1212,
         861, 5265, 576, 3403, 919, 12228, 107, 108],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 2437, 1212, 861, 5265, 576, 3403, 919, 12228, 107, 108]
    ),
    (
        ("Finish the lyrics: Speak with criminal slang, begin like a violin",
         "End like Leviathan, it's deep? Well, let me try again"),
        [2, 106, 1645, 108, 30695, 573, 23658, 235292, 63707, 675, 15412, 89618, 235269, 3293, 1154, 476, 47244,
         107, 108, 106, 2516, 108, 3898, 1154, 205552, 235269, 665, 235303, 235256, 5271, 235336, 6775, 235269, 2142,
         682, 3418, 1653, 107, 108],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 3898, 1154, 205552, 235269, 665, 235303,
         235256, 5271, 235336, 6775, 235269, 2142, 682, 3418, 1653, 107, 108]
    ),
    (
        ("Finish the lyrics: MCs eavesdrop, though they need not to sneak",
         "My poetry's deep, I never fell"),
        [2, 106, 1645, 108, 30695, 573, 23658, 235292, 18621, 235256, 137083, 9188, 235269, 2862, 984, 1476, 780, 577,
         64381, 107, 108, 106, 2516, 108, 2926, 21181, 235303, 235256, 5271, 235269, 590, 2447, 10652, 107, 108],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 2926, 21181, 235303, 235256, 5271,
         235269, 590, 2447, 10652, 107, 108]
    ),
    (
        ("Finish the lyrics: I can't call it, the beats make me fallin' asleep",
         "I keep fallin', but never fallin' six feet deep"),
        [2, 106, 1645, 108, 30695, 573, 23658, 235292, 590, 798, 235303, 235251, 2409, 665, 235269, 573, 37308, 1501,
         682, 3881, 473, 235303, 30702, 107, 108, 106, 2516, 108, 235285, 2745, 3881, 473, 920, 901, 2447, 3881, 473,
         235303, 4442, 5368, 5271, 107, 108],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 235285, 2745, 3881, 473,
         920, 901, 2447, 3881, 473, 235303, 4442, 5368, 5271, 107, 108]
    ),
    (
        ("Finish the lyrics: I sip the Dom P, watchin' Gandhi 'til I'm charged, then",
         "Writin' in my book of rhymes, all the words past the margin"),
        [2, 106, 1645, 108, 30695, 573, 23658, 235292, 590, 58980, 573, 12850, 596, 235269, 4234, 473, 235303, 43107,
         777, 1136, 590, 235303, 235262, 12497, 235269, 1492, 107, 108, 106, 2516, 108, 15928, 473, 235303, 575, 970,
         2870, 576, 105878, 235269, 832, 573, 3907, 3433, 573, 10176, 107, 108],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, 15928, 473,
         235303, 575, 970, 2870, 576, 105878, 235269, 832, 573, 3907, 3433, 573, 10176, 107, 108]
    )
]

def count_leading_zeros(numbers: list[int]) -> int:
    count = 0
    for value in numbers:
        if value == 0:
            count += 1
        else:
            break
    return count

def remove_trailing_zeros(nums):
    """
    Removes any trailing sequence of zeros from the list of integers.

    :param nums: List of integers
    :return: A sublist with trailing zeros removed
    """
    # Start iterating from the end, and find the first non-zero element
    for i in reversed(range(len(nums))):
        if nums[i] != 0:
            # Return a sublist up to this point
            return nums[:i + 1]
    # If the list contains only zeros, return an empty list
    return []

class TestInputMasking:
    def test_something(self):
        ds = []
        post_processing_output = {}
        for text, tokenization, input_masked_padded in test_messages:
            length = len(tokenization)
            ds.append(tokenization)
            response_prefix_length = remove_trailing_zeros(tokenization)
            post_processing_output[tuple(input_masked_padded)] = (response_prefix_length, text, length)
        ds = MockDataset(ds)
        masker = InputMasker(GEMMA_RESPONSE_GENERATION_TOKENS)
        for inputs, response_prefix_lengths, lengths in masker.iterate_completion_batches(ds,
                                                                                          MockTokenizer(),
                                                                                          2,
                                                                                          2048):
            for idx, item in enumerate([*inputs]):
                as_list = item.tolist()
                as_list_no_r_padding = remove_trailing_zeros(as_list)
                info = post_processing_output.get(tuple(as_list_no_r_padding))
                assert info is not None
                assert response_prefix_lengths[idx].item() == count_leading_zeros(as_list)
                assert lengths[idx] == info[2]
