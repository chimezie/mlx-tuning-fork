from typing import List, Tuple


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
