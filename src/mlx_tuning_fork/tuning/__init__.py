from mlx_lm.tuner.datasets import CompletionsDataset, ChatDataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Callable, Union

class ExtendableCompletionsDataset(CompletionsDataset):
    """
    A version of CompletionsDataset that can return its prompt and completion separately as well as
    return an item in combinations of returned as string or tokens and with or without a generation prompt.
    """
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
    ):
        super().__init__(data, tokenizer, prompt_key, completion_key)

    def get_prompt_and_completion(self, idx: int):
        return self._data[idx][self._prompt_key], self._data[idx][self._completion_key]

    def get_item(
        self, idx: int, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        prompt, completion = self.get_prompt_and_completion(idx)
        return self._tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

class ExtendableChatDataset(ChatDataset):
    """
    A version of ChatDataset whose chat key can be specified
    """
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        chat_key: str = "messages",
    ):
        super().__init__(data)
        self._tokenizer = tokenizer
        self._chat_key = chat_key

    def __getitem__(self, idx: int):
        messages = self._data[idx][self._chat_key]
        text = self._tokenizer.apply_chat_template(
            messages,
            tools=self._data[idx].get("tools", None),
            tokenize=False,
            add_generation_prompt=True,
        )
        return text

class CompletionsDatasetCollection:
    """
    Extension to support collections of HF completion datasets
    """
    def __init__(self, data: List[Union[ChatDataset, CompletionsDataset]]):
        self.collection = data

    def __fetch_and_process_item__(self, idx: int, handler_fn: Callable):
        iteration = iter(self.collection)
        item = next(iteration)

        curr_idx = idx

        while True:
            try:
                if (curr_idx + 1) <= len(item):
                    return handler_fn(item, curr_idx)
                else:
                    curr_idx -= len(item)
                    item = next(iteration)
            except StopIteration:
                raise IndexError(idx)

    def __getitem__(self, idx: int):
        def getitem(dataset: CompletionsDataset, index: int):
            return dataset[index]

        return self.__fetch_and_process_item__(idx, getitem)

    def get_item(
        self, idx: int, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        def getitem(dataset: CompletionsDataset, index: int):
            return dataset.get_item(index, tokenize, add_generation_prompt)

        return self.__fetch_and_process_item__(idx, getitem)

    def get_prompt_and_completion(self, idx: int):
        def getitem(dataset: CompletionsDataset, index: int):
            return dataset.get_prompt_and_completion(index)

        return self.__fetch_and_process_item__(idx, getitem)

    def __len__(self):
        return sum(map(len, self.collection))
