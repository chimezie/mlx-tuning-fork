from pathlib import Path
from mlx_lm.tuner.datasets import Dataset, ChatDataset, CompletionsDataset, load_local_dataset, load_hf_dataset
from typing import List, Union, Callable
from transformers import PreTrainedTokenizer

class CompletionsDatasetCollection:
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

    def __len__(self):
        return sum(map(len, self.collection))

def load_custom_hf_dataset(args, tokenizer: PreTrainedTokenizer):
    """
    Updated from mlx_lm.tuner.datasets.load_hf_dataset to support hf_datasets and colleciton
    of datasets (https://github.com/ml-explore/mlx-examples/pull/1103)
    """
    import datasets

    def create_hf_dataset(
        dataset_name: Union[None, str],
        text_feature: Union[None, str],
        prompt_feature: Union[None, str],
        completion_feature: Union[None, str],
        chat_feature: Union[None, str],
        split: str = None,
    ):
        ds = datasets.load_dataset(
            dataset_name,
            split=split,
            **hf_args.get("config", {}),
        )
        if prompt_feature and completion_feature:
            return CompletionsDataset(ds, tokenizer, prompt_feature, completion_feature)
        elif chat_feature:
            return ChatDataset(ds, tokenizer, chat_key=chat_feature)
        elif text_feature:
            return Dataset(ds, text_key=text_feature)
        else:
            raise ValueError(
                "Specify either a prompt and completion feature or a text "
                "feature for the Hugging Face dataset."
            )

    def get_hf_custom_features(hf_args):
        return (
            hf_args.get("text_feature"),
            hf_args.get("prompt_feature"),
            hf_args.get("completion_feature"),
            hf_args.get("chat_feature"),
        )

    def get_train_and_valid_splits(hf_args, ds_name):
        train_split = hf_args.get("train_split", "train[:80%]")
        valid_split = hf_args.get("valid_split", "train[-10%:]")
        text_f, prompt_f, completion_f, chat_f = get_hf_custom_features(hf_args)
        train = create_hf_dataset(
            dataset_name=ds_name,
            text_feature=text_f,
            prompt_feature=prompt_f,
            completion_feature=completion_f,
            chat_feature=chat_f,
            split=train_split,
        )
        valid = create_hf_dataset(
            dataset_name=ds_name,
            text_feature=text_f,
            prompt_feature=prompt_f,
            completion_feature=completion_f,
            chat_feature=chat_f,
            split=valid_split,
        )
        return train, valid

    if args.hf_datasets:
        dataset_collection = args.hf_datasets
        train_collection = []
        valid_collection = []
        test_collection = []
        for ds in dataset_collection:
            hf_args = ds["hf_dataset"]
            dataset_name = hf_args["name"]
            print(f"Loading Hugging Face dataset {dataset_name}.")
            text_feature, prompt_feature, completion_feature, chat_f = (
                get_hf_custom_features(hf_args)
            )
            if args.train:
                train, valid = get_train_and_valid_splits(hf_args, dataset_name)
            else:
                train, valid = [], []
            if args.test:
                test = create_hf_dataset(
                    dataset_name=dataset_name,
                    text_feature=text_feature,
                    prompt_feature=prompt_feature,
                    completion_feature=completion_feature,
                    chat_feature=chat_f,
                    split=hf_args.get("test_split"),
                )
            else:
                test = []
            train_collection.append(train)
            valid_collection.append(valid)
            test_collection.append(test)
        return (
            CompletionsDatasetCollection(train_collection),
            CompletionsDatasetCollection(valid_collection),
            CompletionsDatasetCollection(test_collection),
        )
    else:
        hf_args = args.hf_dataset
        dataset_name = hf_args["name"]
        print(f"Loading Hugging Face dataset {dataset_name}.")
        text_feature, prompt_feature, completion_feature, chat_feature = (
            get_hf_custom_features(hf_args)
        )
        if args.train:
            train, valid = get_train_and_valid_splits(hf_args, dataset_name)
        else:
            train, valid = [], []
        if args.test:
            test = create_hf_dataset(
                dataset_name,
                text_feature,
                prompt_feature,
                completion_feature,
                chat_feature,
                split=hf_args.get("test_split"),
            )
        else:
            test = []

    return train, valid, test


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if getattr(args, "hf_dataset", None) is not None or getattr(args, "hf_datasets"):
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)
        if data_path.exists():
            train, valid, test = load_local_dataset(data_path, tokenizer)
        else:
            print(f"Loading Hugging Face dataset {args.data}.")
            train, valid, test = load_hf_dataset(args.data, tokenizer)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test
