import argparse
import os

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from utils import normalize_score

seed = 42


def prepare_data(dataset_name: str, num_samples: int = 1000):
    """For each dataset we do the following:
    1. subsample the original dataset to `num_samples` (e.g., 1,000 examples)
    2. convert the score to the range between (0,1)
    3. prepare train, test, validation splits and store them in csv files

    Args:
        dataset_name (str): Name of the original dataset (`pavlick_formality` or `in_formal_sentences`).
        num_samples (int): Number of instances to sample.
    """
    if dataset_name == "pavlick_formality":
        orig_data = load_dataset("osyvokon/pavlick-formality-scores")
        sampled_data = (
            orig_data["train"]
            .remove_columns("domain")
            .shuffle(seed=seed)
            .select(range(num_samples))
        )
        sampled_data = sampled_data.rename_column("sentence", "text")
        sampled_data = sampled_data.rename_column("avg_score", "label")
        sampled_data = sampled_data.map(lambda x: normalize_score(x["label"], -3, 3, 0, 1))

    elif dataset_name == "in_formal_sentences":
        orig_data = pd.read_csv(f"data/{dataset_name}/{dataset_name}.tsv", sep="\t")
        sampled_data = Dataset.from_pandas(orig_data)

        sampled_data = sampled_data.shuffle(seed=seed).select(range(num_samples))
        sampled_data = sampled_data.remove_columns("id")
        sampled_data = sampled_data.rename_column("score", "label")
        sampled_data = sampled_data.map(lambda x: normalize_score(x["label"], -1, 1, 0, 1))

    else:
        raise NotImplementedError(
            f"Dataset can be either `pavlick_formality` or `in_formal_sentences` or `polish-formality-dataset`, the provided dataset name {dataset_name} is invalid!"
        )

    train_val, test = sampled_data.train_test_split(test_size=0.2, seed=seed).values()
    train, val = train_val.train_test_split(test_size=0.125, seed=seed).values()

    # create DatasetDict with new splits
    dataset = {"train": train, "validation": val, "test": test}
    dataset = DatasetDict(dataset)
    # save each split to csv
    os.makedirs(f"data/{dataset_name}", exist_ok=True)
    dataset["train"].to_csv(f"data/{dataset_name}/train.csv")
    dataset["validation"].to_csv(f"data/{dataset_name}/validation.csv")
    dataset["test"].to_csv(f"data/{dataset_name}/test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for preparing the dataset.")
    parser.add_argument("--dataset_name", type=str, default="pavlick_formality")
    parser.add_argument("--num_samples", type=int, default=1000)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    num_samples = args.num_samples

    prepare_data(dataset_name, num_samples)
