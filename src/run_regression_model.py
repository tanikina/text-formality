import argparse
import os
import random
from pathlib import Path

import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


class RegressionModel(nn.Module):
    def __init__(self, model_name="FacebookAI/xlm-roberta-base"):
        super(RegressionModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)  # for regression
        self.loss = nn.HuberLoss(delta=1.0)

    def forward(self, input_ids, attention_mask, label):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # CLS token
        score = torch.sigmoid(self.regressor(last_hidden_state))  # Output score between 0 and 1
        return score, self.loss(score, label.unsqueeze(dim=1))  # [8,1] dim is correct?


def train(
    model: RegressionModel,
    tuned_model_name: str,
    optimizer: AdamW,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    patience: int = 5,
):
    """Trains a regression model on top of RoBERTa XLM.

    Args:
        model (RegressionModel): Model for fine-tuning.
        tuned_model_name (str): Name of the fine-tuned model.
        optimizer (AdamW): Optimizer.
        train_loader (DataLoader): Data loader for the training set.
        val_loader (DataLoader): Data loader for the validation set.
        num_epochs (int): Number of epochs for training.
        patience (int): Patience parameter for early stopping.
    """
    model.train()
    min_val_loss = None
    no_improvement = 0
    os.makedirs("saved_models", exist_ok=True)
    for epoch in range(num_epochs):
        total_loss = 0
        total_val_loss = 0
        # training loop
        for batch in train_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            optimizer.zero_grad()
            outputs, loss = model(**batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation set evaluation
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs, loss = model(**batch)
                total_val_loss += loss.item()

            print(total_val_loss)
            if min_val_loss is None or min_val_loss > total_val_loss:
                min_val_loss = total_val_loss
                no_improvement = 0
                torch.save(model.state_dict(), "saved_models/" + tuned_model_name + ".pt")
            elif min_val_loss is not None:
                no_improvement += 1
            if no_improvement > patience:
                return


def tokenize_function(data: pd.DataFrame, tokenizer):
    return tokenizer(
        [doc_tokens for doc_i, doc_tokens in enumerate(data["text"])],
        pad_to_max_length=True,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
    )


def evaluate(
    model: RegressionModel, test_loader: DataLoader, tokenizer: AutoTokenizer, output_path: str
):
    """Evaluates the fine-tuned model on the test set.

    Args:
        model (RegressionModel): Fine-tuned model.
        test_loader (DataLoader): Data loader for the test set.
        tokenizer (AutoTokenizer): Model tokenizer.
        output_path (str): Path to store csv files with annotations.
    """

    model.eval()
    total_loss = 0
    texts = []
    gold_scores = []
    annotated_scores = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            predictions, loss = model(**batch)
            labels = batch["label"]
            total_loss += loss.item()

            decoded = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            for sent, pred, gold in zip(decoded, predictions, labels):
                texts.append(sent)
                gold_scores.append(round(gold.item(), 2))
                annotated_scores.append(round(pred.item(), 2))
                # print(sent, "predicted:", round(pred.item(), 2), "gold:", round(gold.item(), 2))

    print("evaluation loss:", total_loss / len(test_loader))

    # write annotations into file
    df = pd.DataFrame(data={"text": texts, "annotated": annotated_scores, "gold": gold_scores})
    Path(output_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=True)


def main(
    num_epochs: int,
    batch_size: int,
    base_model_name: str,
    tuned_model_name: str,
    data_dir: str,
    output_path: str,
    only_evaluate: bool,
    seed: int = 42,
):
    """Training/evaluation of the fine-tuning approach for formality scoring.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        base_model_name (str): Name of the base model for fine-tuning.
        tuned_model_name (str): Name of the fine-tuned model.
        data_dir (str): Path to the directory with train/test/validation data.
        output_path (str): Path to store the annotations with formality scores.
        only_evaluate (bool): Whether to run only evaluation (otherwise, we first train and then evaluate).
        seed (int): Seed for reproducibility.
    """

    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    train_path = f"{data_dir}/train.csv"
    val_path = f"{data_dir}/validation.csv"
    test_path = f"{data_dir}/test.csv"

    train_set = Dataset.from_csv(train_path)
    train_set = train_set.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, batch_size=batch_size
    )
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    val_set = Dataset.from_csv(val_path)
    val_set = val_set.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, batch_size=batch_size
    )
    val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    test_set = Dataset.from_csv(test_path)
    test_set = test_set.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, batch_size=batch_size
    )
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    if not only_evaluate:
        model = RegressionModel(base_model_name)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5)  # 1e-5

        train(model, tuned_model_name, optimizer, train_loader, val_loader, num_epochs)

    model = RegressionModel(base_model_name)

    model.load_state_dict(torch.load("saved_models/" + tuned_model_name + ".pt"))
    model.to(device)
    evaluate(model, test_loader, tokenizer, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training or evaluation parameters.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_model_name", type=str, default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--tuned_model_name", type=str, default="formality_model")
    parser.add_argument("--data_dir", type=str, default="data/in_formal_sentences")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/in_formal_sentences/annotated/finetuned_xlmr_test.csv",
    )
    parser.add_argument("--only_evaluate", type=bool, default=False)
    args = parser.parse_args()

    main(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        base_model_name=args.base_model_name,
        tuned_model_name=args.tuned_model_name,
        data_dir=args.data_dir,
        output_path=args.output_path,
        only_evaluate=args.only_evaluate,
        seed=args.seed,
    )
