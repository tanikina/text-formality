import argparse
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_formality(input_path: str) -> Dict[str, float]:
    """Evaluate text formality metrics, i.e. how they correlate with the original labels (both are
    mapped to the same range [0,1]).

    Args:
        input_path (str): Path to the file with the annotated and gold formality scores.

    Returns:
        Dict[str, float]: Dictionary with the following correlation measures:
        Pearson Correlation, Mean Absolute Error, Root Mean Squared Error, Spearman Correlation.
    """
    df = pd.read_csv(input_path)
    # input_texts = list(df["text"])
    y_true = np.array(df["gold"])
    y_pred = np.array(df["annotated"])

    # Pearson Correlation
    pearson_corr, _ = pearsonr(y_true, y_pred)
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Spearman Correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return {
        "Pearson Correlation": pearson_corr,
        "Spearman Correlation": spearman_corr,
        "MAE": mae,
        "RMSE": rmse,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters.")
    parser.add_argument(
        "--input_path", type=str, default="data/in_formal_sentences/annotated/llama8b_test.csv"
    )
    parsed_args = parser.parse_args()
    res = evaluate_formality(parsed_args.input_path)
    print(res)
