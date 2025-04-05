import argparse
import os
from pathlib import Path
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
        "Pearson Correlation": round(pearson_corr, 4),
        "Spearman Correlation": round(spearman_corr, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters.")
    parser.add_argument(
        "--input_path", type=str, default="data/in_formal_sentences/annotated/llama8b_test.csv"
    )
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        for dataset_name in ["in_formal_sentences", "pavlick_formality"]:
            input_dir = f"data/{dataset_name}"
            eval_results = dict()
            for fname in os.listdir(f"{input_dir}/annotated"):
                eval_results[fname.replace("_test.csv", "")] = evaluate_formality(
                    f"{input_dir}/annotated/{fname}"
                )
            df = pd.DataFrame(eval_results)
            output_path = f"results/{dataset_name}.csv"
            Path(output_path).parent.absolute().mkdir(parents=True, exist_ok=True)
            df.T.to_csv(output_path)
    else:
        res = evaluate_formality(args.input_path)
        print(res)
