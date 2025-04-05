import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_visualization(setting: str, plot_path: str):
    """Create plots to visualize score distribution.

    Args:
        setting (str): Which setting is compared to the gold distribution (`llms` or `traditional`).
        plot_path (str): Where to store the generated plot.
    """
    # Reading data
    if setting == "llms":
        input_path_qwen = "data/pavlick_formality/annotated/qwen7b_test.csv"
        df_qwen = pd.read_csv(input_path_qwen)
        y_true = np.array(df_qwen["gold"])
        y_pred_qwen = np.array(df_qwen["annotated"])

        input_path_llama = "data/pavlick_formality/annotated/llama8b_test.csv"
        df_llama = pd.read_csv(input_path_llama)
        y_pred_llama = np.array(df_llama["annotated"])

        input_path_xlmr = "data/pavlick_formality/annotated/finetuned_xlmr_test.csv"
        df_xlmr = pd.read_csv(input_path_xlmr)
        y_pred_xlmr = np.array(df_xlmr["annotated"])

        # Create overlaying KDE plots
        sns.kdeplot(y_true, color="blue", label="True score distribution", fill=True, alpha=0.4)
        sns.kdeplot(
            y_pred_qwen, color="red", label="Qwen 7b score distribution", fill=True, alpha=0.4
        )
        sns.kdeplot(
            y_pred_llama, color="green", label="Llama 8b score distribution", fill=True, alpha=0.4
        )
        sns.kdeplot(
            y_pred_xlmr,
            color="orange",
            label="Fine-tuned XLMR score distribution",
            fill=True,
            alpha=0.4,
        )
    elif setting == "traditional":
        input_path_readability = "data/pavlick_formality/annotated/avg_readability_score_test.csv"
        df_readability = pd.read_csv(input_path_readability)
        y_true = np.array(df_readability["gold"])
        y_pred_readability = np.array(df_readability["annotated"])

        input_path_heylighen = "data/pavlick_formality/annotated/heylighen_score_test.csv"
        df_heylighen = pd.read_csv(input_path_heylighen)
        y_pred_heylighen = np.array(df_heylighen["annotated"])

        input_path_lexical_diversity = (
            "data/pavlick_formality/annotated/lexical_diversity_test.csv"
        )
        df_lexical_diversity = pd.read_csv(input_path_lexical_diversity)
        y_pred_lexical_diversity = np.array(df_lexical_diversity["annotated"])

        input_path_syntactic_complexity = (
            "data/pavlick_formality/annotated/syntactic_complexity_test.csv"
        )
        df_syntactic_complexity = pd.read_csv(input_path_syntactic_complexity)
        y_pred_syntactic_complexity = np.array(df_syntactic_complexity["annotated"])

        # Create overlaying KDE plots
        sns.kdeplot(y_true, color="blue", label="True score distribution", fill=True, alpha=0.4)
        sns.kdeplot(
            y_pred_readability, color="red", label="Readability score", fill=True, alpha=0.4
        )
        sns.kdeplot(y_pred_heylighen, color="green", label="Heylighen score", fill=True, alpha=0.4)
        sns.kdeplot(
            y_pred_lexical_diversity,
            color="orange",
            label="Lexical diversity score",
            fill=True,
            alpha=0.4,
        )
        sns.kdeplot(
            y_pred_syntactic_complexity,
            color="purple",
            label="Syntactic complexity score",
            fill=True,
            alpha=0.4,
        )

    # Labels and legend
    plt.xlabel("Value", fontsize=10)
    plt.ylabel("Density", fontsize=1)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Overlaying KDE Plots", fontsize=12)
    plt.legend(fontsize=10)
    # plt.show()
    Path(plot_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization parameters.")
    parser.add_argument("--setting", type=str, default="llms", choices=["traditional", "llms"])
    parser.add_argument("--plot_path", type=str, default="figures/llm_based_methods.png")
    args = parser.parse_args()
    create_visualization(args.setting, args.plot_path)
