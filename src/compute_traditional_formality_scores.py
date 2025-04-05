import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import spacy
import textstat
from nltk.tokenize import word_tokenize

from utils import normalize_score

# POS tags for formal and informal words
FORMAL_TAGS = {"NOUN", "ADJ", "ADP", "DET"}  # (nouns, adjectives, prepositions, articles)
INFORMAL_TAGS = {"PRON", "VERB", "ADV", "INTJ"}  # (pronouns, verbs, adverbs, interjections)


def compute_heylighen_score(text, nlp, normalize=True) -> float:
    """Computes the formality score (F-score) based on Heylighen & Dewaele (1999).

    Args:
        text (str): Input text (single sentence in this project).
        nlp (spacy.lang): SpaCy text processing utility for a specific language.
        normalize (bool): Whether to normalize the score and bring it in the range [0,1].

    Returns:
        float: Returns a formality score (higher = more formal, lower = more informal).
    """
    # processing text with spaCy
    doc = nlp(text)
    # counting occurrences of formal and informal POS tags
    formal_count = sum(1 for token in doc if token.pos_ in FORMAL_TAGS)
    informal_count = sum(1 for token in doc if token.pos_ in INFORMAL_TAGS)
    total_words = len([token for token in doc if token.is_alpha])  # Exclude punctuation
    # to avoid division by zero
    if total_words == 0:
        return 0
    # computing formality score
    formality_score = (formal_count - informal_count) / total_words
    formality_score = crop_range(round(formality_score, 2), -1, 1)
    if normalize:
        formality_score = normalize_score(formality_score, -1, 1)

    return formality_score


def crop_range(input_score: float, min_bound: float, max_bound: float) -> float:
    """To ensure that all values are within the allowed range [min_bound, max_bound].
    Args:
        input_score (float): Original score.
        min_bound (float): Minimum score of the new range.
        max_bound (float): Maximum score of the new range.

    Returns:
        float: Cropped value within the specified range.
    """
    if input_score < min_bound:
        return min_bound
    elif input_score > max_bound:
        return max_bound
    return input_score


def compute_readability_score(text, normalize=True) -> Dict[str, float]:
    """Computes readability scores for a given text.
    Args:
        text (str): Input text.
        normalize (bool): Whether to normalize the score.

    Returns:
        Dict[str, float]: Dictionary with the following readability scores:
        Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index, and Automated Readability Index.

    """
    flesch_score = textstat.flesch_reading_ease(text)  # Flesch Reading Ease
    flesch_score = crop_range(flesch_score, 0, 100)
    flesch_kincaid_score = textstat.flesch_kincaid_grade(text)  # Flesch-Kincaid Grade Level
    flesch_kincaid_score = crop_range(flesch_kincaid_score, 0, 12)
    gunning_fox_index = textstat.gunning_fog(text)  # Gunning Fog Index
    gunning_fox_index = crop_range(gunning_fox_index, 6, 20)
    ari_score = textstat.automated_readability_index(text)  # Automated Readability Index
    ari_score = crop_range(ari_score, 0, 14)
    if normalize:
        return {
            "flesch_score": normalize_score(flesch_score, 0, 100),
            "flesch_kincaid_score": normalize_score(flesch_kincaid_score, 0, 12),
            "gunning_fox_index": normalize_score(gunning_fox_index, 6, 20),
            "automated_readability_index": normalize_score(ari_score, 0, 14),
        }
    else:
        return {
            "flesch_score": flesch_score,
            "flesch_kincaid_score": flesch_kincaid_score,
            "gunning_fox_index": gunning_fox_index,
            "automated_readability_index": ari_score,
        }


def compute_lexical_diversity_score(input_text: str) -> float:
    """Computes lexical diversity score for a given text.

    The score is already between 0 and 1, so there is no need to normalize it.
    """
    tokens = word_tokenize(input_text.lower())
    # Type-Token Ratio (TTR)
    ttr_ratio = len(set(tokens)) / len(tokens)
    return ttr_ratio


def compute_syntactic_complexity_score(
    input_text: str, avg_formal_sentence_length: float
) -> float:
    """Computes syntactic complexity score for a given text.

    Note that in this project we have sentence-based texts, thus this metric defaults to computing
    a sentence length and comparing it with an average sentence length for formal sentences (score
    >=0.75) computed based on the given training set.
    """
    avg_sentence_length = len(word_tokenize(input_text))
    syntactic_complexity_score = crop_range(avg_sentence_length / avg_formal_sentence_length, 0, 1)
    return syntactic_complexity_score


def compute_metric(input_path: str, output_path: str, language: str, metric: str, normalize: bool):
    """Computes the formality metric for each sentence of the input file and stores the results in
    a csv file.

    Args:
        input_path (str): Path to the input file.
        output_path (str): Path to the output file.
        language (str): The language of the input texts (needed to load the correct SpaCy model).
        normalize (bool): Whether to normalize the score and bring it in the range [0,1].
    """
    # preparing the data
    df = pd.read_csv(input_path)
    input_texts = list(df["text"])
    gold_labels = list(df["label"])
    scores = []
    avg_formal_sentence_length = 0.0
    # compute average length of formal sentences based on the training data
    if metric == "syntactic_complexity":
        df_train = pd.read_csv(input_path.replace("test.csv", "train.csv"))
        formal_train_texts = list(df_train.loc[df_train["label"] >= 0.75]["text"])
        avg_formal_sentence_length = sum(
            [len(word_tokenize(sent)) for sent in formal_train_texts]
        ) / len(formal_train_texts)

    # loading SpaCy model
    if language == "de":
        nlp = spacy.load("de_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")

    for input_text in input_texts:
        # computing the given metric
        if metric == "heylighen_score":
            score = compute_heylighen_score(input_text, nlp, normalize)
        elif metric == "avg_readability_score":
            readability_metrics = compute_readability_score(input_text, normalize).values()
            score = sum(readability_metrics) / len(readability_metrics)
        elif metric == "lexical_diversity":
            score = compute_lexical_diversity_score(input_text)
        elif metric == "syntactic_complexity":
            score = compute_syntactic_complexity_score(input_text, avg_formal_sentence_length)
        else:
            raise ValueError(f"Unsupported metric: {metric}.")
        scores.append(round(score, 2))

    # write annotations into file
    df = pd.DataFrame(data={"text": input_texts, "annotated": scores, "gold": gold_labels})
    Path(output_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameters for collecting traditionally computed formality scores."
    )
    parser.add_argument("--input_path", type=str, default="data/pavlick_formality/test.csv")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/pavlick_formality/annotated/traditional_metrics_test.csv",
    )
    parser.add_argument("--language", type=str, default="en", choices=["en", "de"])
    parser.add_argument(
        "--metric",
        type=str,
        default="heylighen_score",
        choices=[
            "heylighen_score",
            "avg_readability_score",
            "lexical_diversity",
            "syntactic_complexity",
        ],
    )
    parser.add_argument("--normalize", type=bool, default=False)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    language = args.language
    normalize = args.normalize
    metric = args.metric

    compute_metric(input_path, output_path, language, metric, normalize)
