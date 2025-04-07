## Towards Text Formality Annotation with LLMs and Traditional Approaches

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Paper](https://img.shields.io/badge/report-blue)](https://github.com/tanikina/text-formality/blob/main/report/Text_Formality_Detection_Report.pdf)

## 📌 Description

This repository contains the code for evaluating text formality levels as described in [this report](https://github.com/tanikina/text-formality/blob/main/report/Text_Formality_Detection_Report.pdf).

## 📃 Abstract

Text formality detection is a non-trivial task that has a lot of practical applications. Being able to correctly detect the formality of the text can help to tailor it to a specific audience, making the communication more efficient while adhering to specific situational and cultural norms. In this study we investigate how text formality can be detected via prompting or fine-tuning Large Language Models (LLMs), and compare the results to the scores that we obtain with more traditional metrics (e.g., readability score, lexical diversity or syntactic complexity of the text). We experiment with two datasets annotated with formality scores in English and German, and compute various metrics to estimate how well our annotations align with the original scores. Our findings show that fine-tuning or prompting a sufficiently large LM results in strong performance, with readability metrics also showing competitive results, but lexical diversity is not a good indicator of text formality.

## 🚀 How to reproduce the results

## General Setup

After you cloned this repository, you can follow these steps to set up the environment:

```
conda create -n text-formality python=3.11
conda activate text-formality
pip install -r requirements.txt
```

To reproduce the experiments with the traditional approaches you will also need to download NLTK and SpaCy models

```
python -c "import nltk; nltk.download('punkt')"
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Data Preparation

This work uses _PavlickFormality_ [(Pavlick and Tetreault, 2016)](https://aclanthology.org/Q16-1005/) and _Informal Sentences_ [(Eder et al., 2023)](https://aclanthology.org/2023.findings-eacl.42/) datasets.

```
python src/prepare_datasets.py --dataset_name pavlick_formality --num_samples 1000
python src/prepare_datasets.py --dataset_name in_formal_sentences --num_samples 1000

```

## LLM Fine-tuning

Fine-tuning multilingual `FacebookAI/xlm-roberta-base` model (on NVIDIA GeForce RTX 2080 with 8GB RAM):

```
python src/run_regression_model.py \
--data_dir=data/{dataset_name} \
--output_path=data/{dataset_name}/annotated/finetuned_xlmr_test.csv
```

Note that `{dataset_name}` can be either `in_formal_sentences` or `pavlick_formality`.

## LLM Prompting

If you want to reproduce our results using Llama3 model, please make sure that you have HF_TOKEN available (this can be done e.g. via huggingface-cli, see [the official documentation](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) for more detail).

```
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
```

Our experiments were conducted on a single NVIDIA GeForce RTX 2080 with 8GB RAM.

```
python src/prompt_llm.py \
--demo_path="data/{dataset_name}/train.csv" \
--input_path="data/{dataset_name}/test.csv" \
--output_path="data/{dataset_name}/annotated/test_llama8b.csv" \
--model_name="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ" \
--num_input_demos=10
```

You can run the same script also with `model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"` to reproduce the results for few-shot prompting with Qwen. As in the fine-tuning experiments `{dataset_name}` can be either `in_formal_sentences` or `pavlick_formality`.

## Traditional Methods

We provide the script to compute the following formality-related metrics:

- **F-score** based on Heylighen & Dewaele
  ```
  python src/compute_traditional_formality_scores.py \
  --input_path=data/{dataset_name}/test.csv \
  --output_path=data/{dataset_name}/annotated/heylighen_score_test.csv \
  --language={lang} \
  --metric=heylighen_score \
  --normalize=True
  ```
- **Average readability score** based on Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index, and Automated Readability Index
  ```
  python src/compute_traditional_formality_scores.py \
  --input_path=data/{dataset_name}/test.csv \
  --output_path=data/{dataset_name}/annotated/avg_readability_score_test.csv \
  --language={lang} \
  --metric=avg_readability_score \
  --normalize=True
  ```
- **Syntactic complexity score** (comparison to the average sentence length of the sentences that have high formality scores in the training data)
  ```
  python src/compute_traditional_formality_scores.py \
  --input_path=data/{dataset_name}/test.csv \
  --output_path=data/{dataset_name}/annotated/syntactic_complexity_test.csv \
  --language={lang} \
  --metric=syntactic_complexity
  ```
- **Lexical diversity score**
  ```
  python src/compute_traditional_formality_scores.py \
  --input_path=data/{dataset_name}/test.csv \
  --output_path=data/{dataset_name}/annotated/lexical_diversity_test.csv \
  --language={lang} \
  --metric=lexical_diversity
  ```
  Note that `language={lang}` should be `language=de` for `{dataset_name}=in_formal_sentences` and `language=en` for `{dataset_name}=pavlick_formality`.

## Evaluation and Visualization

To compute correlation metrics for all datasets and settings at once you can run:
`python src/evaluate_formality_scores.py --all`

To visualize score deistributions and compare llm-based and traditional approaches for different datasets, please run:

```
python src/visualize_distribution.py \
--dataset_name={dataset_name} \
--setting={setting} \
--plot_path=figures/{output_filename}
```

Note that the available settings are `setting=llms` or `setting=traditional`.
