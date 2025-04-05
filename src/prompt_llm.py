import argparse
import os
import random
import re
import sys
from os.path import abspath, dirname
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

random.seed(42)


os.environ[
    "VLLM_WORKER_MULTIPROC_METHOD"
] = "spawn"  # this is needed if we use more than 1 GPU with VLLM

FLOAT_PATTERN = r"\d*\.\d+(?:\d+)?"  # matches float numbers

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_float(text: str):
    return re.findall(FLOAT_PATTERN, text)


def is_valid(input_text: str) -> bool:
    """Checks whether the input score is a valid float in the required range."""
    try:
        float(input_text)  # try converting to float
        if input_text.startswith("0.") or input_text == "1.0":  # check the range [0.0, 1.0]
            return True
        else:
            return False
    except ValueError as e:
        print(e)
        return False


def annotate_formality(
    demo_path: str,
    input_path: str,
    output_path: str,
    num_input_demos: int,
    model_name: str,
    verbose: bool,
):
    """Create plots to visualize score distribution.

    Args:
        demo_path (str): Path to the file from which to choose LLM demonstrations.
        input_path (str): Path to the input file with the texts for annotation.
        output_path (str): Path to the file where to store the results with the formality scores.
        num_input_demos (int): Number of demonstrations to include in the prompt.
        model_name (str): Name of the LLM to prompt.
        verbose (bool): Whether to print intermediate results.
    """
    df_demos = pd.read_csv(demo_path)
    demo_texts = list(df_demos["text"])
    demo_labels = list(df_demos["label"])
    df_inputs = pd.read_csv(input_path)
    input_samples = list(df_inputs["text"])
    gold_input_labels = list(df_inputs["label"])

    # Generation

    # setting up vllm generation
    vllm_model = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=1024,  # 4096
        dtype="float16",
    )  # use dtype="float16" if GPU has compute capacity < 8
    vllm_sampling_params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=128)

    annotations = []

    # selecting random demonstrations for formality levels
    demos = []
    for txt, lbl in zip(demo_texts, demo_labels):
        demos.append((txt, lbl))
    random.shuffle(demos)
    demos = demos[:num_input_demos]
    demo_str = ""
    for txt, lbl in demos:
        demo_str += f"Input: {txt} Formality label: {lbl}\n"

    # annotating the test data, note: vllm allows us to also batch prompts,
    # but since my GPU is not very powerful, I get CUDA OOM with multiple prompts per batch
    # so we default to one-sample-at-a-time setting
    for input_sample, gold_label in zip(input_samples, gold_input_labels):
        num_trials = 0
        not_successful = True
        decoded_label = ""
        while not_successful:
            generation_prompt = f"You are required to annotate given example and assign the text formality level that should be in the range between 0 and 1 (0 means most informal and 1 most formal). Here are some examples. {demo_str}\nNow annotate the following sample. Input: {input_sample} Formality label: "
            messages = [
                {
                    "role": "system",
                    "content": "You are an excellent text formality annotator that outputs continuous scores between 0 and 1.",
                },
                {"role": "user", "content": generation_prompt},
            ]

            decoded = vllm_model.chat(messages, vllm_sampling_params)[0].outputs[0].text
            decoded_float = extract_float(decoded)
            if len(decoded_float) > 0:
                decoded_label = decoded_float[0]
            num_trials += 1
            if is_valid(decoded_label):
                annotations.append(decoded_label)
                not_successful = False
            elif num_trials == 10:
                decoded_label = "0.5"  # average
                annotations.append(decoded_label)
                print(
                    f"Could not assign a label to the following instance: {input_sample}. Assigning the default label of 0.5."
                )
                not_successful = False

            if verbose:
                print(
                    f"Input sample: {input_sample}. Formality label: {decoded_label} Gold label: {gold_label}"
                )

    # write annotations into file
    df = pd.DataFrame(
        data={"text": input_samples, "annotated": annotations, "gold": gold_input_labels}
    )
    Path(output_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters.")
    parser.add_argument("--demo_path", type=str, default="data/in_formal_sentences/train.csv")
    parser.add_argument("--input_path", type=str, default="data/in_formal_sentences/test.csv")
    parser.add_argument(
        "--output_path", type=str, default="data/in_formal_sentences/annotated/test_llama8b.csv"
    )

    parser.add_argument(
        "--model_name", type=str, default="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"
    )
    parser.add_argument("--num_input_demos", type=int, default=10)
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, v)

    annotate_formality(
        demo_path=args.demo_path,
        input_path=args.input_path,
        output_path=args.output_path,
        num_input_demos=args.num_input_demos,
        model_name=args.model_name,
        verbose=args.verbose,
    )
