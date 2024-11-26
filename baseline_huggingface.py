import argparse
import yaml
import re
import torch
import warnings
import transformers
from utils import get_couldask, get_response_openai, llm_eval_answerable, get_response_hf
from tqdm import tqdm
from datasets import load_dataset
warnings.filterwarnings("ignore")


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", type=str)
    args = parser.parse_args()
    config = load_config(args.config)

    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    df = get_couldask(subset_name=args.subset_name)

    fewshot = load_dataset("json", data_files="baseline.json", split="train")

    if args.task == "reformulate":
        df_unanswerable = df[df["answerable"] == 0]

        success_num = 0

        print("Loading Model ...")
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.run_model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

        for idx, row in tqdm(df_unanswerable.iterrows(), total=len(df_unanswerable)):
            context = row["context"]
            question = row["question"]
            reformulate_unanswerable_count = 0
            pass_overlap_for_this_idx = 0
            pass_answerable_check_for_this_idx = 0
        
            try:
                if args.type == "zs":
                    messages = [{"role": "user", "content": f"Here is a context: {context}.\nHere is a question: {question}.\nCan you minimally edit the question so that it becomes answerable?"}]
                    new_question = get_response_hf(pipeline=pipeline, messages=messages, max_tokens=64)
                    new_question = get_response_openai(model="gpt-4o-mini", prompt=f"Extract the reformulated question from the following output:\n{new_question}\n.Return the reformulated question only.", temperature=0.0)
                elif args.type == "zscot":
                    messages = [{"role": "user", "content": f"Here is a context: {context}.\nHere is a question: {question}.\nCan you think step by step to reason about the minimal edits you can make to reformulate the question so that the question answerable? Lay out your work and return your reformulated question."}]
                    new_question = get_response_hf(pipeline=pipeline, messages=messages)
                    new_question = get_response_openai(model="gpt-4o-mini", prompt=f"Extract the reformulated question from the following output:\n{new_question}\n.Return the reformulated question only.", temperature=0.0)
                elif args.type == "fs":
                    messages = []
                    for one in fewshot:
                        if one["correction"] == "":
                            continue
                        messages.append({"role": "user", "content": f"Here is a context: {context}.\nHere is a question: {question}.\nCan you minimally edit the question so that it becomes answerable?"})
                        messages.append({"role": "assistant", "content": one["correction"]})
                    messages.append({"role": "user", "content": f"Here is a context: {context}.\nHere is a question: {question}.\nCan you minimally edit the question so that it becomes answerable?"})
                    new_question = get_response_hf(pipeline=pipeline, messages=messages, max_tokens=64)
                    new_question = get_response_openai(model="gpt-4o-mini", prompt=f"Extract the reformulated question from the following output:\n{new_question}\n.Return the reformulated question only.", temperature=0.0)
                elif args.type == "fscot":
                    messages = []
                    for one in fewshot:
                        if one["correction"] == "": 
                            continue
                        messages.append({"role": "user", "content": f"Here is a context: {context}.\nHere is a question: {question}.\nCan you think step by step to reason about the minimum edits you can make to reformulate the question so that the question answerable? Lay out your work."})
                        messages.append({"role": "assistant", "content": one["cot_q"]})
                    messages.append({"role": "user", "content": f"Here is a context: {context}.\nHere is a question: {question}.\nCan you think step by step to reason about the minimum edits you can make to reformulate the question so that the question answerable? Lay out your work and return your reformulated question."})
                    new_question = get_response_hf(pipeline=pipeline, messages=messages)
                    new_question = get_response_openai(model="gpt-4o-mini", prompt=f"Extract the reformulated question from the following output:\n{new_question}\n.Return the reformulated question only.", temperature=0.0)
                    
                entities_overlap = get_response_openai(model="gpt-4o-mini", prompt=f"This is an original question: {question}, it contains the following entities: {row['entities']}.\nThis is a new question: {new_question}.\nTell me the number of overlapping entities between the new question and the original question, they do not need to be strictly the same, as long as mentioned, uppercase or lowercase doesn't matter. Give your analysis within <analysis> tag, and only return the math number of overlap entities within <answer> tags.")
                original_entities_num = len(row["entities"])
                entities_overlap = re.search(r"<answer>(.*?)</answer>", entities_overlap, re.DOTALL).group(1).strip()
                overlap_entites_num = int(entities_overlap)
                if overlap_entites_num >= int((original_entities_num + 1) / 2):
                    pass_overlap_for_this_idx += 1

                reformulate_answerable_check = llm_eval_answerable(context, new_question, "gpt-4o-mini")
                if reformulate_answerable_check == "no":
                    reformulate_unanswerable_count += 1
                else:
                    pass_answerable_check_for_this_idx += 1

                if pass_answerable_check_for_this_idx == 1 and pass_overlap_for_this_idx == 1:
                    success_num += 1

            except:
                pass

        print(f"The success rate for {args.subset_name} dataset is {success_num / len(df_unanswerable)}")


if __name__ == "__main__":
    main()
