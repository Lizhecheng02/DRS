import argparse
import yaml
import torch
import re
import warnings
import transformers
from utils import get_couldask, get_response_openai, llm_eval_answerable, get_response_hf
from tqdm import tqdm
warnings.filterwarnings("ignore")


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def drs(question_entities, context, pipeline):
    n = len(question_entities)
    result = {"final_question": [], "num_used_entity": []}

    def dfs(entities, depth=0, current_question=None, update=False):
        print(entities, depth)

        if len(result["final_question"]) >= 3:
            return
            
        if len(entities) >= min(int((n + 1) / 2), n) and current_question and update:
            messages = [{"role": "user", "content": f"Here is a long context: {context}\nHere is a question: {current_question}\nTell me whether this question is answerable only according to the information in the provided context. Give you analysis within <analysis> tags, then return only the final answer 'yes' or 'no' within <answer> tags."}]
            answerable_check = get_response_hf(pipeline=pipeline, messages=messages)
            answerable_check = re.search(r"<answer>(.*?)</answer>", answerable_check, re.DOTALL).group(1).strip().lower()
            if "no" not in answerable_check:
                result["final_question"].append(current_question)
                result["num_used_entity"].append(len(entities))
            else:
                return
            
        for i in range(len(question_entities)):
            if question_entities[i] not in entities and (len(entities) == 0 or i > question_entities.index(entities[-1])):
                if len(entities) + 1 >= int((n + 1) / 2):
                    entities.append(question_entities[i])
                    messages = [{"role": "user", "content": f"According to the following text:\n<text>{context}</text>\nGenerate a statement you could get from the given text that contains all following entities: [{', '.join(entities)}]. Then you are required to generate a question contains all given entities, which could be answered from your statement. Return the statement within <statement> tags and the question within <question> tags."}]
                    modified_question = get_response_hf(pipeline=pipeline, messages=messages)
                    modified_question = re.search(r"<question>(.*?)</question>", modified_question, re.DOTALL).group(1).strip()
                    messages = [{"role": "user", "content": f"Here is a question: {modified_question}. Does it contain all following entities: [{', '.join(entities)}]?. They do not need to be strictly the same, as long as mentioned, uppercase, lowercase doesn't matter. Think carefully, return your answer 'yes' or 'no' within <check> tags."}]
                    check = get_response_hf(pipeline=pipeline, messages=messages)
                    check = re.search(r"<check>(.*?)</check>", check, re.DOTALL).group(1).strip()
                    if "yes" in check.lower():
                        dfs(entities, depth + 1, modified_question, update=True)
                    else:
                        dfs(entities, depth + 1, current_question, update=False)
                    entities.pop()
                else:
                    entities.append(question_entities[i])
                    dfs(entities, depth + 1, current_question, update=False)
                    entities.pop()

    dfs([])
    
    if len(result["final_question"]) == 1 or len(result["final_question"]) == 0:
        response = "1"
    else:
        prompt = f"I will give you a text and some questions (including how many entities each question contains). You need to tell me which question is the most answerable one only according to the given text, if many questions are ok, choose the one that contains the most entities.\nThe text is:\n<text>{context}</text>\nThe questions are:\n"
        for i, question in enumerate(result["final_question"]):
            prompt += f"Question {i + 1}: {question} - Entity Num: {result['num_used_entity'][i]}\n"
        prompt += "Give your analysis within <analysis> tags and return the id number of the chosen question within <answer> tags. For example, <answer>4</answer> means the 4th question is the most answerable one."
        messages = [{"role": "user", "content": prompt}]
        response = get_response_hf(pipeline=pipeline, messages=messages)
        response = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)[-1].strip()
    return result["final_question"][int(response) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", type=str)
    args = parser.parse_args()
    config = load_config(args.config)

    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    df = get_couldask(subset_name=args.subset_name)

    if args.task == "reformulate":
        df_unanswerable = df[df["answerable"] == 0]
        reformulate_unanswerable_count = 0
        pass_overlap_entity_check = 0
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
            pass_overlap_for_this_idx = 0
            pass_answerable_check_for_this_idx = 0

            try:
                if args.run_model_platform == "huggingface":
                    messages = [{"role": "user", "content": f"Find out all entities in the following question: {question}. The entities you find must be exactly exist in the question sentence. You should only return the entities, separated by comma and space."}]
                    entities_response = get_response_hf(pipeline=pipeline, messages=messages)
                    question_entities = entities_response.split(", ")
                    question_entities = [entity.strip() for entity in question_entities]

                    question_entities_filtered = []
                    for entity in question_entities:
                        messages = [{"role": "user", "content": f"Here is a question:<question>{question}</question>\nHere is an entity in this question:<entity>{entity}</entity>.\nTell me which category does this entity belong to in the given question - subject, object, predicate, attribute or others. You should only consider the situation of given entity in this specific question. Give your analysis within <analysis> tags, and only return its category name within <answer> tags."}]
                        response = get_response_hf(pipeline=pipeline, messages=messages)
                        response = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL).group(1).strip().lower()
                        if "subject" in response or "object" in response or "attribute" in response:
                            question_entities_filtered.append(entity)
                    question_entities = question_entities_filtered

                    new_question = drs(question_entities=question_entities, context=context, pipeline=pipeline)

                    entities_overlap = get_response_openai(model="gpt-4o-mini", prompt=f"This is an original question: {question}, it contains the following entities: {row['entities']}.\nThis is a new question: {new_question}.\nTell me the number of overlapping entities between the new question and the original question, they do not need to be strictly the same, as long as mentioned, uppercase, lowercase or format doesn't matter. Give your analysis within <analysis> tag, and only return the math number of overlap entities within <answer> tags.")
                    original_entities_num = len(row["entities"])
                    entities_overlap = re.search(r"<answer>(.*?)</answer>", entities_overlap, re.DOTALL).group(1).strip()
                    overlap_entites_num = int(entities_overlap)
                    if overlap_entites_num >= int((original_entities_num + 1) / 2):
                        pass_overlap_entity_check += 1
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
