import argparse
import yaml
import re
import warnings
from utils import get_couldask, get_response_openai_prompt, llm_eval_answerable
from tqdm import tqdm
warnings.filterwarnings("ignore")


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def drs(question_entities, context, model):
    n = len(question_entities)
    result = {"final_question": [], "num_used_entity": []}

    def dfs(entities, depth=0, current_question=None, update=False):
        print(entities, depth)

        if len(result["final_question"]) >= 3:
            return
            
        if len(entities) >= min(int((n + 1) / 2), n) and current_question and update:
            if "no" not in llm_eval_answerable(context, current_question, model, "openai").lower():
                result["final_question"].append(current_question)
                result["num_used_entity"].append(len(entities))
            else:
                return
            
        for i in range(len(question_entities)):
            if question_entities[i] not in entities and (len(entities) == 0 or i > question_entities.index(entities[-1])):
                if len(entities) + 1 >= int((n + 1) / 2):
                    entities.append(question_entities[i])
                    modified_question = get_response_openai_prompt(
                        model=model,
                        prompt=f"According to the following text:\n<text>{context}</text>\nGenerate a statement you could get from the given text that contains all following entities: [{', '.join(entities)}]. Then you are required to generate a question contains all given entities, which could be answered from your statement. Return the statement within <statement> tags and the question within <question> tags."
                    )
                    modified_question = re.search(r"<question>(.*?)</question>", modified_question, re.DOTALL).group(1).strip()
                    check = get_response_openai_prompt(
                        model=model,
                        prompt=f"Here is a question: {modified_question}. Does it contain all following entities: [{', '.join(entities)}]?. They do not need to be strictly the same, as long as mentioned, uppercase or lowercase doesn't matter. Think carefully, return your answer 'yes' or 'no' within <check> tags."
                    )
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
        prompt += "Give your analysis within <analysis> tags and return the id number of the chosen question within <id> tags. For example, <id>4</id> means the 4th question is the most answerable one."
        response = get_response_openai_prompt(model=model, prompt=prompt)
        response = re.search(r"<id>(.*?)</id>", response, re.DOTALL).group(1).strip()
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
        print(f"The shape of unanswerable data is: {df_unanswerable.shape}")
        reformulate_unanswerable_count = 0
        pass_overlap_entity_check = 0
        success_num = 0

        for idx, row in tqdm(df_unanswerable.iterrows(), total=len(df_unanswerable)):
            context = row["context"]
            question = row["question"]
            pass_overlap_for_this_idx = 0
            pass_answerable_check_for_this_idx = 0

            try:
                if args.run_model_platform == "openai":
                    entities_response = get_response_openai_prompt(
                        model=args.run_model, 
                        prompt=f"Find out all entities in the following question: {question}. Only return the entities, separated by comma and space."
                    )
                    question_entities = entities_response.split(", ")
                    question_entities = [entity.strip() for entity in question_entities]

                    question_entities_filtered = []
                    for entity in question_entities:
                        response = get_response_openai_prompt(
                            model=args.run_model, 
                            prompt=f"Here is a question:<question>{question}</question>\nHere is an entity in this question:<entity>{entity}</entity>.\nTell me which category does this entity belong to in the given question - subject, object, predicate, attribute or others. Give your analysis within <analysis> tags, and only return its category name within <answer> tags."
                        )
                        response = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL).group(1).strip().lower()
                        if "subject" in response or "object" in response or "attribute" in response:
                            question_entities_filtered.append(entity)
                    question_entities = question_entities_filtered

                    new_question = drs(question_entities=question_entities, context=context, model=args.run_model)

                    entities_overlap = get_response_openai_prompt(
                        model="gpt-4o-mini", 
                        prompt=f"This is an original question: {question}, it contains the following entities: {row['entities']}.\nThis is a new question: {new_question}.\nTell me the number of overlapping entities between the new question and the original question, they do not need to be strictly the same, as long as mentioned, uppercase or lowercase doesn't matter. Give your analysis within <analysis> tags and only return the math number of overlap entities within <answer> tags."
                    )
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
