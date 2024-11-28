import os
import yaml
import re
import warnings
from openai import OpenAI
from datasets import load_dataset
warnings.filterwarnings("ignore")

config_path = os.path.join(os.getcwd(), "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

openai_api_key = config["openai"]["api_key"]
openai_organization = config["openai"]["organization"]
client_openai = OpenAI(api_key=openai_api_key, organization=openai_organization)


def get_couldask(subset_name):
    ds = load_dataset("wentingzhao/couldask", subset_name)
    df = ds["test"].to_pandas()
    return df


def llm_eval_answerable(context, question, eval_model):
    prompt = f"Here is a long context: {context}\nHere is a question: {question}\nTell me whether this question is answerable only according to the information in the provided context. Give you analysis within <analysis> tags, then return only the final answer 'yes' or 'no' within <answer> tags."
    check = get_response_openai_prompt(model=eval_model, prompt=prompt, temperature=0.0, top_p=0.95, max_tokens=512)
    check = re.search(r"<answer>(.*?)</answer>", check, re.DOTALL).group(1).strip()
    return check


def get_response_openai_prompt(model, prompt, temperature=0.0, top_p=0.95, max_tokens=512):
    completion = client_openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    response = completion.choices[0].message.content.strip()
    return response


def get_response_openai_messages(model, messages, temperature=0.0, top_p=0.95, max_tokens=512):
    completion = client_openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    response = completion.choices[0].message.content.strip()
    return response


def get_response_hf(pipeline, messages, temperature=0.01, top_p=0.95, max_tokens=512):
    outputs = pipeline(
        messages,
        max_new_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature
    )
    return outputs[0]["generated_text"][-1]["content"].strip()
