import json
from collections import defaultdict

import attrs
import tqdm

from file_path import SHROOM_AG_DEV_PATH, OUT_FILE_PATH, SHROOM_AW_DEV_PATH, SHROOM_AW_TEST_PATH,  SHROOM_AG_TEST_PATH
from data.ingest import load_instances, dump_instances
from openai import OpenAI
from time import sleep

from prompt_modeling.construct_prompt import (
    general_pipeline_prompt,
    pg_intermediate_prompt
)

instances = load_instances(SHROOM_AW_DEV_PATH)
task_instances = defaultdict(list)
for ins in instances:
    task_instances[ins.meta.task].append(ins)

client = OpenAI(api_key="")

COMPLETION_PARAMS = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 200,  # max tokens in the output
    "messages": [],
}


def run_gpt(prompt_str: str):
    COMPLETION_PARAMS["messages"] = generate_message(prompt_str)

    response = client.chat.completions.create(**COMPLETION_PARAMS)

    content = response.choices[0].message.content
    return content


def generate_message(prompt_str: str):
    return [{"role": "user", "content": prompt_str}]


def gpt_res2file_v2(task_name: str):
    task2prompt_mapping = {
        "DM": general_pipeline_prompt,
        "MT": general_pipeline_prompt,
        "PG": general_pipeline_prompt,
    }
    out_json_file = open(OUT_FILE_PATH.joinpath(f"out.jsonl"), "w")

    for ins in tqdm.tqdm(task_instances[task_name]):
        ins_dict = attrs.asdict(ins)
        if ins_dict["tgt"] == "":
            # print("SWAPPED tgt for src")
            fulfilled_prompt = pg_intermediate_prompt.replace("!tgt!", ins_dict["src"])
            fulfilled_prompt = fulfilled_prompt.replace("!hyp!", ins_dict["hyp"])
        else:
            fulfilled_prompt = general_pipeline_prompt.replace("!tgt!", ins_dict["tgt"])
            fulfilled_prompt = fulfilled_prompt.replace("!hyp!", ins_dict["hyp"])
        # print(fulfilled_prompt.to_string())
        try:
            result = run_gpt(fulfilled_prompt.to_string())
        except:
            print("sleeping 10s ...")
            sleep(10)
            result = run_gpt(fulfilled_prompt.to_string())
        # print(result)
        ins_dict["gpt_output"] = json.loads(result)

        out_json_file.write(json.dumps(ins_dict) + "\n")
    out_json_file.close()


def test_set(dataset):
    d_instances = json.load(open(dataset, "r"))
    # output format should include the following fields:
    # id, label, p(Hallucination)
    # p(Hallucination) is the probability of hallucination
    # label is the label of the instance "Hallucination" or "Not Hallucination"
    # id is the id of the instance provided in the dataset

    # output format should be a json file made by calling dump_instances()

    # call gpt to get the label
    endlist = []
    for ins in tqdm.tqdm(d_instances):
        ins_dict = dict(ins)
        if ins_dict["tgt"] == "":
            # print("SWAPPED tgt for src")
            fulfilled_prompt = pg_intermediate_prompt.replace("!tgt!", ins_dict["src"])
            fulfilled_prompt = fulfilled_prompt.replace("!hyp!", ins_dict["hyp"])
        else:
            fulfilled_prompt = general_pipeline_prompt.replace("!tgt!", ins_dict["tgt"])
            fulfilled_prompt = fulfilled_prompt.replace("!hyp!", ins_dict["hyp"])
        # print(fulfilled_prompt.to_string())
        try:
            result = run_gpt(fulfilled_prompt.to_string())
        except:
            print("sleeping 10s ...")
            sleep(10)
            result = run_gpt(fulfilled_prompt.to_string())
        ins_dict["gpt_output"] = json.loads(result)
        ins_dict["label"] = "Hallucination" if ins_dict["gpt_output"]["answer"] == "Yes" else "Not Hallucination"
        ins_dict["p(Hallucination)"] = 1 if ins_dict["label"] == "Hallucination" else 0
        del ins_dict["gpt_output"]

        endlist.append(ins_dict)
    dump_instances(endlist, OUT_FILE_PATH.joinpath(f"out.json"))

