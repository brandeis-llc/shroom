import json
import re
from file_path import OUT_FILE_PATH


def evaluate_acc(out_json_file):
    label_mapping = {
        "Not Hallucination": 0,
        "Hallucination": 1,
        "No": 0,
        "Yes": 1,
    }
    labels = []
    preds = []
    with open(out_json_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            labels.append(label_mapping[obj["label"]])
            pred_label = obj["gpt_output"]["answer"]
            if pred_label in {"Yes", "No"}:
                preds.append(label_mapping[pred_label])
            else:
                pred_label = set(re.sub("[^a-zA-Z]+", " ", pred_label).split())
                if "Yes" in pred_label and "No" in pred_label:
                    preds.append(2)
                elif "Yes" in pred_label:
                    preds.append(1)
                elif "No" in pred_label:
                    preds.append(0)
                else:
                    preds.append(-1)

    corr = 0
    total = 0
    for l, p in zip(labels, preds):
        if l == p:
            corr += 1
        total += 1
    return corr / total * 100


if __name__ == "__main__":
    print("GPT-4 Prompt 1")
    dm_res = evaluate_acc(OUT_FILE_PATH.joinpath("dm_out_gpt.jsonl"))  # 75%
    print("DM", dm_res)
    mt_res = evaluate_acc(OUT_FILE_PATH.joinpath("mt_out_gpt.jsonl"))  # 83%
    print("MT", mt_res)
    pg_res = evaluate_acc(OUT_FILE_PATH.joinpath("pg_out_gpt.jsonl"))  # 72%
    print("PG", pg_res)
    print("Accuracy", (dm_res * 187 + mt_res * 187 + pg_res * 125) / (187 + 187 + 125)) # 77%
    print()

    print("GPT-4 Prompt 2")
    dm_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_DM_4.jsonl"))  # 75%
    print("DM", dm_res)
    mt_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_MT_4.jsonl"))  # 85%
    print("MT", mt_res)
    pg_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_PG_4.jsonl"))  # 70%
    print("PG", pg_res)
    print("Accuracy", (dm_res * 187 + mt_res * 187 + pg_res * 125)/(187+187+125)) # 78%
    print()

    print("GPT-4 Prompt 2 (Model Aware)")
    dm_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_DM.jsonl"))  # 76%
    print("DM", dm_res)
    mt_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_MT.jsonl"))  # 75%
    print("MT", mt_res)
    #pg_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_PG.jsonl"))  # XX%
    #print("PG", pg_res)
    #print("Accuracy", (dm_res * 187 + mt_res * 187 + pg_res * 125) / (187 + 187 + 125))  # XX%
    print()

    print("GPT-3 Prompt 1")
    dm_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_DM_3.jsonl"))  # 65%
    print("DM", dm_res)
    mt_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_MT_3.jsonl"))  # 73%
    print("MT", mt_res)
    pg_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_PG_3.jsonl"))  # 58%
    print("PG", pg_res)
    print("Accuracy", (dm_res * 187 + mt_res * 187 + pg_res * 125)/(187+187+125))  # 66%
    print()

    print("GPT-4 Prompt 3 (with examples)")
    dm_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_DM_4e.jsonl"))  # 65%
    print("DM", dm_res)
    mt_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_MT_4e.jsonl"))  # 73%
    print("MT", mt_res)
    pg_res = evaluate_acc(OUT_FILE_PATH.joinpath("gpt_out_PG_4e.jsonl"))  # 58%
    print("PG", pg_res)
    print("Accuracy", (dm_res * 187 + mt_res * 187 + pg_res * 125) / (187 + 187 + 125))  # 75%
    print()

    print("======AWARE======")
    print("GPT4 V1 Prompt Results")
    dm_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_DM.jsonl"))
    print("DM", dm_res)
    mt_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_MT.jsonl"))
    print("MT", mt_res)
    pg_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_PG.jsonl"))
    print("PG", pg_res)
    print("Accuracy", (dm_res * 187 + mt_res * 187 + pg_res * 125) / (187 + 187 + 125))
    print()

    print("GPT4 V2 Prompt Results")
    dm_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_DM_tim.jsonl"))
    print("DM", dm_res)
    mt_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_MT_tim.jsonl"))
    print("MT", mt_res)
    pg_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_PG_tim.jsonl"))
    print("PG", pg_res)
    print("Accuracy", (dm_res * 187 + mt_res * 187 + pg_res * 125) / (187 + 187 + 125))
    print()

    print("GPT3 JX Prompt Results")
    dm_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_DM_gpt3.jsonl"))
    print("DM", dm_res)
    mt_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_MT_gpt3.jsonl"))
    print("MT", mt_res)
    pg_res = evaluate_acc(OUT_FILE_PATH.joinpath("aware_gpt_PG_gpt3.jsonl"))
    print("PG", pg_res)
    print("Accuracy", (dm_res * 187 + mt_res * 187 + pg_res * 125) / (187 + 187 + 125))
    print()


