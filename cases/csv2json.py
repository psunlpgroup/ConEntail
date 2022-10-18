import os
import json
import csv
from pprint import pprint


def read_csv(path, skip_head=True):
    result = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            if skip_head and i == 0:
                continue
            else:
                sentiment, subtype, sent = line
                dic = {
                    "in": sent,
                    "out": [subtype + " " + sentiment],
                    "task_name": "covid sentiment",
                }
                result.append(dic)
    return result


def write_json(result, path):
    with open(path, "w") as f:
        for line in result:
            f.write(json.dumps(line))
            f.write("\n")


def gen_labels(result, path):
    # label_set = set(r["out"][0] for r in result)
    label_set = {
        "mild",
        "anger",
        "irony",
        "hate",
        "positive",
        "negative",
        "happy",
        "offensive",
        "support",
        "refute",
        "sad",
    }
    with open(path, "w") as f:
        for label in label_set:
            dic = {"in": "null", "out": [label], "task_name": "covid sentiment"}
            f.write(json.dumps(dic))
            f.write("\n")


if __name__ == "__main__":
    result = read_csv("covid_test.csv")
    write_json(result, "covid_test.json")
    gen_labels(result, "covid_support.json")
    pprint(result[:5])
    print("*" * 50)
