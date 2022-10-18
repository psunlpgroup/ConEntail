import os
import json
import csv
from pprint import pprint


def read_txt(path):
    result = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            dic = {
                "in": line,
                "out": ["unknown"],
                "task_name": "topic classification",
            }
            result.append(dic)
    return result


def write_json(result, path):
    with open(path, "w") as f:
        for line in result:
            f.write(json.dumps(line))
            f.write("\n")


if __name__ == "__main__":
    result = read_txt("topic_test.txt")
    write_json(result, "topic_test.json")
    pprint(result[:5])
    print("*" * 50)
