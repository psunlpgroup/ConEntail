import os


def data_summary(task_dir):
    data_dir = "raw_data/gym"
    join_dir = os.path.join(data_dir, task_dir)
    support_path = test_path = FileNotFoundError
    files = sorted(os.listdir(join_dir))
    for f in files:
        if f.endswith("_support_shot_0.json"):
            support_path = os.path.join(join_dir, f)

        if f.endswith(".json") and "test" in f:
            test_path = os.path.join(join_dir, f)

    support_lines = open(support_path, "r").readlines()
    test_lines = open(test_path, "r").readlines()
    print(len(support_lines), len(test_lines))


if __name__ == "__main__":
    tasks = "glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive".split()
    for t in tasks:
        data_summary(t)
