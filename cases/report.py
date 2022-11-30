import enum
import os
import re
import json
from typing import List
from unittest import case

TEST = "cases/test.json"
RESULT = "cases/result.json"

TEST = "cases/covid_test.json"
RESULT = "cases/covid_test_covid_support_result.txt"

# TEST = "cases/topic_test.json"
# RESULT = "cases/topic_test_topic_support_result.txt"


def read_files_for_up_down():
    cases = []
    with open(TEST, "r") as f_t, open(RESULT, "r") as f_r:
        result_dic = json.load(f_r)
        pred_label = result_dic["pred_label"]
        sim = result_dic["top_sim"]
        for line, label, sim in zip(f_t, pred_label, sim):
            sent = json.loads(line)["in"]
            cases.append((sent, label[:8] + [label[-1]], sim[:8] + [sim[-1]]))
    return cases


def read_files_for_left_right():
    cases = []
    with open(TEST, "r") as f_t, open(RESULT, "r") as f_r:
        result_dic = json.load(f_r)
        pred_label = result_dic["pred_label"]
        sim = result_dic["top_sim"]
        for line, label, sim in zip(f_t, pred_label, sim):
            sent = json.loads(line)["in"]
            cases.append((sent, label, sim))
    return cases


def join_labels(s):
    for i in range(len(s)):
        if "-" in s[i]:
            s[i] = "".join(s[i].split(" "))
    return s


def case_list_to_up_down_label(cases: List[str], begin, end):
    bolddiff = 30
    cell_size = 0.09
    num_size = 0.03
    tabular = (
        "\\begin{table*}[] \
    \\scriptsize \
    \\centering \
    \\extrarowheight=\\aboverulesep \
    \\addtolength{\\extrarowheight}{\\belowrulesep} \
    \\aboverulesep=1pt \
    \\belowrulesep=1pt \
\\begin{tabular}{p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth}} \
\\toprule    "
        % (
            cell_size,
            num_size,
            cell_size,
            num_size,
            cell_size,
            num_size,
            cell_size,
            num_size,
            cell_size,
            num_size,
        )
    )
    for i, case in enumerate(cases):
        if i < begin:
            continue
        if i >= end:
            break
        sent, label, sim = case
        label = join_labels(label)
        tabular += "\\multicolumn{10}{p{0.9\\textwidth}}{" + tex_escape(sent) + "} \\\\"
        tabular += "\n"
        tabular += (
            " & ".join(
                (
                    "\\cellcolor{gray!%f}" % max(100 * s - bolddiff, 0)
                    + tex_escape(l)
                    + " & "
                    + "\\cellcolor{gray!%f}" % max(100 * s - bolddiff, 0)
                    + ("%.2f" % s)
                )
                for s, l in zip(sim[:5], label[:5])
            )
            + " \\\\"
        )
        tabular += "\n"
        # tabular += (
        #     " & ".join(
        #         "\\cellcolor{gray!%f}" % max(100 * s - bolddiff, 0) + ("%.2f" % s)
        #         for s, l in zip(sim[:5], label[:5])
        #     )
        #     + " \\\\"
        # )
        tabular += (
            " & ".join(
                (
                    "\\cellcolor{gray!%f}" % max(100 * s - bolddiff, 0)
                    + tex_escape(l)
                    + " & "
                    + "\\cellcolor{gray!%f}" % max(100 * s - bolddiff, 0)
                    + ("%.2f" % s)
                )
                for s, l in zip(sim[5:8], label[5:8])
            )
            + " & ... & ... &"
            + tex_escape(label[-1])
            + " & "
            + ("%.2f" % sim[-1])
            + " \\\\"
        )
        tabular += "\n"
        # tabular += (
        #     " & ".join(
        #         "\\cellcolor{gray!%f}" % max(100 * s - bolddiff, 0) + ("%.2f" % s)
        #         for s, l in zip(sim[5:8], label[5:8])
        #     )
        #     + " & ... & "
        #     + ("%.2f" % sim[-1])
        #     + " \\\\"
        # )
        tabular += "\n"
        tabular += "\\midrule"
        tabular += "\n"
        # if i >= 10:
        #     break
    tabular += "\\end{tabular} \
                \\end{table*}"
    return tabular


def case_list_to_left_right_label(cases: List[str], begin, end):
    sentiment_list = ["positive", "negative", "happy", "anger", "hate", "non-hate"]
    bolddiff = 30
    sent_size = 0.6
    # cell_size = 0.0
    num_size = 0.04
    tabular = (
        "\\begin{table*}[] \
    \\scriptsize \
    \\centering \
    \\extrarowheight=\\aboverulesep \
    \\addtolength{\\extrarowheight}{\\belowrulesep} \
    \\aboverulesep=1pt \
    \\belowrulesep=1pt \
\\begin{tabular}{p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth} p{%f\\textwidth}} \
\\toprule    "
        % (
            sent_size,
            num_size,
            num_size,
            num_size,
            num_size,
            num_size,
            num_size,
        )
    )
    tabular += "sentence & " + " & ".join(sentiment_list) + "\\\\"
    for i, case in enumerate(cases):
        if i < begin:
            continue
        if i >= end:
            break
        sent, label, sim = case
        label = join_labels(label)
        dic = {l: s for l, s in zip(label, sim)}
        new_sim = [dic[l] for l in sentiment_list]
        tabular += tex_escape(sent) + " & "
        tabular += " & ".join(("%.2f" % s) for s in new_sim) + " \\\\"
        tabular += "\n"
        tabular += "\\midrule"
        tabular += "\n"
    tabular += "\\end{tabular} \
            \\end{table*}"
    return tabular


def tex_escape(text):
    """
    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    regex = re.compile(
        "|".join(
            re.escape(str(key))
            for key in sorted(conv.keys(), key=lambda item: -len(item))
        )
    )
    return regex.sub(lambda match: conv[match.group()], text)


def gen_updown_table():
    cases = read_files_for_up_down()

    tabular1 = case_list_to_up_down_label(cases, 0, 8)
    tabular2 = case_list_to_up_down_label(cases, 6, 12)
    tabular3 = case_list_to_up_down_label(cases, 28, 34)
    tabular4 = case_list_to_up_down_label(cases, 34, 42)
    tabular5 = case_list_to_up_down_label(cases, 37, 48)
    # print(tabular1)
    # print(tabular2)
    # print(tabular3)
    # print(tabular4)

    print(tabular5)


def gen_leftright_table():
    cases = read_files_for_left_right()
    tabular1 = case_list_to_left_right_label(cases, 0, 6)
    print(tabular1)


if __name__ == "__main__":
    # cases = read_files_for_up_down()

    gen_updown_table()
    # gen_leftright_table()
