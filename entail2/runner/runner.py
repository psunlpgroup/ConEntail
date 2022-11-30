# from entail2.model.entail2 import Entail2_Test
from entail2.dataloader import gym_finetune, gym_efl_finetune
from entail2.model import get_encoder, get_model
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from torch import max_pool1d_with_indices, nn, optim
from transformers import AdamW, get_linear_schedule_with_warmup

from entail2.dataloader import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import sys
import os
import json
import random
import argparse
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Runner(nn.Module):
    def __init__(self, args):
        super().__init__()

        # bert_name = args.bert_name
        model_name = args.model_name
        batch_size = args.train_batch_size
        max_epoch = args.num_train_epochs
        learning_rate = args.learning_rate
        warmup_ratio = args.warmup_ratio
        use_sampler = args.use_sampler
        training_shots = args.training_shots
        # if_finetune_metatest = args.finetune_metatest
        test_shots = args.test_shots

        self.model_name = model_name
        self.model_class = get_model(model_name=model_name)

        bert_name = {
            "entail2": "bert",
            "efl_no_cl": "bert",
            "efl": "bert",
            "efl_multichoice": "bert",
            "crossfit": "bart",
            "unifew": "bart",
        }[model_name]

        encoder_class = get_encoder(bert_name)
        self.encoder = encoder_class.encoder
        self.tokenizer = encoder_class.tokenizer

        self.use_sampler = use_sampler
        self.model = self.model_class(self.encoder, self.tokenizer)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.thr = 0.5
        self.test_shots = test_shots
        self.training_shots = training_shots
        # self.if_finetune_metatest = if_finetune_metatest
        # self.ckpt = "ckpt/entail2.ckpt"
        self.ckpt = os.path.join(
            "ckpt", model_name + "_" + str(training_shots) + ".ckpt"
        )
        self.out_dir = "out_dir"

        self.mkdir()

        if model_name == "efl_no_cl":
            loader_funs = [gym_efl]
        else:
            loader_funs = [gym]

        self.chain_train_loader = Chain_dataloader(
            loader_funs,
            batch_size,
            "train",
            self.tokenizer,
            self.use_sampler,
            self.training_shots,
        )
        # self.chain_eval_loader = Chain_dataloader(
        # loader_funs, batch_size, "eval", self.tokenizer
        # )
        self.total_steps = len(self.chain_train_loader) * self.max_epoch
        self.warmup_steps = int(self.total_steps * warmup_ratio)

    def init_optim(self):
        parameters_to_optimize = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        parameters_to_optimize = [
            {
                "params": [
                    p
                    for n, p in parameters_to_optimize
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            parameters_to_optimize, lr=self.learning_rate, correct_bias=False
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

    def mkdir(self):
        root_path = "."
        sys.path.append(root_path)
        if not os.path.exists("ckpt"):
            os.mkdir("ckpt")
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    # def data_summary(self, data_dir, task_dir):
    #     join_dir = os.path.join(data_dir, task_dir)
    #     support_path = test_path = FileNotFoundError
    #     files = sorted(os.listdir(join_dir))
    #     for f in files:
    #         if f.endswith("_support_shot_0.json"):
    #             support_path = os.path.join(join_dir, f)

    #         if f.endswith(".json") and "test" in f:
    #             test_path = os.path.join(join_dir, f)

    #     print(len(open(support_path, 'r').readlines()), len(open(test_path, 'r').readlines()))

    def train_model(self):
        self.model = self.model.cuda()
        self.init_optim()
        global_step = 0

        best_metric = 0
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            pbar = tqdm(
                enumerate(self.chain_train_loader), total=len(self.chain_train_loader)
            )
            for batch_idx, batch in pbar:
                # for batch_idx, batch in enumerate(self.chain_train_loader):
                global_step += 1
                for k in batch:
                    batch[k] = batch[k].cuda()

                output = self.model(**batch)
                loss = output["loss"]

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                pbar.set_description(
                    "Loss: {:.2f}, epoch: {}/{}:".format(
                        loss.item(), epoch, self.max_epoch
                    )
                )
                # torch.save(self.model,
                # self.ckpt,
                # )
                # print("success save!")
                # exit()
            # # Val
            # torch.save(
            #     {"state_dict": self.model.encoder.state_dict()},
            #     self.ckpt,
            # )
            torch.save(
                self.model,
                self.ckpt,
            )
            # if global_step >= self.total_steps:
            #     break

            # acc = self.eval_model(ckpt=None)

            # if acc > best_metric:
            #     torch.save(
            #         {"state_dict": self.model.encoder.state_dict()},
            #         self.ckpt,
            #     )
            #     best_metric = acc
            #     print("Best ckpt and saved.")

            # print("=== Epoch %d val ===" % epoch)
            # result = self.eval_model(self.val_loader)
            #     print("AUC: %.4f" % result["auc"])
            #     print("Micro F1: %.4f" % (result["micro_f1"]))
            #     if result[metric] > best_metric:
            #         print("Best ckpt and saved.")
            #         torch.save({"state_dict": self.model.module.state_dict()}, self.ckpt)
            #         best_metric = result[metric]
            # print("Best %s on val set: %f" % (metric, best_metric))

    # def eval_model(self, ckpt):
    #     if ckpt is not None:
    #         self.encoder = self.load_state_dict(torch.load(ckpt)["state_dict"])
    #         self.model.set_encoder(self.encoder)
    #         self.model = self.model.cuda()
    #         print("load trained model successfully.")

    #     self.model.eval()
    #     y_pred = []
    #     y_true = []
    #     y_base = []
    #     with torch.no_grad():
    #         pbar = tqdm(
    #             enumerate(self.chain_eval_loader), total=len(self.chain_eval_loader)
    #         )
    #         for batch_idx, batch in pbar:
    #             for k in batch:
    #                 batch[k] = batch[k].cuda()
    #             output = self.model(**batch)
    #             loss = output["loss"]
    #             sim = output["sim"]
    #             y_base.extend(torch.ones_like(sim).view(-1).long().tolist())
    #             y_pred.extend((sim > self.thr).long().view(-1).tolist())
    #             y_true.extend(output["blabel"].view(-1).long().tolist())
    #     p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    #     pb, rb, fb, _ = precision_recall_fscore_support(y_true, y_base, average="micro")

    #     acc = accuracy_score(y_true, y_pred)
    #     print("Model\tF: {:.2f}, P: {:.2f}, R: {:.2f}".format(f, p, r))
    #     print("Baseline\tF: {:.2f}, P: {:.2f}, R: {:.2f}".format(fb, pb, rb))
    #     print("Accuracy\t: {:.2f} ({:.2f})".format(acc, accuracy_score(y_true, y_base)))
    #     return acc

    def finetune_single(
        self, data_dir, task_dir, shots, times, no_load_supervised_pertrained=False
    ):
        join_dir = os.path.join(data_dir, task_dir)
        support_path = test_path = FileNotFoundError
        files = sorted(os.listdir(join_dir))
        for f in files:
            if f.endswith("_support_shot_%d_seed_%d.json" % (shots, times)):
                support_path = os.path.join(join_dir, f)

            if f.endswith("_test.json"):
                test_path = os.path.join(join_dir, f)

        # self.encoder = self.load_state_dict(torch.load(self.ckpt)["state_dict"])

        # self.model.set_encoder(self.encoder)
        if not no_load_supervised_pertrained:
            self.model = torch.load(self.ckpt)
            print("load supervised pretrained model successfully.")
        else:
            self.model = self.model_class(self.encoder, self.tokenizer)
            print("load BERT/BART successfully.")

        self.model = self.model.cuda()
        self.model.train()
        self.init_optim()

        if self.model_name == "efl_no_cl":
            finetune_loader, _ = gym_efl_finetune(support_path, self.tokenizer)
        else:
            finetune_loader, _ = gym_finetune(support_path, self.tokenizer)

        for epoch in range(self.max_epoch):
            # for epoch in range(2):

            pbar = tqdm(
                enumerate(finetune_loader), total=len(finetune_loader), disable=True
            )
            for batch_idx, batch in pbar:
                #  for batch_idx, batch in enumerate(finetune_loader):

                for k in batch:
                    batch[k] = batch[k].cuda()

                output = self.model(**batch)
                loss = output["loss"]

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                pbar.set_description(
                    "Loss: {:.2f}, epoch: {}/{}:".format(
                        loss.item(), epoch, self.max_epoch
                    )
                )
        print("Fine-tuning finished!")
        self.model.eval()
        batch_sz = 8
        test_loader, _ = gym_test(
            batch_sz=batch_sz,
            support_path=support_path,
            test_path=test_path,
            tokenizer=self.tokenizer,
        )
        y_pred = []
        y_true = []
        y_base = []
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for batch_idx, batch in pbar:
                for k in batch:
                    batch[k] = batch[k].cuda()

                result = self.model.predict(**batch)
                y_base.extend(result["y_base"])
                y_true.extend(result["y_true"])
                y_pred.extend(result["y_pred"])

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
        pb, rb, fb, _ = precision_recall_fscore_support(y_true, y_base, average="micro")

        acc = accuracy_score(y_true, y_pred)
        # print("Model\tF: {:.3f}, P: {:.2f}, R: {:.2f}".format(f, p, r))
        # print("Baseline\tF: {:.2f}, P: {:.2f}, R: {:.2f}".format(fb, pb, rb))
        print("Accuracy\t: {:.2f} ({:.2f})".format(acc, accuracy_score(y_true, y_base)))
        result = {
            "model": {"f": f, "p": p, "r": r},
            "baseline": {"f": fb, "p": pb, "r": rb},
            "acc": {"pred": acc, "base": accuracy_score(y_true, y_base)},
            "shots": shots,
            "times": times,
        }
        return result

    def case_studies(self, support_file_name, test_file_name):

        # join_dir = os.path.join(data_dir, task_dir)
        support_path = os.path.join("cases", support_file_name)
        test_path = os.path.join("cases", test_file_name)
        # support_path = test_path = FileNotFoundError
        # files = sorted(os.listdir(join_dir))
        # for f in files:
        #     if f.endswith("_support_shot_0.json"):
        #         support_path = os.path.join(join_dir, f)

        #     # if f.endswith(".json") and "test" in f:
        #     if f.endswith("_test.json"):
        #         test_path = os.path.join(join_dir, f)

        # self.encoder = self.load_state_dict(torch.load(self.ckpt)["state_dict"])

        # self.model.set_encoder(self.encoder)
        self.model = torch.load(self.ckpt)
        self.model = self.model.cuda()
        self.model.eval()
        print("load trained model successfully.")

        batch_sz = 2
        test_loader, _ = gym_test(
            batch_sz=batch_sz,
            support_path=support_path,
            test_path=test_path,
            tokenizer=self.tokenizer,
        )

        top_sim = []
        pred_label = []
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for batch_idx, batch in pbar:
                for k in batch:
                    batch[k] = batch[k].cuda()

                result = self.model.topk_predict(**batch)
                top_sim.extend(result["top_sim"])
                pred_label.extend(result["pred_label"])
        return {"top_sim": top_sim, "pred_label": pred_label}

    def test_gym_model_on_one_dataset(self, data_dir, task_dir, shots, times):
        join_dir = os.path.join(data_dir, task_dir)
        support_path = test_path = FileNotFoundError
        files = sorted(os.listdir(join_dir))
        for f in files:
            if f.endswith("_support_shot_0.json"):
                support_path = os.path.join(join_dir, f)

            # if f.endswith(".json") and "test" in f:
            if f.endswith("_test.json"):
                test_path = os.path.join(join_dir, f)

        # self.encoder = self.load_state_dict(torch.load(self.ckpt)["state_dict"])

        # self.model.set_encoder(self.encoder)
        self.model = torch.load(self.ckpt)
        self.model = self.model.cuda()
        self.model.eval()
        print("load trained model successfully.")

        batch_sz = 32
        test_loader, _ = gym_test(
            batch_sz=batch_sz,
            support_path=support_path,
            test_path=test_path,
            tokenizer=self.tokenizer,
        )
        y_pred = []
        y_true = []
        y_base = []
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for batch_idx, batch in pbar:
                for k in batch:
                    batch[k] = batch[k].cuda()

                result = self.model.predict(**batch)
                y_base.extend(result["y_base"])
                y_true.extend(result["y_true"])
                y_pred.extend(result["y_pred"])

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
        pb, rb, fb, _ = precision_recall_fscore_support(y_true, y_base, average="micro")

        acc = accuracy_score(y_true, y_pred)
        print("Model\tF: {:.3f}, P: {:.2f}, R: {:.2f}".format(f, p, r))
        print("Baseline\tF: {:.2f}, P: {:.2f}, R: {:.2f}".format(fb, pb, rb))
        print("Accuracy\t: {:.2f} ({:.2f})".format(acc, accuracy_score(y_true, y_base)))
        result = {
            "model": {"f": f, "p": p, "r": r},
            "baseline": {"f": fb, "p": pb, "r": rb},
            "acc": {"pred": acc, "base": accuracy_score(y_true, y_base)},
            "shots": shots,
            "times": times,
        }
        return result

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict)
        return self.encoder


if __name__ == "__main__":
    mode_list = ["train", "test", "finetune_test", "data_summary", "case"]
    model_name_list = [
        "entail2",
        "efl",
        "efl_multichoice",
        "efl_no_cl",
        "crossfit",
        "unifew",
    ]
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt", default="ckpt/entail2.ckpt", help="Checkpoint name")

    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.005,
        type=float,
        help="The initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--predict_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument("--bert_name", default="bert", required=False)
    # parser.add_argument("--model_name", default="entail2", required=False)
    parser.add_argument("--task_dir", default="glue-sst2", required=False)
    parser.add_argument("--data_dir", default="raw_data/gym", required=False)
    # Seed
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument("--mode", choices=mode_list)
    parser.add_argument("--model_name", choices=model_name_list)
    # parser.add_argument("--finetune_metatest", action="store_true")
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument(
        "--test_shots", default=5, type=int, help="shot number on meta-test phase"
    )
    parser.add_argument(
        "--test_times",
        default=20,
        type=int,
        help="test times on different support sets",
    )
    parser.add_argument("--training_shots", type=int, default=128)
    parser.add_argument("--case_support_file", default="support.json", required=False)
    parser.add_argument("--case_test_file", default="test.json", required=False)
    # Set random seed
    parser.add_argument("--no_load_supervised_pertrained", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    # bertname = "bert-base-uncased"
    # train_sets = [WikitextDataset("train", 5), WordnetDataset("train", 5)]
    # loader_funs = [fewrel, wikitext, wordnet]
    # loader_funs = [gym, fewrel]
    # loader_funs = [gym]

    runner = Runner(args=args)
    # runner.data_summary()
    # runner.train_model()
    if args.mode == "test":
        with open(
            os.path.join(
                args.data_dir,
                args.task_dir,
                args.model_name
                + "test_shots_"
                + str(args.test_shots)
                + "training_shots_"
                + str(args.training_shots)
                + "_result.json",
            ),
            "w",
        ) as f:
            for t in range(args.test_times):
                result_dict = runner.test_gym_model_on_one_dataset(
                    args.data_dir, args.task_dir, shots=args.test_shots, times=t
                )
                result_str = json.dumps(result_dict)
                f.write(result_str)
                f.write("\n")
    elif args.mode == "case":
        with open(
            os.path.join(
                "cases",
                args.case_test_file[:-5]
                + "_"
                + args.case_support_file[:-5]
                + "_result.txt",
            ),
            "w",
        ) as f:
            result_dict = runner.case_studies(
                args.case_support_file, args.case_test_file
            )
            result_str = json.dumps(result_dict)
            f.write(result_str)
            f.write("\n")
    elif args.mode == "train":
        runner.train_model()

    elif args.mode == "finetune_test":
        with open(
            os.path.join(
                args.data_dir,
                args.task_dir,
                args.model_name
                + "_shots_"
                + str(args.test_shots)
                + "_finetune_result.json",
            ),
            "w",
        ) as f:
            for t in range(args.test_times):
                result_dict = runner.finetune_single(
                    args.data_dir,
                    args.task_dir,
                    shots=args.test_shots,
                    times=t,
                    no_load_supervised_pertrained=args.no_load_supervised_pertrained,
                )
                result_str = json.dumps(result_dict)
                f.write(result_str)
                f.write("\n")
    # elif args.mode == "data_summary":
    #     runner.data_summary(
    #                 args.data_dir, args.task_dir
    #             )
    else:
        print("please pick a mode")
        exit()
    # runner.eval_model(args.ckpt)
    # runner.submmit_fewrel_all(args.ckpt)
