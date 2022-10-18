from typing import List, Dict, Tuple
import json
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from itertools import dropwhile
from collections import Counter
from operator import add
from functools import reduce
from overrides import overrides


STOPWORDS = set(stopwords.words("english"))
ps = PorterStemmer()


class DNC(object):
    def __init__(self) -> None:
        relation_path_train = "DNC/train/recast_kg_relations_data.json"
        relation_path_test = "DNC/test/recast_kg_relations_data.json"
        entity_path_train = "DNC/test/recast_ner_data.json"

        self.data = [
            Relation_classification(relation_path_train),
            Entity_recognition(entity_path_train),
        ]

    # TODO: implement a sampling generator, for item in DNC()

    # def __iter__(self):
    # def __next__(self):


class DNC_base(object):
    def __init__(self, path) -> None:

        self.ori_data_list = json.load(open(path, "r"))

        self.hypothesis_list = [item["hypothesis"] for item in self.ori_data_list]

        self.data = NotImplementedError("abstract member")
        self.label_set = NotImplementedError("abstract member")
        self.class2datalist = NotImplementedError("abstract member")

    @staticmethod
    def sent_preprocessing(sent: str, remove_stopwords_punc: bool) -> List[str]:
        if remove_stopwords_punc:
            word_list = [
                ps.stem(w.lower())
                for w in nltk.word_tokenize(sent)
                if w.lower() not in STOPWORDS and w not in string.punctuation
            ]
        else:
            word_list = [ps.stem(w.lower()) for w in nltk.word_tokenize(sent)]
        return word_list

    def dnc_preprocessing(self, remove_stopwords_punc: bool):
        data = self.ori_data_list
        data = self._filter_out_false_pair(data)
        data = self._add_class_label(data, remove_stopwords_punc)
        class2datalist = self._data_list2dict(data)

        self.data = data
        self.class2datalist = class2datalist

        # print({k: len(v) for k, v in class2datalist.items()})
        return data, class2datalist

    @staticmethod
    def _filter_out_false_pair(data):
        return [d for d in data if d["binary-label"] is True]

    def _add_class_label(self, data, remove_stopwords_punc):
        result = []
        match_list = sorted(self.label_set, key=lambda x: -len(x))
        # print(match_list)
        for d in data:
            sent = " ".join(
                self.sent_preprocessing(
                    d["hypothesis"], remove_stopwords_punc=remove_stopwords_punc
                )
            )
            for w in match_list:

                if sent.endswith(w) or w in sent:
                    d["class"] = w
                    result.append(d)
                    break

        return result

    def _data_list2dict(self, data):
        dic = {k: [] for k in self.label_set}
        for d in data:
            dic[d["class"]].append(d)
        return dic

    def print_label2sent_eg(self) -> None:
        raise NotImplementedError("abs method")

    def __len__(self) -> int:
        return len(self.hypothesis_list)


class Relation_classification(DNC_base):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.label_set = set(
            "play,serv,work,member,locat,produc,found,air,son,citi,base,replac,develop,run,flow,leader,repres,host,wrote,oper,compet,live,manag,part,appoint,partner,belong,own,divis,flew,coach,succeed,influenc,use,subsidiari,creat,releas,command,station,follow,design,buri,written,includ,govern,student,built,border,album,child,associ,capit,appear,head,provid,train,field,licens,draft,situat,mother,judg,sign,practic,cross,marri,affili,languag,broadcast,direct,relat,publish,agenc,fli,parti,nicknam,servic,power,origin,voic,die,deputi,style,fought,currenc,join,hit,born,court,graduat,captain,spoken,black,race,forc,interpret,affil,tributari,suburb,support,lead,edit,studi,relev,citizen,municip,saint,present,celebr,meet,headquart,establish,nomin".split(
                ","
            )
        )

        self.dnc_preprocessing(remove_stopwords_punc=True)

    @property
    def hypothesis_cnt(self):
        return Counter(
            reduce(
                add, (self.sent_preprocessing(sent) for sent in self.hypothesis_list)
            )
        )

    @overrides
    def print_label2sent_eg(self) -> None:
        _label_candidates = ",".join(
            [k for k, v in self.hypothesis_cnt.most_common(200)]
        )
        label_candidates = self.label_set
        print(_label_candidates)
        print("-" * 80)
        for label in label_candidates:
            for sent in self.hypothesis_list:
                if label in self.sent_preprocessing(sent, remove_stopwords_punc=True):
                    print(label, "\t", sent)
                    break


class Entity_recognition(DNC_base):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.label_set = set(
            "a locat,an organ,a person,a person 's titl,a time or durat of time,a natur entiti,a day of the month,a geographical/polit entiti,a famou person,a calendar year,a person 's given name,a day of the week,an event,a time on the clock,an artefact,a month".split(
                ","
            )
        )

        self.dnc_preprocessing(remove_stopwords_punc=False)

    @overrides
    def print_label2sent_eg(self) -> None:
        def drop_be(sent: str) -> str:
            words = self.sent_preprocessing(sent, remove_stopwords_punc=False)
            words = " ".join(
                list(dropwhile(lambda x: x not in ("is", "are", "was", "were"), words))[
                    1:
                ]
            )
            return words

        cnt = Counter(map(drop_be, self.hypothesis_list))
        # print(','.join(cnt.keys()))
        print(cnt)


if __name__ == "__main__":
    relation_path_train = "DNC/train/recast_kg_relations_data.json"
    relation_path_test = "DNC/test/recast_kg_relations_data.json"
    entity_path_train = "DNC/test/recast_ner_data.json"

    # Relation_classification(relation_path_train).print_label2sent_eg()

    # Relation_classification(relation_path_train)
    Entity_recognition(entity_path_train)
