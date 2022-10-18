from entail2.dataloader import *


class Test_Dataloader:
    def test_chain_loader(self):
        loader_funs = [wordnet, wikitext]

        # chainloader = Chain_dataloader(loader_funs, 32, "eval")
        # len_chain = len(chainloader)
        # len_wordnet = len(wordnet(32)[1])
        # len_wiki = len(wikitext(32)[1])

        # print(len_chain)
        # print(len_wiki)
        # print(len_wordnet)
        wordnet_train, wordnet_test = wordnet(32)
        wiki_train, wiki_test = wikitext(32)

        # assert len(chainloader) == (len(wordnet(32)[1]) + len(wikitext(32)[1]))
        i = 0
        for _, b in enumerate(wordnet_test):
            i += 1
        assert i == len(wordnet_test)

        i = 0
        for _, b in enumerate(wiki_test):
            i += 1
        assert i == len(wiki_test)

        i = 0
        for _, b in enumerate(wordnet_train):
            i += 1
        assert i == len(wordnet_train)

        i = 0
        for _, b in enumerate(wiki_train):
            i += 1
        assert i == len(wiki_train)
