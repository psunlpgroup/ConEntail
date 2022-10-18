from entail2.dataloader.wikitext2entail import wikitext
from entail2.dataloader.gym2entail_multitask import gym

# from entail2.dataloader.gym2entail_singletask import gym_single

# from entail2.dataloader.wordnet2entail import wordnet
from entail2.dataloader.fewrel2entail import fewrel, fewrel_submmit
from entail2.dataloader.base import Map_CH_dataset, dataset2loader
from entail2.dataloader.chain_dataset import Chain_dataloader
from entail2.dataloader.gym_test_single import gym_test
from entail2.dataloader.gym_enumerate_label_train import gym_efl
from entail2.dataloader.gym_test_finetune_single import gym_efl_finetune, gym_finetune
