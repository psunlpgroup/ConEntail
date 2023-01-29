
# ConEntail
Source code for **EACL 2023 Paper** [ConEntail: An Entailment-based Framework for Universal Zero and Few Shot Classification with Supervised Contrastive Pretraining](https://arxiv.org/pdf/2210.07587.pdf)

## Supervised Pretraining Data

You can either download our preprocessed supervised pretrained data (128 examples per label) [Google_Drive](https://drive.google.com/file/d/11Si6nVjE5_E32kbb_qLuXS96fccuoO15/view?usp=sharing). You don't have to install CrossFit env if you download the data. 

move the downloaded data to
```bash
mkdir raw_data
mkdir raw_data/gym
```

How to build your own customized data: You need to install [crossfit](https://github.com/INK-USC/CrossFit) env:

CrossFit Environment

```bash
# Create a new conda environment (optional)
conda create -n crossfit python=3.6.9
conda activate crossfit
# For building the NLP Few-shot Gym
pip install datasets==1.4.0 py7zr wget
# For reproducing the baseline methods
pip install torch==1.1.0 higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1 rouge==1.0.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

Download the datasets

```bash
conda activate crossfit
cd scripts
bash zero_para_download.sh
```

## ConEntail Environment

Install the conda environment
```bash
conda create -n entail2 python=3.6.9
conda activate entail2
pip install -r requirements.txt
pip install -e .
```

This step is used to compose the individual datasets into a single dataset for supervised pretraining. You can skip this step if you download the preprocessed data. Be sure to use ```conda activate entail2``` before running the following command.
```bash
# generate the supervised pretraining dataset
python entail2/dataloader/gym2entail_multitask.py 
```

## Run

see ```scripts``` for more examples. 


Training

```
CUDA_VISIBLE_DEVICES=0 \
python entail2/runner/runner.py \
--learning_rate 1e-5 \
--warmup_ratio 0.06 \
--train_batch_size 32 \
--num_train_epochs 10 \
--bert_name bert \
--model_name entail2 \
--use_sampler \
--mode train;
```

Evaluation


For evaluation, you have to make sure you have downloaded individual datasets through crossfit or from huggingface datasets (and put the data in `raw_data/gym`). You don't have to download all the datasets. As long as you have a dataset of interest, you can modify the scripts below for a customized evaluation. 

e.g., zero-shot evaluation, see [example](./scripts/eval_models_on_tasks_0shot.sh) for complete scripts, and you can use for-loop to run multiple models on multiple test sets.
Few-shot evaluation: [here](./scripts/finetune_15_100_shot_bert.sh) and [here](./scripts/finetune_15_100_shot_bart.sh) 

First, you need to generate the test sets and zero-shot support sets (only label names)
```bash
    python scripts/gen_singletask_test.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK}
    python scripts/gen_singletask_zeroshot_support.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} --shots 1 --times 1
```

Then, you'll need to run the model on each task:
```bash
    python entail2/runner/runner.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} \
    --model ${MODEL} \
    --test_times 1 \
    --test_shots 1 \
    --mode test
```

Other baselines:
modify the ${MODEL} variable in scrips to
```
MODELS=(efl_no_cl entail2 crossfit unifew)
```


## Citation
```bibtex
@article{zhang2023conentail,
      title={ConEntail: An Entailment-based Framework for Universal Zero and Few Shot Classification with Supervised Contrastive Pretraining}, 
      author={Zhang, Ranran Haoran and Fan, Aysa Xuemo and Zhang, Rui},
      booktitle={EACL 2023},
      year={2022},
}
``` 
