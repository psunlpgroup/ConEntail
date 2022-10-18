
# ConEntail
Source code for [ConEntail: An Entailment-based Framework for Universal Zero and Few Shot Classification with Supervised Contrastive Pretraining](https://arxiv.org/pdf/2210.07587.pdf)

## Crossfit Data

You need to install [crossfit](https://github.com/INK-USC/CrossFit) env to download the datasets first.

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
# generate the supervised pretraining dataset
python entail2/dataloader/gym2entail_multitask.py 
```

## Environment

```bash
conda create -n entail2 python=3.6.9
conda activate entail2
pip install -r requirements.txt
pip install -e .
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



## Citation
```bibtex
@article{conentail,
      title={ConEntail: An Entailment-based Framework for Universal Zero and Few Shot Classification with Supervised Contrastive Pretraining}, 
      author={Zhang, Haoran and Fan, Aysa Xuemo and Zhang, Rui},
      journal={ArXiv},
      year={2022},
}
``` 