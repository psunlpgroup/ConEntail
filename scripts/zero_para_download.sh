# 16 32 48 64 80 96 112 128
cd zero_para_tasks
python _build_gym.py --build --n_proc=10
cd ..
# rm -rf ../raw_data/gym
mv data/* ../raw_data/gym