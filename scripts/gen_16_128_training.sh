shots=( 16 32 48 64 80 96 112 128 )
# shots=( 16 32 )
 
for s in "${shots[@]}"
do
  python entail2/dataloader/gym2entail_multitask.py --training_shots $s
done

