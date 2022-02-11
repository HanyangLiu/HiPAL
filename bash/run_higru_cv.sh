#!/usr/bin/env bash
PYTHONPATH="/storage1/lu/Active/ehrlog/burnout:$PYTHONPATH"
export PYTHONPATH

memo=CV_hrnn
if_pretrain=False
if_train_predictor=True
if_deepsup=False
if_taildrop=False
epochs_lower=60
epochs_higher=50

train_model(){
  nohup python higru.py --memo $memo \
                         --max_n_shift 30 \
                         --max_shift_len 3008 \
                         --dim_embedding 100 \
                         --epochs_lower $epochs_lower \
                         --epochs_higher $epochs_higher \
                         --bs_lower 16 \
                         --bs_higher 4 \
                         --workers 5 \
                         --tcn_layers 6 \
                         --time_size 50 \
                         --alpha 0.1 \
                         --lr 0.001 \
                         --dropout 0.3 \
                         --gpu_id 0 \
                         --if_pretrain $if_pretrain \
                         --pretrain_model lower_pretrained \
                         --if_train_predictor $if_train_predictor \
                         --if_neptune True \
                         --if_earlystopping True \
                         --if_mem_constr True \
                         --if_deepsup $if_deepsup \
                         --if_taildrop $if_taildrop \
                         --trial_idx $1 \
                         --cv_idx $2 \
                         >> out/h-rnn.out
}

for (( trail_idx = 4; trail_idx <= 5; trail_idx++ ))
do
  for cv_idx in 0 1 2 3 4
  do
    echo "Start trail=$trail_idx, cv fold=$cv_idx ..."
    train_model ${trail_idx} ${cv_idx} &&
    echo "Finished!"
  done
done &


