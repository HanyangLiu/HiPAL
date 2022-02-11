#!/usr/bin/env bash
PYTHONPATH="/storage1/lu/Active/ehrlog/burnout:$PYTHONPATH"
export PYTHONPATH

memo=CV_hitcn_wo_dilation
if_pretrain=False
if_train_predictor=False
if_deepsup=True
if_taildrop=True
low_encoder=tcn
epochs_lower=60
epochs_higher=100

train_model(){
  nohup python hitcn.py --memo $memo \
                         --max_n_shift 30 \
                         --max_shift_len 3008 \
                         --dim_embedding 100 \
                         --epochs_lower $epochs_lower \
                         --epochs_higher $epochs_higher \
                         --bs_lower 16 \
                         --bs_higher 2 \
                         --workers 5 \
                         --tcn_layers 6 \
                         --time_size 50 \
                         --alpha 0.1 \
                         --lr 0.001 \
                         --dropout 0.3 \
                         --gpu_id 0 \
                         --if_pretrain $if_pretrain \
                         --pretrain_model lower_pretrained_wo_exp \
                         --if_train_predictor $if_train_predictor \
                         --if_neptune True \
                         --if_earlystopping True \
                         --if_mem_constr True \
                         --if_deepsup $if_deepsup \
                         --if_taildrop $if_taildrop \
                         --low_encoder $low_encoder \
                         --trial_idx $1 \
                         --cv_idx $2 \
                         --horizon $3 \
                         >> out/hitcn_horizon.out
}
for (( horizon = 1; horizon <= 10; horizon++ ))
do
  for (( trail_idx = 0; trail_idx <= 5; trail_idx++ ))
  do
    for cv_idx in 0 1 2 3 4
    do
      echo "Start trail=$trail_idx, cv fold=$cv_idx ..."
      train_model ${trail_idx} ${cv_idx} ${horizon} &&
      echo "Finished!"
    done
  done
done &


