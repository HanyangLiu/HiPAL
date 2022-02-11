#!/usr/bin/env bash
PYTHONPATH="/storage1/lu/Active/ehrlog/burnout:$PYTHONPATH"
export PYTHONPATH

memo=CV_single_FCN_6lyr
model_name=fcn
if_pretrain=False
if_train_predictor=True
if_deepsup=False

train_model(){
  nohup python single_level_models.py --memo $memo \
                         --max_n_actions 50000 \
                         --dim_embedding 100 \
                         --epochs 50 \
                         --batch_size 1 \
                         --workers 5 \
                         --gpu_id 0 \
                         --if_pretrain $if_pretrain \
                         --if_train_predictor $if_train_predictor \
                         --if_neptune True \
                         --if_earlystopping True \
                         --if_mem_constr True \
                         --if_deepsup $if_deepsup \
                         --trial_idx $1 \
                         --cv_idx $2 \
                         --model_name $model_name \
                         >> out/single_1.out
}

for (( trail_idx = 0; trail_idx <= 5; trail_idx++ ))
do
  for cv_idx in 0 1 2 3 4
  do
    echo "Start trail=$trail_idx, cv fold=$cv_idx ..."
    train_model ${trail_idx} ${cv_idx} &&
    echo "Finished!"
  done
done &
