# HiPAL
Code for KDD'22 Applied Data Science Track submission -- HiPAL: A Deep Framework for Physician Burnout Prediction Using Activity Logs in Electronic Health Records

<img width="1491" alt="image" src="https://user-images.githubusercontent.com/12636809/153683643-512be547-c8aa-44f7-bf3c-687d58cffa5b.png">

## Setup
```
pip install requirements.txt
```

## To Run
Parse activities from EHR log files:
```
python log_parsing.py
```
Run HiPAL and its variants:
```
bash bash/run_hitcn_cv.sh
```
Run single-level models (FCN, CausalNet, ResTCN):
```
bash bash/run_single_level.sh
```
Run hierarchial RNNs (H-RNN, HiGRU):
```
bash bash/run_higru_cv.sh
```

## Interpretable Burnout Prediction


<img width="1248" alt="image" src="https://user-images.githubusercontent.com/12636809/161802902-446a600c-7cc8-4caf-8266-35805f2e4fac.png">




