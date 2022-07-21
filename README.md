# HiPAL
Code for KDD'22 Paper -- HiPAL: A Deep Framework for Physician Burnout Prediction Using Activity Logs in Electronic Health Records

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


<img width="1394" alt="image" src="https://user-images.githubusercontent.com/12636809/161803105-4b09591c-6cde-4428-b38e-553c95997e9b.png">


## Citation

Please consider citing our work if you find this repository useful!

```
@article{liu2022hipal,
  title={HiPAL: A Deep Framework for Physician Burnout Prediction Using Activity Logs in Electronic Health Records},
  author={Liu, Hanyang and Lou, Sunny S and Warner, Benjamin C and Harford, Derek R and Kannampallil, Thomas and Lu, Chenyang},
  journal={ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2022}
}
```





