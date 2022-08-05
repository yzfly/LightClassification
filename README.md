# LightGBM Binary Classification

A simple example of Tabular data Binary Classification using [LightGBM](https://github.com/microsoft/LightGBM). 🐶

## Requirements
```bash
pip install pandas numpy scipy scikit-learn lightgbm
```

## Usage

```Bash
python main.py --thres=0.2 --data_path=datasets/bank-additional-full.csv
```

🌝 output is something like this, you can optimizing the params to get better results.
```
[LightGBM] [Info] Total Bins 675
[LightGBM] [Info] Number of data points in the train set: 32950, number of used features: 10
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.113475 -> initscore=-2.055727
[LightGBM] [Info] Start training from score -2.055727
Training until validation scores don't improve for 10 rounds
Early stopping, best iteration is:
[51]    valid_0's auc: 0.947001
Saving LightGBM model...
Start predicting...
Acc: 0.908, F1: 0.645, Precision: 0.556, Recall: 0.768
Confusion Matrix: 
 [[6785  552]
 [ 209  692]]
```

## Acknowledgments

* [LightGBM](https://github.com/microsoft/LightGBM)
* [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

