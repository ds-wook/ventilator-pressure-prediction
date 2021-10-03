# ventilator-pressure-prediction
Simulate a ventilator connected to a sedated patient's lung


### Benchmark
|Score|CV|Public LB|private LB|
|-----|--|------|-------|
|Stacking-LightGBM(5-GroupKfold)|0.0996|0.158|-|
|LSTM(5-Kfold)|0.418|0.369|-|


### Requirements
+ numpy
+ pandas
+ scikit-learn
+ lightgbm
+ xgboost
+ optuna
+ neptune.ai
+ hydra
+ pytorch-tabnet
+ pytorch
