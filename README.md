# ventilator-pressure-prediction
Simulate a ventilator connected to a sedated patient's lung


### Benchmark
|Score|CV|Public LB|private LB|
|-----|--|------|-------|
|LightGBM(5-GroupKfold)|||-|
|LSTM(5-GroupKfold)|0.4945|0.455|-|


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
