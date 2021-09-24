# ventilator-pressure-prediction
Simulate a ventilator connected to a sedated patient's lung


### Benchmark
|Score|CV|Public LB|private LB|
|-----|--|------|-------|
|LightGBM(5-group kfold)|-|-|-|

### Requirements
+ numpy
+ pandas
+ scikit-learn
+ lightgbm
+ xgboost
+ optuna
+ neptune
+ hydra
+ pytorch-tabnet
+ pytorch
