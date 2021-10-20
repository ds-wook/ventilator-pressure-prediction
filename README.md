# ventilator-pressure-prediction
Simulate a ventilator connected to a sedated patient's lung


### Benchmark
|Score|CV|Public LB|private LB|
|-----|--|------|-------|
|Stacking-LightGBM(10-GroupKfold)|0.1609|0.1455|-|
|LSTM(7-Kfold)|0.1620|0.1440|-|


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
+ tensorflow 2.0


### Reference
+ [LightAutoML](https://www.kaggle.com/tsano430/lightautoml-bidirectional-lstm)
+ [Bidirect-LSTM](https://www.kaggle.com/tsano430/tensor-bidirect-lstm-n-splits-10)
+ [FineTune of Bidirect-LSTM](https://www.kaggle.com/tenffe/finetune-of-tensorflow-bidirectional-lstm)
+ [Ensemble Folds with MEDIAN](https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153)
+ [Ventilator Train classification](https://www.kaggle.com/takamichitoda/ventilator-train-classification)
+ [LightGBM select features](https://www.kaggle.com/alexxanderlarko/lgbm-sel-feat-1)