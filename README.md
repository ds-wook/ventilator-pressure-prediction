# ventilator-pressure-prediction

### Introduction
This repository is the code that [Google Brain - Ventilator Pressure Prediction](https://www.kaggle.com/c/ventilator-pressure-prediction) competition.

### Learning Process
[Learning Visualization](https://app.neptune.ai/ds-wook/ventilator-pressure/experiments?split=bth&dash=charts&viewId=standard-view)

### Model Architecture


### Benchmark
|Score|CV|Public LB|private LB|
|-----|--|------|-------|
|Bidirect-LSTM1|0.1740|0.1618|-
|Bidirect-LSTM2|0.1693|0.1577|-|
|Bidirect-LSTM3|0.1693|0.1530|-|
|Bidirect-LSTM4|0.1714|0.1652|-|
|Bidirect-LSTM5|0.16884|0.1575|-|
|Bidirect-LSTM6|0.16934|0.1577|-|
|Regression|0.1764|0.1658|-|
|LightAutoML-liner|0.1756|0.1610|-|
|Resnet-LSTM|0.1601|0.1405|-|
|DNN-LSTM|0.1563|0.1385|-|
|Stacking-LightGBM(15-GroupKfold)|**0.1521**|**0.1365**|-|
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
+ tensorflow 2.0


### Reference
+ [LightAutoML](https://www.kaggle.com/tsano430/lightautoml-bidirectional-lstm)
+ [Bidirect-LSTM](https://www.kaggle.com/tsano430/tensor-bidirect-lstm-n-splits-10)
+ [FineTune of Bidirect-LSTM](https://www.kaggle.com/tenffe/finetune-of-tensorflow-bidirectional-lstm)
+ [Ensemble Folds with MEDIAN](https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153)
+ [Ventilator Train classification](https://www.kaggle.com/takamichitoda/ventilator-train-classification)
+ [LightGBM select features](https://www.kaggle.com/alexxanderlarko/lgbm-sel-feat-1)
