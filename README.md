# ventilator-pressure-prediction

### Introduction
This repository is the code that placed 171st [Google Brain - Ventilator Pressure Prediction](https://www.kaggle.com/c/ventilator-pressure-prediction) competition.

### Learning Process
[Learning Visualization](https://app.neptune.ai/ds-wook/ventilator-pressure/experiments?split=bth&dash=charts&viewId=standard-view)

### Model Architecture
![competition-model](https://user-images.githubusercontent.com/46340424/140250859-3b96c624-27a1-40d1-8065-3327f5cf7f48.png)

### Benchmark
|Score|CV|Public LB|Private LB|
|-----|--|------|-------|
|LightAutoML-liner|0.1756|0.1641|0.1671|
|Resnet-LSTM|0.1601|0.1405|0.1416|
|DNN-LSTM|0.1563|0.1385|0.1414|
|Stacking-LightGBM(15-GroupKfold)|**0.1521**|**0.1365**|**0.1386**|
|LSTM(10-Kfold)|0.1620|0.1456|0.1475|


### Requirements
+ numpy
+ pandas
+ scikit-learn
+ lightgbm
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
