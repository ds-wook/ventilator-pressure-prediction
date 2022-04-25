import gc
from typing import List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def get_model(train: pd.DataFrame) -> tf.keras.Model:
    inputs = keras.layers.Input(shape=train.shape[-2:])
    #     x = keras.layers.Bidirectional(keras.layers.LSTM(2048, return_sequences=True))(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True))(
        inputs
    )
    x1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))(x)
    x2 = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(x1)

    #     z2 = keras.layers.Bidirectional(keras.layers.GRU(units=256, return_sequences=True))(x2)
    #     z3 = keras.layers.Bidirectional(keras.layers.GRU(units=128, return_sequences=True))(keras.layers.Add()([x2, z2]))
    x3 = tf.keras.layers.Concatenate(axis=2)([x1, x2])
    x4 = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x3)

    x5 = keras.layers.Dense(100, activation="selu")(x4)
    x6 = keras.layers.Dense(100, activation="selu")(x5)
    x7 = keras.layers.Dense(100, activation="selu")(x6)
    x7 = tf.keras.layers.Concatenate(axis=2)([x7, x5])
    outputs = keras.layers.Dense(1)(x7)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_model")
    return model


def run_kfold_lstm(
    n_fold: int,
    train: pd.DataFrame,
    targets: pd.DataFrame,
    test: pd.DataFrame,
    epochs: int,
    batch_size: int,
    verbose: Union[int, bool] = False,
) -> List[np.ndarray]:
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=2021)
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        U_OUT_IDX = 3
        y_weight = np.ones_like(y_train)
        u_out_values = X_train[:, :, U_OUT_IDX]
        y_weight[u_out_values == 1] = 0.1
        model = get_model()
        model.compile(optimizer="adam", loss="mae", sample_weight_mode="temporal")

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        es = EarlyStopping(
            monitor="val_loss",
            patience=20,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )

        checkpoint_filepath = f"folds{fold}.hdf5"
        sv = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            options=None,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=y_weight.reshape((-1, 80, 1)),
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr, es, sv],
        )
        test_preds.append(model.predict(test).squeeze().reshape(-1, 1).squeeze())

        del X_train, X_valid, y_train, y_valid, model
        gc.collect()

    return test_preds
