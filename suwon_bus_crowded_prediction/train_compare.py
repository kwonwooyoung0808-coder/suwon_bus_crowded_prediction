from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM
from tensorflow.keras.models import Sequential


DATA_FILE = Path("data/model_sequences.npz")
RESULT_FILE = Path("data/model_compare_results.csv")
TIME_STEPS = 6
NUM_CLASSES = 5
VOCAB_SIZE = 5
SEED = 42

tf.keras.utils.set_random_seed(SEED)


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError("먼저 preprocess.py를 실행해 model_sequences.npz를 만들어야 합니다.")

    bundle = np.load(DATA_FILE)
    x = bundle["X"]
    y = bundle["y"]

    y = y - 1
    return x, y


def train_valid_split(x: np.ndarray, y: np.ndarray, valid_ratio: float = 0.2):
    split_idx = max(1, int(len(x) * (1 - valid_ratio)))
    if split_idx >= len(x):
        split_idx = len(x) - 1

    x_train, x_valid = x[:split_idx], x[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]
    return x_train, x_valid, y_train, y_valid


def make_lstm_model() -> Sequential:
    model = Sequential(
        [
            Embedding(input_dim=VOCAB_SIZE, output_dim=8, input_length=TIME_STEPS),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_gru_model() -> Sequential:
    model = Sequential(
        [
            Embedding(input_dim=VOCAB_SIZE, output_dim=8, input_length=TIME_STEPS),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_training(model_name: str, model: Sequential, x_train, y_train, x_valid, y_valid):
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=30,
        batch_size=8,
        verbose=0,
        callbacks=callbacks,
    )

    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    valid_loss, valid_acc = model.evaluate(x_valid, y_valid, verbose=0)

    return {
        "model": model_name,
        "epochs_ran": len(history.history["loss"]),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
    }


def main():
    x, y = load_dataset()
    if len(np.unique(y)) < 2:
        raise ValueError(
            "현재 라벨 종류가 너무 적습니다. crowded가 여러 단계로 더 수집된 뒤 다시 학습해야 합니다."
        )

    x_train, x_valid, y_train, y_valid = train_valid_split(x, y)
    if len(x_valid) == 0:
        raise ValueError("검증 데이터가 없습니다. 더 많은 시퀀스를 수집해 주세요.")

    results = []
    results.append(run_training("LSTM", make_lstm_model(), x_train, y_train, x_valid, y_valid))
    results.append(run_training("GRU", make_gru_model(), x_train, y_train, x_valid, y_valid))

    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    print(result_df.to_string(index=False))
    print(f"비교 결과 저장 완료: {RESULT_FILE}")


if __name__ == "__main__":
    main()
