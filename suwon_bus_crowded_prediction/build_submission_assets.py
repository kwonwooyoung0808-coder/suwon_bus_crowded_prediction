from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "ppt_assets"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")
tf.keras.utils.set_random_seed(42)

TIME_STEPS = 6
VOCAB_SIZE = 5
NUM_CLASSES = 5


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_text_canvas(path: Path, size: tuple[int, int], title: str, lines: list[str], subtitle: str | None = None) -> None:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(30, bold=True)
    body_font = get_font(22)
    sub_font = get_font(18)

    x, y = 50, 40
    draw.text((x, y), title, fill="#173b69", font=title_font)
    y += 60
    if subtitle:
        draw.text((x, y), subtitle, fill="#58708e", font=sub_font)
        y += 42

    for line in lines:
        wrapped = wrap(line, width=60)
        for part in wrapped:
            draw.text((x, y), f"- {part}", fill="#1f2937", font=body_font)
            y += 34
        y += 10

    img.save(path)


def load_raw() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "bus_crowded_raw.csv", encoding="utf-8-sig")
    routes = pd.read_csv(DATA_DIR / "suwon_route_list.csv", encoding="utf-8-sig", dtype=str)[["route_id", "source_route_name"]].drop_duplicates()
    raw["route_id"] = raw["route_id"].astype(str)
    raw["collected_at"] = pd.to_datetime(raw["collected_at"])
    raw = raw.merge(routes, on="route_id", how="left")
    raw["route_name_display"] = raw["source_route_name"].fillna(raw["route_name"])
    raw["hour"] = raw["collected_at"].dt.hour
    return raw


def load_dataset():
    bundle = np.load(DATA_DIR / "model_sequences.npz")
    x = bundle["X"]
    y = bundle["y"] - 1
    split_idx = max(1, int(len(x) * 0.8))
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
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_models():
    x_train, x_valid, y_train, y_valid = load_dataset()
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

    lstm = make_lstm_model()
    lstm_hist = lstm.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=20, batch_size=8, verbose=0, callbacks=callbacks)

    gru = make_gru_model()
    gru_hist = gru.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=20, batch_size=8, verbose=0, callbacks=callbacks)

    gru_pred = gru.predict(x_valid, verbose=0).argmax(axis=1)
    return (lstm_hist.history, gru_hist.history, y_valid, gru_pred)


def save_table_image(df: pd.DataFrame, path: Path, title: str, size: tuple[int, int]) -> None:
    fig, ax = plt.subplots(figsize=(size[0] / 180, size[1] / 180), dpi=180)
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=180)
    plt.close()


def make_assets() -> None:
    raw = load_raw()
    compare = pd.read_csv(DATA_DIR / "model_compare_results.csv", encoding="utf-8-sig")

    # slide 7
    table_df = raw[["collected_at", "route_name_display", "vehicle_id", "station_seq", "crowded_label"]].head(12).copy()
    table_df.columns = ["수집시각", "노선", "차량ID", "정류장순번", "혼잡도"]
    save_table_image(table_df, OUT_DIR / "image14.png", "수집 원본 데이터 예시", (1863, 584))

    # slide 8
    fig, axes = plt.subplots(2, 1, figsize=(1299 / 180, 1375 / 180), dpi=180)
    crowded_counts = raw["crowded_label"].value_counts().reindex(["여유", "보통", "혼잡", "매우혼잡"]).fillna(0)
    sns.barplot(x=crowded_counts.index, y=crowded_counts.values, palette=["#2a9d8f", "#f4a261", "#e76f51", "#b00020"], ax=axes[0])
    axes[0].set_title("혼잡도 분포")
    axes[0].set_xlabel("혼잡도")
    axes[0].set_ylabel("건수")
    hourly_mean = raw.groupby("hour")["crowded_code"].mean()
    axes[1].plot(hourly_mean.index, hourly_mean.values, marker="o", linewidth=2, color="#1d4e89")
    axes[1].set_title("시간대별 평균 혼잡도")
    axes[1].set_xlabel("hour")
    axes[1].set_ylabel("mean crowded_code")
    axes[1].set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "image15.png", dpi=180)
    plt.close()

    # slide 9
    dup_count = int(raw.duplicated().sum())
    miss = raw.isna().sum().rename("결측개수").reset_index()
    miss.columns = ["컬럼", "결측개수"]
    miss["중복행수"] = ""
    miss.loc[0, "중복행수"] = dup_count
    save_table_image(miss, OUT_DIR / "image16.png", "결측치 및 중복값 통계", (1131, 405))

    # slide 10
    fig, ax = plt.subplots(figsize=(1098 / 180, 347 / 180), dpi=180)
    sns.histplot(raw["hour"], bins=12, kde=False, color="#3a86ff", ax=ax)
    ax.set_title("Validation sample actual/pred comparison")
    ax.set_xlabel("sample")
    ax.set_ylabel("crowded_code")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "image28.png", dpi=180)
    plt.close()

    # slide 17
    save_text_canvas(
        OUT_DIR / "image29.png",
        (479, 477),
        "Streamlit",
        [
            "통학 현황",
            "모델 비교",
            "통학 예측",
        ],
        subtitle="대시보드 메뉴",
    )

    # slide 18
    save_text_canvas(
        OUT_DIR / "image30.png",
        (834, 611),
        "모델 비교 화면",
        [
            "LSTM vs GRU 검증 정확도",
            "학습 데이터 규모",
            "예측 라벨 분포",
        ],
        subtitle="앱 화면 구성",
    )
    save_text_canvas(
        OUT_DIR / "image31.png",
        (810, 413),
        "통학 예측 화면",
        [
            "노선 선택",
            "방향 선택",
            "시간대 선택",
            "현재/다음/시간대 예측 혼잡도",
        ],
        subtitle="앱 화면 구성",
    )


if __name__ == "__main__":
    make_assets()
