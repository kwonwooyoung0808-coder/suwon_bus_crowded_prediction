from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "report_assets"
ASSET_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(DATA_DIR / "bus_crowded_raw.csv", encoding="utf-8-sig")
    route_df = pd.read_csv(DATA_DIR / "suwon_route_list.csv", encoding="utf-8-sig", dtype=str)[
        ["route_id", "source_route_name"]
    ].drop_duplicates()
    compare_df = pd.read_csv(DATA_DIR / "model_compare_results.csv", encoding="utf-8-sig")

    raw_df["route_id"] = raw_df["route_id"].astype(str)
    raw_df["collected_at"] = pd.to_datetime(raw_df["collected_at"])
    raw_df = raw_df.merge(route_df, on="route_id", how="left")
    raw_df["route_name_display"] = raw_df["source_route_name"].fillna(raw_df["route_name"])
    raw_df["hour"] = raw_df["collected_at"].dt.hour
    return raw_df, route_df, compare_df


def save_crowded_distribution(raw_df: pd.DataFrame) -> Path:
    out = ASSET_DIR / "01_crowded_distribution.png"
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=raw_df,
        x="crowded_label",
        order=["여유", "보통", "혼잡", "매우혼잡"],
        palette=["#2a9d8f", "#f4a261", "#e76f51", "#b00020"],
    )
    plt.title("혼잡도 분포")
    plt.xlabel("혼잡도")
    plt.ylabel("건수")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_hourly_mean(raw_df: pd.DataFrame) -> Path:
    out = ASSET_DIR / "02_hourly_mean_crowded.png"
    hourly_mean = raw_df.groupby("hour")["crowded_code"].mean()
    plt.figure(figsize=(10, 4))
    plt.plot(hourly_mean.index, hourly_mean.values, marker="o", linewidth=2, color="#1d4e89")
    plt.title("시간대별 평균 혼잡도")
    plt.xlabel("hour")
    plt.ylabel("mean crowded_code")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_route_heatmap(raw_df: pd.DataFrame) -> Path:
    out = ASSET_DIR / "03_route_hour_heatmap.png"
    major_routes = ["10", "66", "62-1", "720-2", "13-4", "11-1", "20"]
    route_hour = (
        raw_df[raw_df["route_name_display"].isin(major_routes)]
        .groupby(["route_name_display", "hour"])["crowded_code"]
        .mean()
        .reset_index()
    )
    pivot = route_hour.pivot(index="route_name_display", columns="hour", values="crowded_code")
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues")
    plt.title("주요 노선별 시간대 평균 혼잡도")
    plt.xlabel("hour")
    plt.ylabel("route")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_model_compare(compare_df: pd.DataFrame) -> Path:
    out = ASSET_DIR / "04_model_compare.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=compare_df, x="model", y="valid_acc", palette=["#3a86ff", "#ff006e"], ax=axes[0])
    axes[0].set_title("valid_acc")
    axes[0].set_xlabel("model")
    axes[0].set_ylabel("score")

    sns.barplot(data=compare_df, x="model", y="valid_loss", palette=["#3a86ff", "#ff006e"], ax=axes[1])
    axes[1].set_title("valid_loss")
    axes[1].set_xlabel("model")
    axes[1].set_ylabel("score")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_route_table(raw_df: pd.DataFrame) -> Path:
    out = ASSET_DIR / "05_route_summary.csv"
    summary = (
        raw_df.groupby("route_name_display")
        .agg(
            수집건수=("crowded_code", "size"),
            평균혼잡도=("crowded_code", "mean"),
            차량수=("vehicle_id", "nunique"),
        )
        .sort_values("수집건수", ascending=False)
        .reset_index()
    )
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    raw_df, route_df, compare_df = load_data()
    outputs = [
        save_crowded_distribution(raw_df),
        save_hourly_mean(raw_df),
        save_route_heatmap(raw_df),
        save_model_compare(compare_df),
        save_route_table(raw_df),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
