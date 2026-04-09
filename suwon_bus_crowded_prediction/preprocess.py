from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILE = Path("data/bus_crowded_raw.csv")
OUTPUT_FILE = Path("data/model_sequences.npz")
META_FILE = Path("data/model_sequences_preview.csv")
ENRICHED_FILE = Path("data/bus_crowded_enriched.csv")
TIME_STEPS = 6


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILE, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("raw 데이터가 비어 있습니다. collector.py crowded를 더 실행해 주세요.")

    df["collected_at"] = pd.to_datetime(df["collected_at"])
    df["route_id"] = df["route_id"].astype(str)
    df["vehicle_id"] = df["vehicle_id"].astype(str)
    df["route_name"] = df["route_name"].astype(str).replace("nan", "")
    df["crowded_code"] = pd.to_numeric(df["crowded_code"], errors="coerce")
    df["station_seq"] = pd.to_numeric(df["station_seq"], errors="coerce")
    df = df.dropna(subset=["crowded_code"]).copy()
    df["crowded_code"] = df["crowded_code"].astype(int)
    return df


def infer_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["route_id", "vehicle_id", "collected_at"]).copy()
    grouped = df.groupby(["route_id", "vehicle_id"], sort=False)

    df["station_seq_diff"] = grouped["station_seq"].diff()
    df["direction_code"] = np.where(
        df["station_seq_diff"] > 0,
        1,
        np.where(df["station_seq_diff"] < 0, -1, 0),
    )
    df["direction_label"] = df["direction_code"].map(
        {
            1: "정방향",
            -1: "역방향",
            0: "방향유지",
        }
    )

    # 통학 프로젝트용 이름. 실제 등교/귀가 매핑은 정류장 기준 추가 검증 전까지 추정값으로 사용.
    df["commute_direction"] = df["direction_code"].map(
        {
            1: "등교추정",
            -1: "귀가추정",
            0: "판별보류",
        }
    )
    return df


def build_sequences(df: pd.DataFrame, time_steps: int = TIME_STEPS):
    samples = []
    labels = []
    meta_rows = []

    grouped = df.sort_values("collected_at").groupby(["route_id", "vehicle_id"], sort=False)
    for (route_id, vehicle_id), group in grouped:
        values = group["crowded_code"].tolist()
        timestamps = group["collected_at"].tolist()
        route_names = group["route_name"].tolist()
        direction_labels = group["direction_label"].tolist()
        commute_directions = group["commute_direction"].tolist()

        if len(values) <= time_steps:
            continue

        for idx in range(len(values) - time_steps):
            x = values[idx : idx + time_steps]
            y = values[idx + time_steps]
            target_time = timestamps[idx + time_steps]

            samples.append(x)
            labels.append(y)
            meta_rows.append(
                {
                    "route_id": route_id,
                    "route_name": route_names[idx + time_steps],
                    "vehicle_id": vehicle_id,
                    "target_time": target_time,
                    "direction_label": direction_labels[idx + time_steps],
                    "commute_direction": commute_directions[idx + time_steps],
                    "input_sequence": x,
                    "target_label": y,
                }
            )

    if not samples:
        raise ValueError(
            "시퀀스를 만들 수 없습니다. 같은 차량/노선에 대해 시간차를 두고 더 많이 수집해야 합니다."
        )

    x = np.array(samples, dtype=np.int32)
    y = np.array(labels, dtype=np.int32)
    return x, y, pd.DataFrame(meta_rows)


def main():
    df = load_raw()
    df = infer_direction(df)
    df.to_csv(ENRICHED_FILE, index=False, encoding="utf-8-sig")
    x, y, meta_df = build_sequences(df)

    np.savez(OUTPUT_FILE, X=x, y=y)
    meta_df.to_csv(META_FILE, index=False, encoding="utf-8-sig")

    print(f"방향 포함 raw 저장 완료: {ENRICHED_FILE}")
    print(f"시퀀스 저장 완료: {OUTPUT_FILE}")
    print(f"미리보기 저장 완료: {META_FILE}")
    print(f"샘플 수: {len(x)}")
    print(f"입력 shape: {x.shape}")
    print(f"라벨 분포: {dict(pd.Series(y).value_counts().sort_index())}")


if __name__ == "__main__":
    main()
