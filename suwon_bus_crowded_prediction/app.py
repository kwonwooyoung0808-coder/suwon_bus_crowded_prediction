from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_FILE = DATA_DIR / "bus_crowded_raw.csv"
ENRICHED_FILE = DATA_DIR / "bus_crowded_enriched.csv"
COMPARE_FILE = DATA_DIR / "model_compare_results.csv"
PREVIEW_FILE = DATA_DIR / "model_sequences_preview.csv"
ROUTE_FILE = DATA_DIR / "suwon_route_list.csv"

LABEL_MAP = {
    1: "여유",
    2: "보통",
    3: "혼잡",
    4: "매우혼잡",
}

LABEL_COLOR = {
    "여유": "#2a9d8f",
    "보통": "#f4a261",
    "혼잡": "#e76f51",
    "매우혼잡": "#b00020",
}


st.set_page_config(page_title="통학 버스 혼잡도 예측", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef6ff 0%, #f8fbff 42%, #ffffff 100%);
    }
    .hero {
        padding: 24px 28px;
        border-radius: 24px;
        background: linear-gradient(135deg, #133c73 0%, #1e5aa8 55%, #6ec1ff 100%);
        color: white;
        box-shadow: 0 20px 50px rgba(19, 60, 115, 0.18);
        margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
    }
    .hero p {
        margin: 8px 0 0;
        font-size: 1rem;
        opacity: 0.95;
    }
    .section-card {
        background: white;
        padding: 20px 22px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(17, 44, 87, 0.08);
        border: 1px solid rgba(19, 60, 115, 0.06);
        margin-bottom: 16px;
    }
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(17, 44, 87, 0.08);
        border: 1px solid rgba(19, 60, 115, 0.06);
        text-align: center;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4f6785;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #163b69;
        margin-top: 6px;
    }
    .prediction-box {
        border-radius: 22px;
        padding: 20px 24px;
        color: white;
        font-weight: 700;
        margin-bottom: 14px;
    }
    .small-note {
        color: #58708e;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_route_map() -> pd.DataFrame:
    if not ROUTE_FILE.exists():
        return pd.DataFrame(columns=["route_id", "source_route_name"])
    df = pd.read_csv(ROUTE_FILE, encoding="utf-8-sig", dtype=str)
    return df[["route_id", "source_route_name"]].drop_duplicates()


@st.cache_data
def load_raw() -> pd.DataFrame:
    if not RAW_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(RAW_FILE, encoding="utf-8-sig")
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    return df


@st.cache_data
def load_enriched() -> pd.DataFrame:
    if not ENRICHED_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(ENRICHED_FILE, encoding="utf-8-sig")
    df["collected_at"] = pd.to_datetime(df["collected_at"])
    return df


@st.cache_data
def load_compare() -> pd.DataFrame:
    if not COMPARE_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARE_FILE, encoding="utf-8-sig")


@st.cache_data
def load_preview() -> pd.DataFrame:
    if not PREVIEW_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREVIEW_FILE, encoding="utf-8-sig")
    if "input_sequence" in df.columns:
        df["input_sequence"] = df["input_sequence"].apply(
            lambda x: tuple(literal_eval(x)) if isinstance(x, str) else tuple()
        )
    return df


def attach_route_name(df: pd.DataFrame, route_map: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "route_id" in out.columns:
        out["route_id"] = out["route_id"].astype(str)
    merged = out.merge(route_map, on="route_id", how="left")
    merged["route_display"] = merged["source_route_name"].fillna(merged.get("route_name", ""))
    merged["route_display"] = merged["route_display"].replace("", "노선미상")
    return merged


def badge_html(label: str, subtitle: str) -> str:
    color = LABEL_COLOR.get(label, "#4f6785")
    return f"""
    <div class="prediction-box" style="background:{color};">
        <div style="font-size:0.95rem; opacity:0.92;">{subtitle}</div>
        <div style="font-size:2rem; margin-top:6px;">{label}</div>
    </div>
    """


def neutral_badge_html(title: str, subtitle: str) -> str:
    return f"""
    <div class="prediction-box" style="background:#6c7f99;">
        <div style="font-size:0.95rem; opacity:0.92;">{subtitle}</div>
        <div style="font-size:1.5rem; margin-top:6px;">{title}</div>
    </div>
    """


def score_to_label(score: float) -> tuple[int, str]:
    clipped = max(1.0, min(4.0, float(score)))
    code = int(round(clipped))
    return code, LABEL_MAP.get(code, "정보없음")


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def predict_next_label(route_id: str, direction: str, recent_sequence: tuple[int, ...], preview_df: pd.DataFrame) -> tuple[int, str]:
    subset = preview_df[preview_df["route_id"].astype(str) == str(route_id)].copy()
    if direction != "전체":
        subset = subset[subset["commute_direction"] == direction]

    exact = subset[subset["input_sequence"] == recent_sequence]
    if not exact.empty:
        score = float(exact["target_label"].mean())
        pred, _ = score_to_label(score)
        return pred, f"최근 6개 흐름과 동일한 패턴 평균 {score:.2f}"

    if not subset.empty:
        score = float(subset["target_label"].mean())
        pred, _ = score_to_label(score)
        return pred, f"같은 노선·방향의 누적 평균 {score:.2f}"

    if not preview_df.empty:
        score = float(preview_df["target_label"].mean())
        pred, _ = score_to_label(score)
        return pred, f"전체 학습 데이터 평균 {score:.2f}"

    return 1, "기본값"


def predict_by_hour(route_df: pd.DataFrame, target_hour: int) -> tuple[int | None, str]:
    if route_df.empty:
        return None, "데이터 부족"

    temp = route_df.copy()
    temp["hour"] = temp["collected_at"].dt.hour
    same_hour = temp[temp["hour"] == target_hour]
    if not same_hour.empty:
        score = float(same_hour["crowded_code"].mean())
        pred, _ = score_to_label(score)
        return pred, f"{target_hour:02d}시 평균 혼잡도 {score:.2f}"

    return None, f"{target_hour:02d}시 데이터 없음"


route_map = load_route_map()
raw_df = attach_route_name(load_raw(), route_map)
enriched_df = attach_route_name(load_enriched(), route_map)
compare_df = load_compare()
preview_df = attach_route_name(load_preview(), route_map)

st.markdown(
    """
    <div class="hero">
        <h1>통학 버스 혼잡도 예측 대시보드</h1>
        <p>아주대입구·우리은행 / 아주대학교입구에서 성빈센트병원까지 가는 통학 노선을 기준으로 혼잡도 현황과 예측 결과를 보여줍니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if raw_df.empty:
    st.error("수집 데이터가 없습니다. 먼저 혼잡도 수집을 진행해주세요.")
    st.stop()

page = st.sidebar.radio(
    "화면 선택",
    ["통학 현황", "모델 비교", "통학 예측"],
)

last_collected = raw_df["collected_at"].max()
total_rows = len(raw_df)
total_routes = raw_df["route_id"].nunique()
total_vehicles = raw_df["vehicle_id"].nunique()


if page == "통학 현황":
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("누적 수집 건수", f"{total_rows:,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("수집 노선 수", f"{total_routes}"), unsafe_allow_html=True)
    c3.markdown(metric_card("관측 차량 수", f"{total_vehicles}"), unsafe_allow_html=True)
    c4.markdown(metric_card("마지막 수집 시각", last_collected.strftime("%H:%M")), unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("노선별 수집 현황")
    route_count = (
        raw_df.groupby("route_display")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .rename("수집건수")
    )
    st.bar_chart(route_count)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("시간대별 평균 혼잡도")
        temp = raw_df.copy()
        temp["hour"] = temp["collected_at"].dt.hour
        hourly = temp.groupby("hour")["crowded_code"].mean()
        st.line_chart(hourly)
        st.caption("1=여유, 2=보통, 3=혼잡, 4=매우혼잡")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("혼잡도 단계 분포")
        crowded_counts = raw_df["crowded_label"].value_counts().reindex(["여유", "보통", "혼잡", "매우혼잡"]).fillna(0)
        st.bar_chart(crowded_counts)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("최근 수집 데이터")
    show_cols = ["collected_at", "route_display", "plate_no", "station_seq", "crowded_label"]
    recent_df = raw_df.sort_values("collected_at", ascending=False)[show_cols].head(20)
    recent_df.columns = ["수집시각", "노선", "차량번호", "정류장순번", "혼잡도"]
    st.dataframe(recent_df, width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "모델 비교":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("LSTM vs GRU 비교 결과")
    if compare_df.empty:
        st.warning("모델 비교 결과 파일이 없습니다.")
    else:
        winner = compare_df.sort_values("valid_acc", ascending=False).iloc[0]
        col1, col2 = st.columns([1.2, 1.8])
        with col1:
            st.markdown(
                badge_html(
                    winner["model"],
                    f"검증 정확도 {winner['valid_acc']:.3f}로 가장 높은 모델",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='small-note'>현재 저장된 비교 결과 기준으로는 {winner['model']} 모델이 가장 안정적입니다.</div>",
                unsafe_allow_html=True,
            )
        with col2:
            view = compare_df.copy()
            for col in ["train_loss", "train_acc", "valid_loss", "valid_acc"]:
                view[col] = view[col].map(lambda x: round(float(x), 4))
            view.columns = ["모델", "학습 epoch", "학습 loss", "학습 acc", "검증 loss", "검증 acc"]
            st.dataframe(view, width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("학습 데이터 규모")
        st.metric("시퀀스 샘플 수", f"{len(preview_df):,}")
        st.metric("사용 노선 수", f"{preview_df['route_id'].astype(str).nunique() if not preview_df.empty else 0}")
        st.metric("차량 수", f"{preview_df['vehicle_id'].astype(str).nunique() if not preview_df.empty else 0}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("예측 라벨 분포")
        if not preview_df.empty:
            label_dist = preview_df["target_label"].map(LABEL_MAP).value_counts().reindex(["여유", "보통", "혼잡", "매우혼잡"]).fillna(0)
            st.bar_chart(label_dist)
        else:
            st.info("시퀀스 미리보기 파일이 없습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("모델 설명")
    st.write("LSTM과 GRU 모두 최근 6개 시점의 crowded 흐름을 입력으로 받아 다음 시점의 혼잡도 단계를 예측합니다.")
    st.write("현재 데이터에서는 GRU가 근소하게 더 높은 검증 정확도를 보였고, 두 모델 모두 85%대의 검증 정확도를 기록했습니다.")
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "통학 예측":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("다음 시점 혼잡도 예측")
    route_options = sorted(raw_df["route_display"].dropna().astype(str).unique().tolist())
    selected_route = st.selectbox("통학 노선 선택", route_options)

    route_ids = (
        raw_df.loc[raw_df["route_display"] == selected_route, "route_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    selected_route_id = route_ids[0]

    route_df = enriched_df[enriched_df["route_id"].astype(str) == str(selected_route_id)].copy()
    direction_options = ["전체"]
    if not route_df.empty and "commute_direction" in route_df.columns:
        direction_options += [d for d in route_df["commute_direction"].dropna().astype(str).unique().tolist() if d]
    selected_direction = st.radio("방향 선택", direction_options, horizontal=True)

    if selected_direction != "전체" and not route_df.empty:
        route_df = route_df[route_df["commute_direction"] == selected_direction]

    route_df = route_df.sort_values("collected_at")
    selected_hour = st.slider("예측 시간대 선택", min_value=0, max_value=23, value=18)

    if route_df.empty:
        st.warning("선택한 조건의 수집 데이터가 아직 충분하지 않습니다.")
    else:
        recent_seq = tuple(route_df["crowded_code"].tail(6).astype(int).tolist())
        current_label = LABEL_MAP.get(int(route_df["crowded_code"].iloc[-1]), "정보없음")
        pred_code, basis = predict_next_label(selected_route_id, selected_direction, recent_seq, preview_df)
        pred_label = LABEL_MAP.get(pred_code, "정보없음")
        hour_pred_code, hour_basis = predict_by_hour(route_df, selected_hour)
        hour_pred_label = LABEL_MAP.get(hour_pred_code, "정보없음") if hour_pred_code else None

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(badge_html(current_label, "현재 기준 혼잡도"), unsafe_allow_html=True)
        with col2:
            st.markdown(badge_html(pred_label, "다음 시점 예측 혼잡도"), unsafe_allow_html=True)
        with col3:
            if hour_pred_label:
                st.markdown(
                    badge_html(hour_pred_label, f"{selected_hour:02d}시 예상 혼잡도"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    neutral_badge_html("데이터 없음", f"{selected_hour:02d}시 예상 혼잡도"),
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"<div class='small-note'>다음 시점 예측 기준: {basis}<br>{selected_hour:02d}시 예측 기준: {hour_basis}</div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("최근 6개 혼잡 흐름")
            seq_labels = [LABEL_MAP.get(v, str(v)) for v in recent_seq]
            seq_df = pd.DataFrame({"순서": [f"T-{5-i}" for i in range(6)], "혼잡도": seq_labels})
            st.dataframe(seq_df, width="stretch", hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("시간대별 혼잡 패턴")
            trend = route_df[["collected_at", "crowded_code"]].tail(30).set_index("collected_at")
            hourly = route_df.assign(hour=route_df["collected_at"].dt.hour).groupby("hour")["crowded_code"].mean()
            st.bar_chart(hourly)
            st.caption("값이 높을수록 혼잡합니다. 퇴근시간대가 높게 보이는지 바로 확인할 수 있습니다.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("최근 관측 기록")
        table = route_df[["collected_at", "plate_no", "station_seq", "crowded_label"]].tail(15).sort_values("collected_at", ascending=False)
        table.columns = ["수집시각", "차량번호", "정류장순번", "혼잡도"]
        st.dataframe(table, width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
