from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import math
import os
import time
import sys
import xml.etree.ElementTree as ET

import pandas as pd
import requests


BASE_ROUTE_URL = "https://apis.data.go.kr/6410000/busrouteservice/v2"
BASE_LOCATION_URL = "https://apis.data.go.kr/6410000/buslocationservice/v2"
ROUTE_METHOD = "getBusRouteListv2"
LOCATION_METHOD = "getBusLocationListv2"

DATASET_DIR = Path("dataset")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ENV_FILE = Path(".env")
LOCAL_ROUTE_SOURCE = next(DATASET_DIR.glob("*.csv"))
ROUTE_LIST_FILE = DATA_DIR / "suwon_route_list.csv"
CROWDED_RAW_FILE = DATA_DIR / "bus_crowded_raw.csv"
COMMUTE_ROUTE_NAMES = [
    "15-1",
    "3",
    "720-2",
    "66",
    "10",
    "82-1",
    "62-1",
    "13-4",
    "20",
    "66-4",
    "65",
    "10-5",
    "83-1",
    "88",
    "11-1",
    "20-1",
    "37",
    "10-2",
    "27",
    "32-4",
    "999",
]
ROUND_ROBIN_STATE_FILE = DATA_DIR / "collect_state.csv"
DEFAULT_BATCH_SIZE = 5
REQUEST_DELAY_SECONDS = 1.2


def load_env() -> dict[str, str]:
    env = {}
    if not ENV_FILE.exists():
        return env

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def get_api_keys() -> tuple[str, str]:
    env = load_env()
    route_key = env.get("GBIS_ROUTE_API_KEY") or os.getenv("GBIS_ROUTE_API_KEY", "")
    location_key = env.get("GBIS_LOCATION_API_KEY") or os.getenv("GBIS_LOCATION_API_KEY", "")
    return route_key, location_key


def ensure_route_csv() -> None:
    if ROUTE_LIST_FILE.exists():
        return

    with ROUTE_LIST_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_route_name",
                "route_id",
                "route_name",
                "route_type_cd",
                "company_name",
                "district_cd",
                "up_first_station_name",
                "up_last_station_name",
            ]
        )


def ensure_raw_csv() -> None:
    if CROWDED_RAW_FILE.exists():
        return

    with CROWDED_RAW_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "collected_at",
                "route_id",
                "route_name",
                "vehicle_id",
                "plate_no",
                "station_seq",
                "station_id",
                "station_name",
                "crowded_code",
                "crowded_label",
            ]
        )


def crowded_label(code: int) -> str:
    mapping = {
        1: "여유",
        2: "보통",
        3: "혼잡",
        4: "매우혼잡",
    }
    return mapping.get(code, "정보없음")


def build_url(base: str, method: str) -> str:
    return f"{base.rstrip('/')}/{method}"


def create_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def request_api(url: str, params: dict) -> list[dict]:
    session = create_session()
    response = session.get(url, params=params, timeout=20)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "json" in content_type or response.text.strip().startswith("{"):
        return extract_items_from_json(response.json())
    return extract_items_from_xml(response.text)


def extract_items_from_json(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if "routeId" in payload or "routeName" in payload or "crowded" in payload:
        return [payload]

    queue = [payload]
    found_lists = []
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            for value in current.values():
                if isinstance(value, list):
                    found_lists.append(value)
                elif isinstance(value, dict):
                    if "routeId" in value or "routeName" in value or "crowded" in value:
                        found_lists.append([value])
                    queue.append(value)

    candidates = []
    for items in found_lists:
        dict_items = [item for item in items if isinstance(item, dict)]
        if dict_items:
            candidates.append(dict_items)

    if not candidates:
        return []
    return max(candidates, key=len)


def extract_items_from_xml(text: str) -> list[dict]:
    root = ET.fromstring(text)
    items = []
    for item in root.findall(".//busRouteList"):
        items.append({child.tag: (child.text or "").strip() for child in item})
    for item in root.findall(".//busLocationList"):
        items.append({child.tag: (child.text or "").strip() for child in item})
    for item in root.findall(".//item"):
        items.append({child.tag: (child.text or "").strip() for child in item})
    return items


def normalize_route_item(source_route_name: str, item: dict) -> dict:
    return {
        "source_route_name": source_route_name,
        "route_id": item.get("routeId") or item.get("route_id") or item.get("노선아이디", ""),
        "route_name": item.get("routeName") or item.get("route_name") or item.get("노선명", ""),
        "route_type_cd": item.get("routeTypeCd") or item.get("route_type_cd", ""),
        "company_name": item.get("companyName") or item.get("company_name", ""),
        "district_cd": item.get("districtCd") or item.get("district_cd", ""),
        "up_first_station_name": item.get("upFirstStationName") or item.get("up_first_station_name", ""),
        "up_last_station_name": item.get("upLastStationName") or item.get("up_last_station_name", ""),
    }


def normalize_location_item(route_id: str, item: dict) -> dict:
    return {
        "route_id": route_id,
        "route_name": item.get("routeName") or item.get("route_name", ""),
        "vehicle_id": item.get("vehId") or item.get("vehicle_id", ""),
        "plate_no": item.get("plateNo") or item.get("plate_no", ""),
        "station_seq": item.get("stationSeq") or item.get("station_seq", ""),
        "station_id": item.get("stationId") or item.get("station_id", ""),
        "station_name": item.get("stationName") or item.get("station_name", ""),
        "crowded_code": item.get("crowded") or item.get("crowded_code", ""),
    }


def fetch_route_list(keyword: str, route_api_key: str) -> list[dict]:
    params = {
        "serviceKey": route_api_key,
        "keyword": keyword,
    }
    return request_api(build_url(BASE_ROUTE_URL, ROUTE_METHOD), params)


def fetch_bus_locations(route_id: str, location_api_key: str) -> list[dict]:
    params = {
        "serviceKey": location_api_key,
        "routeId": route_id,
    }
    return request_api(build_url(BASE_LOCATION_URL, LOCATION_METHOD), params)


def save_route_rows(rows: list[dict], replace: bool = False) -> None:
    ensure_route_csv()
    new_df = pd.DataFrame(rows)
    if not replace and ROUTE_LIST_FILE.exists():
        existing = pd.read_csv(ROUTE_LIST_FILE, encoding="utf-8-sig")
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df = new_df.drop_duplicates(subset=["source_route_name", "route_id"])
    new_df.to_csv(ROUTE_LIST_FILE, index=False, encoding="utf-8-sig")


def append_location_rows(rows: list[dict]) -> None:
    ensure_raw_csv()
    with CROWDED_RAW_FILE.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for row in rows:
            code = int(str(row.get("crowded_code", "0") or "0") or 0)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    row.get("route_id", ""),
                    row.get("route_name", ""),
                    row.get("vehicle_id", ""),
                    row.get("plate_no", ""),
                    row.get("station_seq", ""),
                    row.get("station_id", ""),
                    row.get("station_name", ""),
                    code,
                    crowded_label(code),
                ]
            )


def load_round_robin_index(total_routes: int) -> int:
    if total_routes <= 0 or not ROUND_ROBIN_STATE_FILE.exists():
        return 0
    try:
        df = pd.read_csv(ROUND_ROBIN_STATE_FILE, encoding="utf-8-sig")
        if df.empty:
            return 0
        current = int(df.iloc[0]["next_index"])
        return current % total_routes
    except Exception:
        return 0


def save_round_robin_index(next_index: int) -> None:
    with ROUND_ROBIN_STATE_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["next_index"])
        writer.writerow([next_index])


def select_route_batch(route_df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    if route_df.empty:
        return route_df
    total_routes = len(route_df)
    if batch_size <= 0 or batch_size >= total_routes:
        save_round_robin_index(0)
        return route_df

    start_idx = load_round_robin_index(total_routes)
    end_idx = start_idx + batch_size
    if end_idx <= total_routes:
        batch = route_df.iloc[start_idx:end_idx]
    else:
        batch = pd.concat(
            [
                route_df.iloc[start_idx:total_routes],
                route_df.iloc[0 : end_idx - total_routes],
            ],
            ignore_index=True,
        )

    save_round_robin_index((start_idx + batch_size) % total_routes)
    return batch


def load_local_route_names(limit: int = 10) -> list[str]:
    df = pd.read_csv(LOCAL_ROUTE_SOURCE, encoding="cp949")
    route_names = (
        df["노선명"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return route_names[:limit]


def load_commute_route_names() -> list[str]:
    return COMMUTE_ROUTE_NAMES.copy()


def collect_route_ids(limit: int = 10) -> None:
    route_key, _ = get_api_keys()
    if not route_key:
        raise RuntimeError("GBIS_ROUTE_API_KEY가 없습니다.")

    source_route_names = load_local_route_names(limit=limit)
    all_rows = []
    for route_name in source_route_names:
        try:
            items = fetch_route_list(route_name, route_key)
        except Exception as exc:
            print(f"[route] {route_name} 조회 실패: {exc}")
            continue

        matched = []
        for item in items:
            normalized = normalize_route_item(route_name, item)
            if normalized["route_name"] == route_name and normalized["route_id"]:
                matched.append(normalized)

        if not matched and items:
            fallback = normalize_route_item(route_name, items[0])
            if fallback["route_id"]:
                matched.append(fallback)

        print(f"[route] {route_name}: {len(matched)}건 저장")
        all_rows.extend(matched)

    save_route_rows(all_rows)
    print(f"노선 목록 저장 완료: {ROUTE_LIST_FILE}")


def collect_commute_route_ids() -> None:
    route_key, _ = get_api_keys()
    if not route_key:
        raise RuntimeError("GBIS_ROUTE_API_KEY가 없습니다.")

    all_rows = []
    for route_name in load_commute_route_names():
        try:
            items = fetch_route_list(route_name, route_key)
        except Exception as exc:
            print(f"[commute] {route_name} 조회 실패: {exc}")
            continue

        matched = []
        for item in items:
            normalized = normalize_route_item(route_name, item)
            admin_name = str(item.get("adminName", "")).strip()
            region_name = str(item.get("regionName", "")).strip()
            normalized_route_name = str(normalized["route_name"]).strip()
            if (
                normalized_route_name == route_name
                and normalized["route_id"]
                and ("수원" in admin_name or "수원" in region_name)
            ):
                matched.append(normalized)

        print(f"[commute] {route_name}: {len(matched)}건 저장")
        all_rows.extend(matched)

    save_route_rows(all_rows, replace=True)
    print(f"통학 노선 목록 저장 완료: {ROUTE_LIST_FILE}")


def collect_crowded_once(limit: int | None = None) -> None:
    _, location_key = get_api_keys()
    if not location_key:
        raise RuntimeError("GBIS_LOCATION_API_KEY가 없습니다.")
    if not ROUTE_LIST_FILE.exists():
        raise RuntimeError("먼저 route 수집을 해야 합니다. python collector.py routes 를 먼저 실행하세요.")

    route_df = pd.read_csv(ROUTE_LIST_FILE, encoding="utf-8-sig", dtype=str)
    if limit is not None:
        route_df = route_df.head(limit)
    else:
        route_df = select_route_batch(route_df, batch_size=DEFAULT_BATCH_SIZE)

    all_rows = []
    saw_rate_limit = False
    for _, row in route_df.iterrows():
        route_id = str(row["route_id"]).replace(".0", "").strip()
        source_route_name = str(row.get("source_route_name", "")).strip()
        try:
            items = fetch_bus_locations(route_id, location_key)
        except Exception as exc:
            if "429" in str(exc):
                saw_rate_limit = True
            print(f"[location] route={source_route_name} routeId={route_id} 조회 실패: {exc}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        normalized_rows = []
        for item in items:
            normalized = normalize_location_item(route_id, item)
            if not normalized["route_name"]:
                normalized["route_name"] = source_route_name
            if normalized["crowded_code"] not in ("", None):
                normalized_rows.append(normalized)

        print(f"[location] route={source_route_name} routeId={route_id}: {len(normalized_rows)}건 저장")
        all_rows.extend(normalized_rows)
        time.sleep(REQUEST_DELAY_SECONDS)

    append_location_rows(all_rows)
    print(f"혼잡도 raw 저장 완료: {CROWDED_RAW_FILE}")
    if saw_rate_limit and not all_rows:
        raise RuntimeError("API 요청 제한(429)으로 이번 회차 수집이 차단되었습니다.")


def print_usage() -> None:
    print("사용법")
    print("python collector.py routes        -> 수원시 CSV 기준 노선명으로 routeId 수집")
    print("python collector.py commute       -> 통학 루트 후보 노선(11-1, 13-4, 20) routeId 수집")
    print("python collector.py crowded       -> 저장된 routeId로 혼잡도 1회 수집")
    print("python collector.py check         -> 키와 파일 상태 확인")


def main() -> None:
    ensure_route_csv()
    ensure_raw_csv()

    route_key, location_key = get_api_keys()
    command = sys.argv[1] if len(sys.argv) > 1 else "check"

    if command == "check":
        print(f"Route key loaded: {'YES' if route_key else 'NO'}")
        print(f"Location key loaded: {'YES' if location_key else 'NO'}")
        print(f"Local route source: {LOCAL_ROUTE_SOURCE}")
        print_usage()
    elif command == "routes":
        collect_route_ids(limit=10)
    elif command == "commute":
        collect_commute_route_ids()
    elif command == "crowded":
        collect_crowded_once(limit=None)
    else:
        print_usage()


if __name__ == "__main__":
    main()
