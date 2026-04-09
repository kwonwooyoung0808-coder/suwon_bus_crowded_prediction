from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import csv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_FILE = DATA_DIR / "bus_crowded_raw.csv"
ARCHIVE_DIR = DATA_DIR / "archive"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)

    if RAW_FILE.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = ARCHIVE_DIR / f"bus_crowded_raw_before_commute_{timestamp}.csv"
        shutil.copy2(RAW_FILE, backup_path)
        print(f"기존 raw 파일 백업 완료: {backup_path}")

    with RAW_FILE.open("w", newline="", encoding="utf-8-sig") as f:
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

    print(f"통학 루트 전용 raw 파일 초기화 완료: {RAW_FILE}")


if __name__ == "__main__":
    main()
