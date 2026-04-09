from __future__ import annotations

from datetime import datetime
import subprocess
import sys
import time


def run_crowded_collection() -> int:
    result = subprocess.run(
        [sys.executable, "collector.py", "crowded"],
        capture_output=True,
        text=True,
        encoding="cp949",
        errors="replace",
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] collector.py crowded")
    if result.stdout and result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr and result.stderr.strip():
        print(result.stderr.strip())
    return result.returncode


def main():
    minutes = 5
    rounds = 12

    if len(sys.argv) >= 2:
        minutes = int(sys.argv[1])
    if len(sys.argv) >= 3:
        rounds = int(sys.argv[2])

    print(f"{minutes}분 간격으로 {rounds}번 수집을 시작합니다.")
    print("중간에 멈추려면 Ctrl+C를 누르세요.")

    for idx in range(rounds):
        print(f"\n===== {idx + 1}/{rounds} 회차 =====")
        code = run_crowded_collection()
        wait_minutes = minutes
        if code != 0:
            print(f"collector.py 실행 실패: exit code {code}")
            wait_minutes = max(minutes, 30)
            print(f"요청 제한 또는 오류 가능성이 있어 {wait_minutes}분 후 다시 시도합니다.")

        if idx < rounds - 1:
            sleep_seconds = wait_minutes * 60
            print(f"다음 수집까지 {wait_minutes}분 대기합니다.")
            time.sleep(sleep_seconds)

    print("자동 수집이 완료되었습니다.")


if __name__ == "__main__":
    main()
