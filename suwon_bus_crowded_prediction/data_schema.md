# Data Schema

## 1. Route List

수원시 대상 노선 목록을 먼저 확보합니다.

권장 컬럼:

- `route_id`
- `route_name`
- `route_type_cd`
- `start_station_name`
- `end_station_name`
- `area_name`

## 2. Realtime Crowded Collection

GBIS 버스 위치정보 조회에서 반복 수집할 컬럼입니다.

필수 컬럼:

- `collected_at`
- `route_id`
- `route_name`
- `vehicle_id`
- `plate_no`
- `station_seq`
- `station_id`
- `station_name`
- `crowded_code`
- `crowded_label`

추가로 있으면 좋은 컬럼:

- `remain_seat_cnt`
- `low_plate`
- `location_no`
- `turn_seq`

## 3. Modeling Dataset

모델 입력용으로는 차량별 또는 노선별로 시간순 정렬된 혼잡도 시퀀스를 만듭니다.

예시:

- 입력: 최근 6개 시점 `crowded_code`
- 출력: 다음 시점 `crowded_code`

권장 전처리:

- `crowded_code`는 1~4 정수 유지
- 시간순 정렬
- 결측 구간 제거 또는 직전값 보간
- 노선별 또는 차량별 그룹 생성
