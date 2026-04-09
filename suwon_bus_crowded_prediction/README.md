# Suwon Bus Crowded Prediction

GBIS `crowded` 값을 수집해 수원시 버스 혼잡도를 예측하고, LSTM과 GRU 성능을 비교하는 프로젝트입니다.

## Project Structure

- `collector.py`: GBIS API에서 수원시 노선의 실시간 혼잡도를 수집합니다.
- `app.py`: 수집 데이터 현황과 모델 비교 결과를 보여주는 Streamlit 앱입니다.
- `data_schema.md`: 어떤 데이터를 저장해야 하는지 정리한 문서입니다.

## Workflow

1. 수원시 노선 목록을 정합니다.
2. `collector.py`로 일정 주기마다 `crowded`를 수집합니다.
3. 누적 데이터로 LSTM, GRU 모델을 학습합니다.
4. `app.py`에서 데이터 분석과 예측 결과를 시연합니다.
