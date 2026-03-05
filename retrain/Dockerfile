FROM python:3.10-slim

WORKDIR /app

COPY retrain/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 재학습 스크립트
COPY retrain/ retrain/

# 기존 학습 데이터 (merge용)
COPY data/train.csv data/train.csv
COPY data/test.csv data/test.csv