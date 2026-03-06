FROM python:3.10-slim

WORKDIR /app

# 의존성 먼저 설치 (캐시 활용)
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드
COPY serving/app.py .

# ONNX 모델 + 토크나이저
COPY model_onnx/ /app/model_onnx/

# Docker 환경에서의 모델 경로
ENV MODEL_DIR=/app/model_onnx

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
