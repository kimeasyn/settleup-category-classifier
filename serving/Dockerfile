FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# 모델 파일은 빌드 시 복사 (docker build 컨텍스트에 model_onnx/ 포함 필요)
COPY model_onnx/ /app/model_onnx/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
