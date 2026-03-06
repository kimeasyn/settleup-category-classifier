"""
카테고리 분류 API 서버
- ONNX Runtime으로 klue/bert-base fine-tuned 모델 서빙
- 6-class 분류: 식비 / 교통 / 숙박 / 관광 / 쇼핑 / 기타
"""
from contextlib import asynccontextmanager
import os
from pathlib import Path

import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

prediction_counter = Counter('prediction_total', 'Predictions by category', ['category'])
prediction_confidence = Histogram('prediction_confidence', 'Prediction confidence scores')
prediction_latency = Histogram('prediction_latency_seconds', 'Inference latency')

# ── 설정 ──
# 로컬: 프로젝트 루트의 model_onnx/, Docker: /app/model_onnx/
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model_onnx"))
CATEGORIES = ["식비", "교통", "숙박", "관광", "쇼핑", "기타"]
MAX_LENGTH = 64

# ── 전역 상태 ──
tokenizer: Tokenizer | None = None
session: ort.InferenceSession | None = None


# ── Lifespan ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, session
    print(f"모델 로드 중... ({MODEL_DIR})")
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.enable_padding(length=MAX_LENGTH, pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    session = ort.InferenceSession(
        str(MODEL_DIR / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    print("모델 로드 완료")
    yield


app = FastAPI(title="Category Classifier API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


# ── 스키마 ──
class PredictRequest(BaseModel):
    description: str


class PredictResponse(BaseModel):
    description: str
    category: str
    confidence: float
    all_categories: dict[str, float]


# ── 엔드포인트 ──
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    encoded = tokenizer.encode(req.description)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoded.type_ids], dtype=np.int64)

    logits = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })[0]

    # softmax
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = (exp / exp.sum(axis=-1, keepdims=True))[0]

    pred_id = int(probs.argmax())
    all_categories = {cat: round(float(probs[i]), 4) for i, cat in enumerate(CATEGORIES)}

    prediction_latency.observe(time.time() - start)
    prediction_counter.labels(category=CATEGORIES[pred_id]).inc()
    prediction_confidence.observe(float(probs[pred_id]))
    
    return PredictResponse(
        description=req.description,
        category=CATEGORIES[pred_id],
        confidence=round(float(probs[pred_id]), 4),
        all_categories=all_categories,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
