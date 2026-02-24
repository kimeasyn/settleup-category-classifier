"""
fine-tuning된 klue/bert-base 모델을 ONNX로 변환
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# ── 프로젝트 루트 ──
ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = str(ROOT / "model_final")
ONNX_DIR = str(ROOT / "model_onnx")
ONNX_PATH = f"{ONNX_DIR}/model.onnx"

# 모델, 토크나이저 로드
print("모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
model.to("cpu")

# 더미 입력 생성 (ONNX 변환 시 입력 형태를 알려주기 위한 것)
dummy_text = "스타벅스 아메리카노"
dummy_input = tokenizer(dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

# ONNX 변환
print("ONNX 변환 중...")
Path(ONNX_DIR).mkdir(exist_ok=True)

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input["token_type_ids"]),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "token_type_ids": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
)

# 토크나이저도 같이 저장 (서빙 서버에서 필요)
tokenizer.save_pretrained(ONNX_DIR)

# 파일 크기 확인
onnx_size = Path(ONNX_PATH).stat().st_size / 1024 / 1024
print(f"\n변환 완료: {ONNX_PATH} ({onnx_size:.1f}MB)")

# 변환된 모델로 추론 테스트
print("\nONNX 추론 테스트...")
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(ONNX_PATH)

tests = ["스타벅스 아메리카노", "인천공항 택시", "롯데호텔 2박", "경복궁 입장료", "올리브영 선크림", "여행자보험"]
id2label = model.config.id2label

for text in tests:
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=64)
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"],
    })
    probs = np.exp(outputs[0]) / np.exp(outputs[0]).sum(axis=-1, keepdims=True)
    pred_id = probs.argmax()
    conf = probs[0][pred_id]
    print(f"  {text} → {id2label[pred_id]} ({conf:.1%})")
