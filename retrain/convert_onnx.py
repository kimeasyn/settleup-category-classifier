import os, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(ROOT / "retrain" / "output"))

def convert():
    # 채택 여부 확인
    result_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(result_path) as f:
        result = json.load(f)

    if not result["adopted"]:
        print("모델 기각됨. ONNX 변환 스킵.")
        return

    model_dir = os.path.join(OUTPUT_DIR, "model_new")
    onnx_dir = os.path.join(OUTPUT_DIR, "model_onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # 모델 + 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        attn_implementation="eager",
    )
    model.eval()

    # 더미 입력
    dummy = tokenizer("테스트", return_tensors="pt", padding="max_length", truncation=True, max_length=64)

    # ONNX 변환
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        dynamo=False,
    )

    # 토크나이저도 ONNX 디렉토리에 복사
    tokenizer.save_pretrained(onnx_dir)

    print(f"✅ ONNX 변환 완료 → {onnx_path}")
    print(f"   모델 크기: {os.path.getsize(onnx_path) / 1024 / 1024:.1f}MB")

if __name__ == "__main__":
    convert()