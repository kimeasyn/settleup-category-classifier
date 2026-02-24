"""
SettleUp 카테고리 분류 모델 fine-tuning
- 모델: klue/bert-base
- 태스크: 6-class 텍스트 분류 (식비/교통/숙박/관광/쇼핑/기타)
"""
import os
from pathlib import Path
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── 프로젝트 루트 ──
ROOT = Path(__file__).resolve().parent.parent

# ── 설정 ──
MODEL_NAME = "klue/bert-base"
CATEGORIES = ["식비", "교통", "숙박", "관광", "쇼핑", "기타"]
LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}
OUTPUT_DIR = str(ROOT / "model_output")
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5

# ── 데이터 로드 ──
print("데이터 로드 중...")
train_df = pd.read_csv(ROOT / "data" / "train.csv")
test_df = pd.read_csv(ROOT / "data" / "test.csv")
print(f"train: {len(train_df)}건, test: {len(test_df)}건")

# category 텍스트 → 숫자 라벨로 변환
train_df["label"] = train_df["category"].map(LABEL2ID)
test_df["label"] = test_df["category"].map(LABEL2ID)

# HuggingFace Dataset으로 변환
train_dataset = Dataset.from_pandas(train_df[["description", "label"]])
test_dataset = Dataset.from_pandas(test_df[["description", "label"]])

# ── 토크나이저 ──
print("토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["description"], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# ── 모델 ──
print("모델 로드 중...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(CATEGORIES),
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)

# ── 평가 함수 ──
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ── 학습 설정 ──
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=False,  # CPU에서는 False
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ── 학습 시작 ──
print("\n학습 시작...")
print(f"모델: {MODEL_NAME}")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
print(f"디바이스: {training_args.device}")
print("-" * 50)

trainer.train()

# ── 최종 평가 ──
print("\n최종 평가 중...")
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

print("\n" + "=" * 50)
print("분류 성능 리포트")
print("=" * 50)
print(classification_report(labels, preds, target_names=CATEGORIES))

# ── 모델 저장 ──
SAVE_DIR = str(ROOT / "model_final")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\n모델 저장 완료: {SAVE_DIR}")

# ── 테스트 추론 ──
print("\n샘플 추론 테스트:")
test_texts = [
    "스타벅스 아메리카노",
    "인천공항 택시",
    "롯데호텔 2박",
    "경복궁 입장료",
    "올리브영 선크림",
    "여행자보험",
]

model.eval()
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_id = probs.argmax().item()
    confidence = probs[0][pred_id].item()
    print(f"  '{text}' → {ID2LABEL[pred_id]} ({confidence:.1%})")
