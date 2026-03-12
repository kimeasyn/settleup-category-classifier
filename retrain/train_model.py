# retrain/train_model.py
"""
재학습 스크립트
- merged_train.csv로 학습
- 기존 모델보다 accuracy 높을 때만 저장
"""
import os, json
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(ROOT / "retrain" / "output"))

MODEL_NAME = "klue/bert-base"
CATEGORIES = ["식비", "교통", "숙박", "관광", "쇼핑", "기타"]
LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}

# 기존 모델 accuracy (이것보다 좋아야 채택)
BASELINE_ACCURACY = float(os.getenv("BASELINE_ACCURACY", "0.91"))

BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 2e-5

def train():
    # ── 데이터 로드 ──
    data_path = os.path.join(OUTPUT_DIR, "merged_train.csv")
    df = pd.read_csv(data_path)
    print(f"총 데이터: {len(df)}건")

    # train/test 분리 (8:2)
    # train_df, test_df = train_test_split(
    #     df, test_size=0.2, stratify=df["category"], random_state=42
    # )
    # 변경 후: 기존 test.csv를 고정 사용
    train_df = df  # merged 전체를 학습에 사용
    test_df = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    print(f"train: {len(train_df)}건, test: {len(test_df)}건")

    # 라벨 변환
    train_df["label"] = train_df["category"].map(LABEL2ID)
    test_df["label"] = test_df["category"].map(LABEL2ID)

    train_dataset = Dataset.from_pandas(train_df[["description", "label"]].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[["description", "label"]].reset_index(drop=True))

    # ── 토크나이저 ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["description"], padding="max_length", truncation=True, max_length=64)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # ── 모델 ──
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
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    # ── 학습 ──
    model_output = os.path.join(OUTPUT_DIR, "checkpoints")
    training_args = TrainingArguments(
        output_dir=model_output,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,   # 배치 1 x 4 = 실효 배치 4
        gradient_checkpointing=True,     # 메모리 절약 핵심
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n학습 시작 (baseline accuracy: {BASELINE_ACCURACY})")
    trainer.train()

    # ── 평가 ──
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    print(f"\n{'='*50}")
    print(classification_report(labels, preds, target_names=CATEGORIES))
    print(f"Accuracy: {accuracy:.4f} (baseline: {BASELINE_ACCURACY})")
    print(f"F1: {f1:.4f}")

    # ── 채택 여부 판단 ──
    if accuracy >= BASELINE_ACCURACY:
        save_dir = os.path.join(OUTPUT_DIR, "model_new")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

        # 결과 저장
        result = {"accuracy": accuracy, "f1": f1, "adopted": True}
        with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
            json.dump(result, f)

        print(f"\n✅ 채택! 새 모델 저장 → {save_dir}")
    else:
        result = {"accuracy": accuracy, "f1": f1, "adopted": False}
        with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
            json.dump(result, f)

        print(f"\n❌ 기각. accuracy {accuracy:.4f} < baseline {BASELINE_ACCURACY}")

if __name__ == "__main__":
    train()