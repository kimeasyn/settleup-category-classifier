from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(ROOT / "model_final")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to('cpu')
model.eval()

tests = ['스타벅스 아메리카노', '인천공항 택시', '롯데호텔 2박', '경복궁 입장료', '올리브영 선크림', '여행자보험']
for text in tests:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)
    pred_id = probs.argmax().item()
    conf = probs[0][pred_id].item()
    print(f'{text} → {model.config.id2label[pred_id]} ({conf:.1%})')
