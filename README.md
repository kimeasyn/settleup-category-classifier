# Category Classifier

klue/bert-base 모델을 fine-tuning하여 지출 내역의 카테고리(식비/교통/숙박/관광/쇼핑/기타)를 자동 분류하는 프로젝트.

## 디렉토리 구조

```
category-classifier/
├── data/
│   ├── raw/                  ← 원본 CSV 파일 (01~06.csv)
│   ├── merge_csv.py          ← 원본 CSV → result.xlsx 병합
│   ├── prepare_data.py       ← result.xlsx → train/test 분리
│   ├── result.xlsx           ← 라벨링된 전체 데이터
│   ├── train.csv             ← 학습 데이터 (889건)
│   └── test.csv              ← 평가 데이터 (223건)
├── training/
│   └── train.py              ← klue/bert-base fine-tuning
├── export/
│   ├── convert_onnx.py       ← PyTorch → ONNX 변환
│   └── checkdata.py          ← 모델 추론 테스트
├── serving/                  ← API 서버 (구현 예정)
├── model_final/              ← 최종 PyTorch 모델 (git 제외)
├── model_onnx/               ← ONNX 모델 (git 제외)
└── model_output/             ← 학습 체크포인트 (git 제외)
```

## 실행 순서

```bash
# 1. 원본 CSV를 xlsx로 병합 (라벨링 작업용)
python data/merge_csv.py

# 2. 라벨링 완료된 xlsx를 train/test로 분리
python data/prepare_data.py

# 3. 모델 학습
python training/train.py

# 4. ONNX 변환 (서빙용)
python export/convert_onnx.py
```

## 학습 결과

| Epoch | Accuracy | F1 (weighted) | Eval Loss |
|-------|----------|---------------|-----------|
| 1     | 0.8879   | 0.8884        | 0.3576    |
| 2     | 0.9103   | 0.9116        | 0.3698    |
| **3** | **0.9283** | **0.9292**  | **0.3771** |
| 4     | 0.9238   | 0.9239        | 0.3941    |
| 5     | 0.9283   | 0.9282        | 0.3981    |

- Best model: Epoch 3 (F1 = 0.9292)
- 데이터: train 889건 / test 223건 (8:2 stratified split)
- 하이퍼파라미터: batch_size=8, lr=2e-5, max_length=64

## 카테고리

| 라벨 | 예시 |
|------|------|
| 식비 | 스타벅스 아메리카노 |
| 교통 | 인천공항 택시 |
| 숙박 | 롯데호텔 2박 |
| 관광 | 경복궁 입장료 |
| 쇼핑 | 올리브영 선크림 |
| 기타 | 여행자보험 |
