# EcoSortNet

PyTorch CNN으로 쓰레기 이미지를 6종(cardboard, glass, metal, paper, plastic, trash)으로 분류하고, 분류 결과에 맞는 분리배출 가이드를 안내하는 프로젝트입니다.

## 구성

- `main.py` — 모델 정의(`ImprovedCNN`), 학습/평가/추론, 결과 시각화(학습 곡선, confusion matrix, 오분류 사례)
- `experiment.py` — augmentation 유무, learning rate, epoch 수 등을 바꿔가며 비교하는 실험 스크립트
- `run_final.py` — 학습 → 분석 → 샘플 추론까지 이어지는 파이프라인을 순서대로 실행
- `results/` — 학습 곡선, confusion matrix 등 실행 결과 예시

## 실행

```bash
pip install -r requirements.txt
```

`data/` 아래에 클래스별 폴더(`cardboard/`, `glass/`, `metal/`, `paper/`, `plastic/`, `trash/`)로 이미지를 넣은 뒤:

```bash
python main.py --mode train      # 학습
python main.py --mode analyze    # 성능 분석 및 시각화
python main.py --mode infer --image <path>   # 단일 이미지 추론
```

또는 전체 파이프라인을 한 번에:

```bash
python run_final.py
```

데이터셋과 학습된 모델 가중치는 용량 문제로 저장소에 포함하지 않았습니다.
