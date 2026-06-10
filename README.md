# 다중분류 예측 — 시계열 앙상블 분류

> **머신러닝1** 수업 경진대회 프로젝트 (2026년 1학기)  
> 28명 참가 | 최종 **1위** | 리더보드 최고점: **0.8713**

---

## Overview

| 항목 | 내용 |
|------|------|
| **문제** | 9-클래스 다중분류, 2048차원 feature, 평가지표 accuracy |
| **데이터** | train 10,000행 / test 3,590행, 결측치 없음 |
| **핵심 발견** | 2048개 feature = **길이 2048의 시계열** (lag-1 상관 0.888) |
| **최종 모델** | 7개 시계열 전용 베이스 모델 → Logistic Stacking → EM 사전보정 |
| **정직 추정** | Nested CV 0.9136 (선택편향 제거) |
| **리더보드** | **0.8713 / 1위** (2위 0.8710, 0.0003 차이) |

---

## Results

| 지표 | 값 | 설명 |
|------|-----|------|
| Nested CV (정직) | **0.9136** | 선택편향 제거한 정직 추정치 |
| Full-OOF refit (낙관) | 0.9228 | 참고용, 성능 지표로 쓰지 않음 |
| 리더보드 (실제 test) | **0.8713** | 대회 실점, 최종 **1위** |
| Nested CV → LB 갭 | -0.0423 | 공변량 shift 주원인 (domain AUC 0.946) |

| 날짜 | 단계 | 방법 | Nested CV | 리더보드 |
|------|------|------|-----------|---------|
| 6/4 | 초기 | SVM + HGB + RF 스택 | OOF 0.7294 | 0.6858 |
| 6/5 | 전환 | 7모델 시계열 스택 + EM 보정 | 0.9136 | **0.8655** (1위) |
| 6/6 | 개선 | 스케일-불변 변형 추가 | 0.8931 | **0.8713** (1위) |

0.686에서 0.871로 올린 건 더 복잡한 모델이 아니라 feature 해석 방식이었다.

---

## Key Finding

처음엔 2048개 feature를 서로 독립적인 변수로 보고 접근했다. SVM-RBF 0.60, HistGBM 0.71 — 모델을 바꿔도 여기서 더 오르지 않았다.

열 순서를 살펴보니:
- 인접 feature 간 상관계수: **0.888** (열 순서가 의미 있음)
- 행 자기상관(ACF lag-1) 평균: 0.61
- 클래스별 ACF 차이: class 1 ≈ -0.11 (노이즈형) / class 4, 6, 8 ≈ 0.80~0.83 (평탄형)

각 행은 클래스별 AR 파라미터로 생성된 길이 2048의 시계열이었다. 클래스는 AR 구조·스케일·평균·왜도로 구분된다.

---

## Approach

### Step 1 — Time-series Feature Engineering

각 행에서 총 **247종**의 피처를 추출:

```
ACF(lag 1~48)           48개  │  AR(24) 계수           24개
PACF(16-lag)            16개  │  Welch PSD(64-band)    ~64개
diff 통계                 9개  │  분위수(13+2)          15개
Zero-crossing rate        1개  │  Hurst 지수              1개
Higuchi FD                1개  │  Rolling stats          20개
스펙트럼 기술자            5개  │  (+ 기타)              ...
```

### Step 2 — Seven Time-series Base Models

| 모델 | 방법 | OOF 정확도 |
|------|------|-----------|
| **MiniRocket** | 10,000개 랜덤 컨볼루션 커널 + RidgeCV | **0.8919** |
| **TS-Kitchen** | 247종 통계 피처 + LGB/HGB blend | **0.864** |
| **CNN1D** | 3채널(raw/std/diff) 1D CNN | ~0.82 |
| Spectral | FFT/Welch PSD + MLP/LGBM | 0.743 |
| AR Likelihood | 클래스별 AR(24) 우도 feature + HGB | 0.696 |
| LightGBM | raw 2048 → GBDT | 0.701 |
| LGBM+PCA | PCA(256) + top-분산 256 병합(512차원) → LightGBM | ~0.711 |

### Step 3 — Nested CV Logistic Stacking

```python
for outer_fold in range(5):
    # C selected on inner 4-fold only — held-out labels not exposed to selection
    best_C = select_C_on_inner_folds(train_OOF[inner_folds])
    acc = evaluate_on(outer_fold)
# nested CV = 0.9136 | 63-dim log-clip prob → LogisticRegression(C=0.02)
```

선택편향: K=20 탐색에서 +0.17pp 낙관 → 보정 추정치 **~0.9071**

### Step 4 — BBSE/EM Label-Shift Correction

```
문제: test 데이터의 class prior가 train과 다름
  → class 0/8 과예측, class 2/5/7 과소예측

해법: EM 고정점 반복 (Saerens et al. 2002)
  - test 예측 확률 P에서 test prior π_test 추정
  - 행별 재가중: Q ∝ P × (π_test / π_train)

검증: 합성 shift(train 0/8 2배, 2/5/7 0.5배) 200회 → +0.22pp (98% 양수)
적용: test argmax 3.1%만 변경(보수적)
```

---

## Validation

### Leakage Prevention
- 행단위 변환(ACF, FFT, 분위수, AR 우도) — 자기 행만 사용, cross-row 누설 0
- Cross-row fit(scaler, PCA) — fold 학습행에만 fit, test는 transform만
- 모든 OOF — 고정 `folds.npy` (StratifiedKFold 5, seed=42)
- 앙상블 recipe 선택 — Nested CV (선택편향 제거)
- ts_kitchen blend / ar_likelihood variant — per-fold honest

### Adversarial Validation

| 검증 | 결과 | 해석 |
|------|------|------|
| Permutation test | 라벨 셔플 OOF 0.1334 (실제 0.91) | CV 누설 없음 |
| Fresh-split test | seed-42 vs seed-123: 갭 -0.0007 | 분할 독립성 확인 |
| Bootstrap 95% CI | [0.9030, 0.9143] | 안정적 추정 |
| Covariate shift | Domain AUC 0.946 (강한 shift) | LB 갭 주원인 |

### Honest Evaluation

```
Nested CV (정직)   : 0.9136  ← 배포 결정에 사용
Full-OOF (낙관)    : 0.9228  ← 참고용, 성능 주장에 사용 X
Leaderboard (실제) : 0.8713

갭 분석:
  공변량 shift 페널티  : -1.9pp  (IW-OOF 기준)
  라벨 shift 이득      : +1.1pp  (EM 보정)
  선택편향             : +0.17pp (K=20)
  → 기대 범위: 0.87~0.91
```

---

## How to Run

### Setup

```bash
git clone https://github.com/yumiykim/ML1-MiniRocket-Classification.git
cd ML1-MiniRocket-Classification
pip install -r requirements.txt
```

### Data

```
data/
├── train.csv    # (10000, 2049): X0~X2047 + class_idx (0~8)
└── test.csv     # (3590, 2048):  X0~X2047
```

> 데이터는 동국대 ISE4045 수업 경진대회 제공 원본으로, 이 repository에는 포함되지 않습니다.  
> `data/folds.npy`(고정 5-fold 할당)는 재현성을 위해 포함되어 있습니다.

### Run

```bash
# 전체 파이프라인 (base 모델 재학습 → 앙상블 → 제출 파일)
python pipelines/final_pipeline.py --rebuild

# 캐시(oof/, test_pred/)가 있으면 앙상블만 재실행
python pipelines/final_pipeline.py

# 출력: submission.csv (3590행, class 0~8)
```

### Individual Models

```bash
python models/minirocket_train.py    # OOF 0.8919
python models/ts_kitchen_train.py    # OOF 0.864
python models/cnn1d_v2_train.py      # OOF ~0.82
python models/ar_likelihood_train.py # OOF 0.696
python models/spectral_train.py      # OOF 0.74
python models/lightgbm_train.py      # OOF 0.70
python models/lgbm_pca_train.py      # OOF ~0.711
```

---

## Structure

```
.
├── pipelines/
│   ├── final_pipeline.py    # 1위 달성 엔드투엔드 파이프라인
│   ├── round6_pipeline.py   # 6/6 제출 — 10모델 시계열 스택 (final과 비교용)
│   └── utils.py             # 공통 함수 (clip_log, EM보정, nested CV)
│
├── models/
│   ├── minirocket_train.py  # MiniRocket + RidgeCV (OOF 0.8919)
│   ├── ts_kitchen_train.py  # TS Feature Kitchen (OOF 0.864)
│   ├── ar_likelihood_train.py # AR 우도 기반 (OOF 0.696)
│   ├── spectral_train.py    # Welch PSD 스펙트럼 (OOF 0.74)
│   ├── lightgbm_train.py    # LightGBM raw (OOF 0.70)
│   ├── lgbm_pca_train.py    # PCA(256)+top-var(256) → LightGBM (OOF ~0.711)
│   ├── cnn1d_v2_train.py    # 1D CNN 3채널 (OOF ~0.82)
│   ├── minirocket_std_train.py  # MiniRocket 스케일-불변 변형 (round6)
│   ├── ts_kitchen_std_train.py  # TS-Kitchen 스케일-불변 변형 (round6)
│   └── cnn1d_std_train.py       # CNN1D 스케일-불변 변형 2채널 (round6)
│
├── docs/
│   └── METHODOLOGY.md       # 방법론 상세 (수식, 의사코드)
│
├── data/
│   └── folds.npy            # 재현성용 고정 5-fold 할당
│
├── requirements.txt
└── .gitignore
```

---

## Key Takeaways

**1. SVM, HistGBM, RandomForest 세 가지를 써봤는데 0.69에서 더 오르지 않았습니다.**  
인접 컬럼 간 상관이 0.888이라는 걸 확인하고 시계열 모델로 바꿨습니다. MiniRocket 하나가 0.89였습니다. 모델이 나쁜 게 아니라 feature를 보는 방식이 달랐습니다.

**2. full-OOF 정확도 0.9228이 실제 성능을 보장하지 않았습니다.**  
Nested CV로 선택편향을 제거하면 0.9136이었습니다. 리더보드 0.8713이 나왔을 때 공변량 shift(-1.9pp)로 갭 원인을 미리 설명할 수 있었습니다.

**3. OOF pairwise 상관을 계산하며 앙상블 구성 모델을 결정했습니다.**  
MiniRocket(0.8919)이 가장 강했지만, AR 우도·스펙트럼·CNN처럼 접근이 다른 모델을 넣을 때만 Nested CV가 올랐습니다.

---

## Stack

`Python` `scikit-learn` `LightGBM` `PyTorch` `sktime`  
`MiniRocket` `Nested CV` `Logistic Stacking` `EM Label-Shift Correction` `Covariate Shift Analysis`

---

*동국대학교 산업시스템공학과 · 머신러닝1 (2026-1학기)*
