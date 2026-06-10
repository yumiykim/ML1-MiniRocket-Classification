# 방법론 상세 | Methodology Details

## 1. 데이터 및 문제 설정

- **입력**: X0~X2047 (2048차원 연속형), train 10,000행, test 3,590행
- **출력**: 9개 클래스 (0~8), 평가지표: accuracy
- **클래스 분포**: 869~1490 (약간 불균형)
- **전처리**: 결측치 없음, 전체 평균 ≈ 0

### 시계열 구조 발견 근거

| 통계 | 값 | 의미 |
|------|-----|------|
| lag-1 컬럼 상관 | **0.888** | 열 순서가 매우 의미 있음 |
| 행 ACF lag-1 평균 | **0.61** | 강한 시간 종속성 |
| 클래스별 ACF lag-1 | **-0.11 ~ 0.83** | 클래스 구분의 핵심 신호 |
| 가설 | 클래스별 AR 파라미터를 가진 합성 시계열 (uint8 양자화 후 표준화) |

---

## 2. 고정 Cross-Validation 프로토콜

```python
# folds.npy: 모든 베이스 모델이 공유하는 고정 5-fold 할당
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for f, (_, va) in enumerate(skf.split(X, y)):
    folds[va] = f
```

- 모든 베이스 모델이 **동일한 folds.npy** 사용 → OOF 행이 정렬되어 stacking 가능
- test 예측 = 5개 fold 모델의 예측 확률 **평균**

---

## 3. 베이스 모델 상세

### 3.1 MiniRocket (OOF 0.8919)

```
입력: (n, 1, 2048) 시계열
  ↓
MiniRocket(num_kernels=10,000): 랜덤 컨볼루션 커널 → PPV feature (n, 10000)
  ↓
StandardScaler (fold 내 학습행에만 fit)
  ↓
RidgeClassifierCV(alphas=logspace(-3,3,13))
  ↓
소프트맥스 온도 튜닝 (fold-train 내부 hold-out으로 선택 — 선택편향 제거)
  ↓
OOF 확률 (n_val, 9)
```

**선택 근거**: 랜덤 커널은 학습 파라미터 없음 → fold 간 재현 가능, 빠른 학습, 강한 시계열 표현력

### 3.2 TS-Kitchen (OOF ~0.864)

**247종 통계 피처** (행별 독립 → cross-row 누설 0):

| 카테고리 | 피처 수 | 예시 |
|---------|---------|------|
| ACF | 48 | lag 1~48 자기상관 |
| AR(24) 계수 | 24 | Yule-Walker |
| PACF | 16 | 편자기상관 16-lag |
| 분위수 | 15 | 13분위수 + 스프레드 |
| Welch PSD | ~73 | 64-band log-PSD + 스펙트럼 기술자 5개 |
| diff 통계 | 9 | 1/2차 차분 |
| Rolling stats | 20 | 4개 윈도우 × std/variability |
| 기타 | 10+ | ZCR, Hurst, Higuchi FD 등 |

**blend 선택**: 각 outer fold에서 나머지 4-fold OOF로만 LGB/HGB blend weight 선택 (fold-honest)

### 3.3 1D CNN (OOF ~0.82)

```
입력 채널 3개 (행별 독립 변환):
  channel 0: raw 시계열
  channel 1: per-row z-normalization
  channel 2: 1차 차분 (zero-padded)

아키텍처:
  AvgPool(pre=2) → Conv1d(3,32,k=7,s=2) → BN → GELU
  → [Conv1d(64,k=5)+BN+GELU+MaxPool(2)] × 3
  → AdaptiveAvgPool + AdaptiveMaxPool → concat(256)
  → Dropout(0.3) → Linear(256, 9)

학습: AdamW(lr=2e-3), CosineAnnealingLR, label_smoothing=0.1
조기종료: patience=5
```

### 3.4 AR Likelihood (OOF 0.696)

```
각 클래스 k에 대해 AR(p) 모델 추정 (p=5, 10, 20):
  - 클래스별 평균 자기공분산 → Yule-Walker → φ_k, σ²_k

feature matrix (n, 45):
  - 조건부 Gaussian 로그우도: AR(p=5,10,20) × 9클래스 = 27
  - row-mean/row-std의 클래스별 Gaussian 우도 = 18

Variant 선택 (fold-honest): Bayes / LR / HGB 중 각 outer fold에서
나머지 4-fold OOF accuracy 기준 선택
```

---

## 4. Logistic Stacking

### 4.1 입력 구성

```python
X_stack = np.hstack([clip_log(OOF[m]) for m in STRONG])
# 7모델 × 9클래스 = 63차원
# clip_log(p) = log(clip(p, 1e-6, 1) / row_sum)
```

### 4.2 Nested CV 프로토콜 (선택편향 제거)

```
for outer_fold f in {0,1,2,3,4}:
  train_mask = (folds != f)
  
  # Inner CV: C 선택
  for inner_fold g in {folds != f 의 unique fold들}:
    fit LogisticRegression(C) on {train_mask AND folds != g}
    score on {train_mask AND folds == g}
  best_C = argmax mean inner accuracy
  
  # Outer 평가
  fit LogisticRegression(best_C) on {train_mask}
  score on {folds == f} → 정직 OOF 예측

nested_acc = mean(5개 outer fold 점수)  # 선택편향 없는 정직 추정
```

**결과**: logstack 7모델 nested CV = **0.9136**

선택편향 추정:
- K=20 비교, ρ=0.9 상관, σ=0.003 → analytic 낙관 ≈ **+0.17pp**
- 보정 추정치: **~0.9071**

---

## 5. BBSE/EM 라벨-shift 보정

### 5.1 배경

test 데이터의 클래스 prior가 train과 다름을 발견:
- train prior: 균형적 (class 7 제외 869~1490)
- test 추정 prior: class 0/8 과다, class 2/5/7 과소

### 5.2 알고리즘 (Saerens et al. 2002)

```python
def em_estimate_pi(P, pi_train, n_iter=2000, tol=1e-9):
    """EM 고정점 반복으로 test prior π_test 추정."""
    base = P / pi_train[None, :]
    pi = pi_train.copy()
    for _ in range(n_iter):
        # E-step: 각 샘플의 posterior
        num = base * pi[None, :]
        num /= num.sum(1, keepdims=True)
        # M-step: prior 갱신
        new_pi = num.mean(0); new_pi /= new_pi.sum()
        if np.max(np.abs(new_pi - pi)) < tol: return new_pi
        pi = new_pi
    return pi

# 적용
Q = P * (pi_test / pi_train)[None, :]
Q /= Q.sum(1, keepdims=True)  # test-only 변환
```

### 5.3 검증

합성 shift 테스트 (train OOF를 class 0/8 2배, 2/5/7 0.5배로 skew):
- 200회 반복: 이득 +0.22pp, **98% 양수** (oracle 천장 +0.27pp 근접)
- 실제 적용: test argmax 3.1%만 변경 (보수적)

---

## 6. 공변량 Shift 분석

### 6.1 Domain Classifier

```python
# train vs test 이진 분류기 학습
X_domain = np.vstack([X_train, X_test])
y_domain = np.array([0]*N + [1]*NT)

# 55종 TS feature 사용
domain_clf = LightGBM(...)
# 5-fold CV AUC
```

결과: **AUC 0.946** (0.5=no shift, 1.0=완전 분리)

주요 신호: scale/tail 특성 (energy, rms, extreme quantiles)  
→ "양자화 후 표준화" 합성 파라미터가 train↔test 간 다름 (조건부 공변량 shift)

### 6.2 Importance-Weighting

```
IW-OOF = OOF에 도메인 분류기 기반 가중치 적용
ESS (Effective Sample Size) = 1,496 / 10,000

IW-OOF 0.9040 vs 비가중 0.9228
→ 공변량 페널티 약 -1.9pp
```

### 6.3 순효과 예측

| 요인 | 방향 | 크기 |
|------|------|------|
| 공변량 shift | ↓ | -1.9pp |
| 라벨 shift (EM 보정) | ↑ | +1.1pp |
| 선택편향 | ↑ | +0.17pp |
| 표본오차 | ± | ±1.4pp (n=3590) |
| **합산 기대 LB** | | **~0.87~0.91** |
| **실측 LB** | | **0.8713** |

---

## 7. Code Audit

선택편향 관련 코드를 감사하고 다음을 수정했습니다:

| 이슈 | 원래 방식 | 수정 방식 |
|------|---------|---------|
| ts_kitchen blend 선택 | **전체 OOF accuracy** 기준 | **나머지 4-fold OOF만** 사용 (fold-honest) |
| ar_likelihood variant 선택 | 동일 | 동일 |
| 앙상블 배포 불일치 | greedy(0.8991) 하드코딩 | nested 최우수 logstack(0.9136)으로 통일 |

수정 후 실질적 효과: ts_kitchen OOF 0.864 유지 (성능 변동 없음, 선택편향 제거 절차로 변경)

---
