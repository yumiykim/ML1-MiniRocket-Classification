# 데이터 가이드

## 원본 파일 위치

이 repository에는 원본 데이터(`train.csv`, `test.csv`)가 포함되어 있지 않습니다.  
동국대 ISE4045 수업 경진대회 제공 원본 데이터를 이 폴더에 배치하면 실행할 수 있습니다.

## 준비 방법

### 1. CSV → npy 변환

원본 CSV 파일을 `data/` 폴더에 넣고 아래 스크립트를 실행하세요:

```python
import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv")  # (10000, 2049): X0~X2047 + class_idx
test  = pd.read_csv("data/test.csv")   # (3590,  2048): X0~X2047

X_train = train.drop(columns=["class_idx"]).values.astype(np.float32)
y_train = train["class_idx"].values.astype(np.int64)
X_test  = test.values.astype(np.float32)

np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy",  X_test)
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
```

### 2. folds.npy 생성 (또는 기존 파일 사용)

재현성을 위해 `folds.npy`가 이미 포함되어 있습니다. 직접 생성하려면:

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

y = np.load("data/y_train.npy")
folds = np.zeros(len(y), dtype=np.int64)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for f, (_, va) in enumerate(skf.split(y, y)):
    folds[va] = f
np.save("data/folds.npy", folds)
```

## 예상 파일 구조

```
data/
├── train.csv          # 원본 (미포함, 직접 준비)
├── test.csv           # 원본 (미포함, 직접 준비)
├── X_train.npy        # (10000, 2048) float32  변환 후 생성
├── y_train.npy        # (10000,)      int64    변환 후 생성
├── X_test.npy         # (3590,  2048) float32  변환 후 생성
└── folds.npy          # (10000,)      int64    재현성용 포함
```

## 데이터 특성

| 항목 | 값 |
|------|-----|
| 행(샘플) 수 | train 10,000 / test 3,590 |
| 열(feature) 수 | 2,048 (X0~X2047, 수치형) |
| 클래스 수 | 9 (0~8) |
| 결측치 | 없음 |
| 전체 평균 | ≈ 0 |
| 클래스별 샘플 수 | 869 ~ 1,490 (약간 불균형) |

데이터의 구조적 특성(시계열 여부, 공변량 shift 분석 등)은 [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) 참조.
