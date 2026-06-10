"""
per-row 표준화 후 MiniRocket + RidgeClassifierCV — 스케일-불변 변형.
진폭 정보를 제거하고 시계열 형태·동역학만 학습. round6 다양성 기여 모델.
"""
import os
os.environ["OMP_NUM_THREADS"]   = "4"
os.environ["MKL_NUM_THREADS"]   = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"
import time
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

t0    = time.time()
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data")
NAME  = "minirocket_std"

X     = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
y     = np.load(os.path.join(DATA, "y_train.npy")).astype(np.int64)
folds = np.load(os.path.join(DATA, "folds.npy")).astype(np.int64)
Xt    = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)


def row_std(A):
    return ((A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)).astype(np.float32)


X  = row_std(X)
Xt = row_std(Xt)

n_train, L = X.shape
n_test     = Xt.shape[0]
n_classes  = 9
NUM_KERNELS = 10000

X3  = X.reshape(n_train, 1, L)
Xt3 = Xt.reshape(n_test,  1, L)

from sktime.transformations.panel.rocket import MiniRocketMultivariate

alphas    = np.logspace(-3, 3, 13)
oof       = np.zeros((n_train, n_classes), dtype=np.float32)
test_pred = np.zeros((n_test,  n_classes), dtype=np.float32)
fold_accs = []


def decision_to_proba(scores, temp):
    s = scores / temp
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


for f in range(5):
    tf  = time.time()
    tr  = folds != f
    va  = folds == f
    Xtr3, Xva3 = X3[tr], X3[va]
    ytr = y[tr]

    mr = MiniRocketMultivariate(num_kernels=NUM_KERNELS, n_jobs=4, random_state=42)
    mr.fit(Xtr3)
    Ftr = np.asarray(mr.transform(Xtr3)).astype(np.float32)
    Fva = np.asarray(mr.transform(Xva3)).astype(np.float32)
    Ftt = np.asarray(mr.transform(Xt3)).astype(np.float32)

    sc    = StandardScaler()
    Ftr_s = sc.fit_transform(Ftr).astype(np.float32)
    Fva_s = sc.transform(Fva).astype(np.float32)
    Ftt_s = sc.transform(Ftt).astype(np.float32)

    clf = RidgeClassifierCV(alphas=alphas)
    clf.fit(Ftr_s, ytr)

    # fold-train 내부 hold-out으로 소프트맥스 온도 선택 — 검증 fold 선택편향 방지
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
    cidx, hidx = next(skf.split(Ftr_s, ytr))
    clf_c = RidgeClassifierCV(alphas=alphas)
    clf_c.fit(Ftr_s[cidx], ytr[cidx])
    dsc_h = clf_c.decision_function(Ftr_s[hidx])
    best_t, best_ll = 1.0, 1e18
    yh = ytr[hidx]
    for temp in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        p  = decision_to_proba(dsc_h, temp)
        ll = -np.log(np.clip(p[np.arange(len(yh)), yh], 1e-12, 1)).mean()
        if ll < best_ll:
            best_ll, best_t = ll, temp

    dsc_va = clf.decision_function(Fva_s)
    dsc_tt = clf.decision_function(Ftt_s)
    oof[va]    = decision_to_proba(dsc_va, best_t)
    test_pred += decision_to_proba(dsc_tt, best_t) / 5.0

    acc = (oof[va].argmax(1) == y[va]).mean()
    fold_accs.append(acc)
    print(f"fold {f}: acc={acc:.4f} temp={best_t} nfeat={Ftr.shape[1]} ({time.time()-tf:.0f}s)", flush=True)

mean_acc = (oof.argmax(1) == y).mean()
print(f"\nPer-fold: {[round(a, 4) for a in fold_accs]}")
print(f"OOF 전체 acc: {mean_acc:.4f}", flush=True)

os.makedirs(os.path.join(BASE, "oof"),       exist_ok=True)
os.makedirs(os.path.join(BASE, "test_pred"), exist_ok=True)
np.save(os.path.join(BASE, "oof",       f"{NAME}.npy"), oof.astype(np.float32))
np.save(os.path.join(BASE, "test_pred", f"{NAME}.npy"), test_pred.astype(np.float32))
print(f"저장 완료: {NAME}  총 {time.time()-t0:.0f}s", flush=True)
