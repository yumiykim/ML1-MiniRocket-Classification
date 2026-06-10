"""
raw 2048차원 피처 → LightGBM — 앙상블 다양성 기여 모델. OOF ~0.70.
스케일 불변 설계로 StandardScaler 불필요.
"""
import os
os.environ["OMP_NUM_THREADS"]    = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"]    = "4"

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score

NAME      = "lightgbm"
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = os.path.join(BASE, "data")
N_CLASSES = 9

X     = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
y     = np.load(os.path.join(DATA, "y_train.npy")).astype(np.int64)
Xte   = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)
folds = np.load(os.path.join(DATA, "folds.npy")).astype(np.int64)

n_train  = X.shape[0]
n_test   = Xte.shape[0]
fold_ids = sorted(np.unique(folds).tolist())

PARAMS = dict(
    objective="multiclass",
    num_class=N_CLASSES,
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.3,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=20,
    max_depth=-1,
    metric="multi_logloss",
    verbosity=-1,
    num_threads=4,
    seed=42,
    deterministic=True,
)
NUM_ROUNDS = 2000
EARLY_STOP = 100


def run_cv(params):
    oof       = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    test_pred = np.zeros((n_test,  N_CLASSES), dtype=np.float64)
    best_iters = []
    for f in fold_ids:
        tr  = folds != f
        va  = folds == f
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        model = lgb.train(
            params, dtr,
            num_boost_round=NUM_ROUNDS,
            valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        bi = model.best_iteration or NUM_ROUNDS
        best_iters.append(bi)
        oof[va]   = model.predict(X[va], num_iteration=bi)
        test_pred += model.predict(Xte, num_iteration=bi) / len(fold_ids)
        print(f"  fold {f}: best_iter={bi:4d}  acc={accuracy_score(y[va], oof[va].argmax(1)):.4f}")
    return oof, test_pred, best_iters


def main():
    print(f"[{NAME}] train={n_train} test={n_test} classes={N_CLASSES}")

    grid = [
        dict(num_leaves=63, min_data_in_leaf=20),
        dict(num_leaves=31, min_data_in_leaf=20),
        dict(num_leaves=63, min_data_in_leaf=50),
    ]
    best = None
    for g in grid:
        p = dict(PARAMS); p.update(g)
        print(f"[cfg] {g}")
        oof, test_pred, best_iters = run_cv(p)
        acc = accuracy_score(y, oof.argmax(1))
        print(f"  → OOF acc={acc:.4f}")
        if best is None or acc > best["acc"]:
            best = dict(acc=acc, cfg=g, oof=oof, test_pred=test_pred)

    print(f"\n최적: {best['cfg']}  OOF={best['acc']:.4f}")

    oof       = best["oof"].astype(np.float32)
    test_pred = best["test_pred"].astype(np.float32)

    os.makedirs(os.path.join(BASE, "oof"),       exist_ok=True)
    os.makedirs(os.path.join(BASE, "test_pred"), exist_ok=True)
    np.save(os.path.join(BASE, "oof",       f"{NAME}.npy"), oof)
    np.save(os.path.join(BASE, "test_pred", f"{NAME}.npy"), test_pred)
    print(f"saved: {NAME}")


if __name__ == "__main__":
    main()
