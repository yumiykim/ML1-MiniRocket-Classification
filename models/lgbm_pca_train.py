"""
PCA(256) + top-분산 256 피처 연결(512차원) → LightGBM. fold별 fit으로 누설 방지.
raw 2048과 OOF 비교 후 더 높은 구성을 저장. OOF ~0.711.
"""
import os
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"]      = "4"

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

NAME      = "lgbm_pca"
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = os.path.join(BASE, "data")
N_CLASSES = 9
PCA_K     = 256
TOPVAR_K  = 256

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
    num_leaves=31,
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


def make_features_fold(Xtr, Xva, Xte_full, params_feat):
    """fold 학습 행에만 fit: PCA + top-분산 피처 블록 연결."""
    pca_k    = params_feat["pca_k"]
    topvar_k = params_feat["topvar_k"]

    blocks_tr, blocks_va, blocks_te = [], [], []

    if pca_k > 0:
        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xva_s = sc.transform(Xva)
        Xte_s = sc.transform(Xte_full)
        pca   = PCA(n_components=pca_k, random_state=42)
        blocks_tr.append(pca.fit_transform(Xtr_s).astype(np.float32))
        blocks_va.append(pca.transform(Xva_s).astype(np.float32))
        blocks_te.append(pca.transform(Xte_s).astype(np.float32))

    if topvar_k > 0:
        var     = Xtr.var(axis=0)
        top_idx = np.argsort(var)[::-1][:topvar_k]
        blocks_tr.append(Xtr[:, top_idx])
        blocks_va.append(Xva[:, top_idx])
        blocks_te.append(Xte_full[:, top_idx])

    Ftr = np.concatenate(blocks_tr, axis=1)
    Fva = np.concatenate(blocks_va, axis=1)
    Fte = np.concatenate(blocks_te, axis=1)
    return Ftr, Fva, Fte


def run_cv(params, feat_mode):
    """5-fold CV. feat_mode=None → raw 2048, 아니면 dict(pca_k, topvar_k)."""
    oof       = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    test_pred = np.zeros((n_test,  N_CLASSES), dtype=np.float64)
    best_iters = []
    for f in fold_ids:
        tr = folds != f
        va = folds == f
        if feat_mode is None:
            Ftr, Fva, Fte = X[tr], X[va], Xte
        else:
            Ftr, Fva, Fte = make_features_fold(X[tr], X[va], Xte, feat_mode)
        dtr = lgb.Dataset(Ftr, label=y[tr])
        dva = lgb.Dataset(Fva, label=y[va], reference=dtr)
        model = lgb.train(
            params,
            dtr,
            num_boost_round=NUM_ROUNDS,
            valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        bi = model.best_iteration or NUM_ROUNDS
        best_iters.append(bi)
        oof[va]    = model.predict(Fva, num_iteration=bi)
        test_pred += model.predict(Fte, num_iteration=bi) / len(fold_ids)
        acc_f = accuracy_score(y[va], oof[va].argmax(1))
        print(f"  fold {f}: best_iter={bi:4d}  acc={acc_f:.4f}")
    return oof, test_pred, best_iters


def main():
    print(f"[{NAME}] train={n_train} test={n_test} classes={N_CLASSES}")

    candidates = [
        ("pca256+topvar256", dict(pca_k=PCA_K, topvar_k=TOPVAR_K)),
        ("raw2048",          None),
    ]

    best = None
    for tag, feat_mode in candidates:
        print(f"[cfg] {tag}")
        oof, test_pred, best_iters = run_cv(PARAMS, feat_mode)
        acc = accuracy_score(y, oof.argmax(1))
        print(f"[cfg] {tag} -> mean OOF acc={acc:.4f}")
        if best is None or acc > best["acc"]:
            best = dict(acc=acc, tag=tag, oof=oof, test_pred=test_pred,
                        best_iters=best_iters)

    print("\n==== BEST CONFIG ====")
    print("cfg:", best["tag"])
    fold_accs = []
    for f in fold_ids:
        va = folds == f
        a  = accuracy_score(y[va], best["oof"][va].argmax(1))
        fold_accs.append(round(float(a), 4))
    print("per-fold acc:", fold_accs)
    print(f"mean OOF acc: {best['acc']:.4f}")
    print("best_iters:",   best["best_iters"])

    oof       = best["oof"].astype(np.float32)
    test_pred = best["test_pred"].astype(np.float32)
    assert np.allclose(oof.sum(1), 1.0, atol=1e-3), oof.sum(1)[:5]
    assert np.allclose(test_pred.sum(1), 1.0, atol=1e-3), test_pred.sum(1)[:5]
    assert oof.shape == (n_train, N_CLASSES)
    assert test_pred.shape == (n_test, N_CLASSES)

    os.makedirs(os.path.join(BASE, "oof"),       exist_ok=True)
    os.makedirs(os.path.join(BASE, "test_pred"), exist_ok=True)
    np.save(os.path.join(BASE, "oof",       f"{NAME}.npy"), oof)
    np.save(os.path.join(BASE, "test_pred", f"{NAME}.npy"), test_pred)
    print(f"저장 완료: {NAME}")


if __name__ == "__main__":
    main()
