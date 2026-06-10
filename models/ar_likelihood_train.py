"""
클래스 조건부 AR(p) 로그우도 (p=5,10,20, Yule-Walker) → 45차원 피처 → HGB. OOF 0.696.
분류기 변형 (Bayes/LR/HGB)은 fold-honest 방식으로 선택 (선택편향 제거).
"""
import os
os.environ["OMP_NUM_THREADS"]    = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"]    = "4"

import numpy as np
import time
from numpy.linalg import solve, LinAlgError

t0 = time.time()
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(BASE, "data")
NAME    = "ar_likelihood"

X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float64)
y_train = np.load(os.path.join(DATA, "y_train.npy")).astype(np.int64)
X_test  = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float64)
folds   = np.load(os.path.join(DATA, "folds.npy")).astype(np.int64)

N, T    = X_train.shape
Ntest   = X_test.shape[0]
NCLASS  = 9
PS      = [5, 10, 20]
MAXLAG  = max(PS)


def pooled_autocov(rows, maxlag):
    m, Tloc = rows.shape
    centered = rows - rows.mean(axis=1, keepdims=True)
    gamma = np.zeros(maxlag + 1)
    for lag in range(maxlag + 1):
        prod = centered * centered if lag == 0 else centered[:, lag:] * centered[:, :-lag]
        gamma[lag] = prod.sum() / (m * Tloc)
    return gamma


def yule_walker(gamma, p):
    R = np.array([[gamma[abs(i - j)] for j in range(p)] for i in range(p)])
    r = gamma[1:p + 1]
    R += np.eye(p) * (1e-8 * gamma[0] + 1e-12)
    try:
        phi = solve(R, r)
    except LinAlgError:
        phi = np.zeros(p)
    sigma2 = max(gamma[0] - phi @ r, 1e-10)
    return phi, sigma2


def ar_loglik_all_rows(X, mu_k, phi, sigma2):
    n, Tloc = X.shape; p = len(phi)
    dev = X - mu_k
    pred = sum(phi[j] * dev[:, p - 1 - j: Tloc - 1 - j] for j in range(p))
    resid = dev[:, p:] - pred
    npts = Tloc - p
    ss = (resid * resid).sum(axis=1)
    ll = -0.5 * npts * np.log(2 * np.pi * sigma2) - 0.5 * ss / sigma2
    return ll / npts


def fit_fold_params(Xtr, ytr):
    params = {}
    priors = np.zeros(NCLASS)
    for k in range(NCLASS):
        rows = Xtr[ytr == k]
        priors[k] = len(rows)
        gamma = pooled_autocov(rows, MAXLAG)
        per_p = {p: yule_walker(gamma, p) for p in PS}
        rmean = rows.mean(axis=1); rstd = rows.std(axis=1)
        params[k] = dict(mu_k=rows.mean(), per_p=per_p,
                         rmean_mu=rmean.mean(), rmean_sd=rmean.std() + 1e-9,
                         rstd_mu=rstd.mean(),  rstd_sd=rstd.std()  + 1e-9)
    return params, priors / priors.sum()


def build_features(X, params):
    n = X.shape[0]
    rmean = X.mean(axis=1); rstd = X.std(axis=1)
    feats = np.zeros((n, NCLASS * len(PS)))
    extra = np.zeros((n, NCLASS * 2))
    col = 0
    for k in range(NCLASS):
        pk = params[k]
        for p in PS:
            phi, sigma2 = pk["per_p"][p]
            feats[:, col] = ar_loglik_all_rows(X, pk["mu_k"], phi, sigma2)
            col += 1
        extra[:, k]         = (-0.5 * np.log(2*np.pi*pk["rmean_sd"]**2)
                               - 0.5 * ((rmean - pk["rmean_mu"]) / pk["rmean_sd"])**2)
        extra[:, NCLASS + k] = (-0.5 * np.log(2*np.pi*pk["rstd_sd"]**2)
                                - 0.5 * ((rstd  - pk["rstd_mu"])  / pk["rstd_sd"])**2)
    return feats, extra


feat_train       = np.zeros((N,     27)); extra_train       = np.zeros((N,     18))
feat_test_folds  = np.zeros((5, Ntest, 27)); extra_test_folds = np.zeros((5, Ntest, 18))
oof_bayes        = np.zeros((N, NCLASS));    test_bayes_folds = np.zeros((5, Ntest, NCLASS))

for f in range(5):
    tr = folds != f; va = folds == f
    params, priors = fit_fold_params(X_train[tr], y_train[tr])
    fv, ev = build_features(X_train[va], params)
    feat_train[va] = fv; extra_train[va] = ev
    ft, et = build_features(X_test, params)
    feat_test_folds[f] = ft; extra_test_folds[f] = et
    pidx = PS.index(10); logp = np.log(priors + 1e-12)
    score_v = np.array([T * fv[:, k*3+pidx] + ev[:, k] + ev[:, NCLASS+k] + logp[k]
                        for k in range(NCLASS)]).T
    score_t = np.array([T * ft[:, k*3+pidx] + et[:, k] + et[:, NCLASS+k] + logp[k]
                        for k in range(NCLASS)]).T
    for s in (score_v, score_t): s -= s.max(axis=1, keepdims=True)
    pv = np.exp(score_v); pv /= pv.sum(1, keepdims=True)
    pt = np.exp(score_t); pt /= pt.sum(1, keepdims=True)
    oof_bayes[va] = pv; test_bayes_folds[f] = pt
    print(f"[fold {f}] direct-Bayes val={( pv.argmax(1)==y_train[va]).mean():.4f}"
          f"  ({time.time()-t0:.0f}s)")

print(f"Direct-Bayes OOF = {(oof_bayes.argmax(1)==y_train).mean():.4f}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

allfeat_train = np.hstack([feat_train, extra_train])
oof_lr = np.zeros((N, NCLASS)); test_lr_folds = np.zeros((5, Ntest, NCLASS))
for f in range(5):
    tr = folds != f; va = folds == f
    sc = StandardScaler().fit(allfeat_train[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    clf.fit(sc.transform(allfeat_train[tr]), y_train[tr])
    oof_lr[va] = clf.predict_proba(sc.transform(allfeat_train[va]))
    test_lr_folds[f] = clf.predict_proba(
        sc.transform(np.hstack([feat_test_folds[f], extra_test_folds[f]])))

oof_hgb = np.zeros((N, NCLASS)); test_hgb_folds = np.zeros((5, Ntest, NCLASS))
for f in range(5):
    tr = folds != f; va = folds == f
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          max_leaf_nodes=31, random_state=0)
    clf.fit(allfeat_train[tr], y_train[tr])
    oof_hgb[va] = clf.predict_proba(allfeat_train[va])
    test_hgb_folds[f] = clf.predict_proba(
        np.hstack([feat_test_folds[f], extra_test_folds[f]]))

# 나머지 4-fold OOF 기준으로 fold별 variant 선택 — 선택편향 방지
variants = {"bayes": (oof_bayes, test_bayes_folds.mean(0)),
            "lr":    (oof_lr,    test_lr_folds.mean(0)),
            "hgb":   (oof_hgb,   test_hgb_folds.mean(0))}
oof_best = np.zeros_like(oof_bayes); chosen = []
for f in range(5):
    sel = folds != f; apply_m = folds == f
    best_k, best_a = None, -1
    for k, (vo, _) in variants.items():
        a = (vo[sel].argmax(1) == y_train[sel]).mean()
        if a > best_a: best_a, best_k = a, k
    oof_best[apply_m] = variants[best_k][0][apply_m]; chosen.append(best_k)

from collections import Counter
test_best = np.mean([variants[k][1] for k in chosen], axis=0)
print(f"FOLD-HONEST OOF = {(oof_best.argmax(1)==y_train).mean():.4f}  chosen={Counter(chosen)}")

os.makedirs(os.path.join(BASE, "oof"),       exist_ok=True)
os.makedirs(os.path.join(BASE, "test_pred"), exist_ok=True)
np.save(os.path.join(BASE, "oof",       NAME + ".npy"), oof_best.astype(np.float32))
np.save(os.path.join(BASE, "test_pred", NAME + ".npy"), test_best.astype(np.float32))
print(f"Done {time.time()-t0:.0f}s")
