"""
per-row 표준화 후 247종 통계 피처 + LGB/HGB blend — 스케일-불변 변형.
ts_kitchen_train.py의 스케일-불변 버전: 진폭 정보를 제거하고 형태·동역학만 학습.
"""
import os
os.environ["OMP_NUM_THREADS"]    = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"]    = "4"
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import time

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data")
NAME  = "ts_kitchen_std"
np.random.seed(42)

X     = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float64)
y     = np.load(os.path.join(DATA, "y_train.npy")).astype(np.int64)
folds = np.load(os.path.join(DATA, "folds.npy")).astype(np.int64)
Xt    = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float64)
N, L  = X.shape
NT    = Xt.shape[0]


def _row_std(A):
    """per-row 표준화 — 진폭 제거, 형태 보존."""
    return (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)


X  = _row_std(X)
Xt = _row_std(Xt)
print("loaded+row-standardized", X.shape, Xt.shape)

QUANTS    = [0.01, 0.02, 0.05, 0.1, 0.16, 0.25, 0.5, 0.75, 0.84, 0.9, 0.95, 0.98, 0.99]
ACF_LAGS  = list(range(1, 49))
ABS_ACF_LAGS = list(range(1, 13))
AR_ORDER  = 24
WELCH_BANDS = 64


def acf(x, lags, xm=None, xv=None):
    n = len(x)
    if xm is None: xm = x.mean()
    xc = x - xm
    if xv is None: xv = np.dot(xc, xc)
    if xv <= 0: return np.zeros(len(lags))
    out = np.empty(len(lags))
    for i, k in enumerate(lags):
        out[i] = np.dot(xc[:-k], xc[k:]) / xv
    return out


def yule_walker(x, order):
    n = len(x)
    xc = x - x.mean()
    r = np.empty(order + 1)
    denom = np.dot(xc, xc)
    if denom <= 0:
        return np.zeros(order)
    for k in range(order + 1):
        r[k] = np.dot(xc[:n - k], xc[k:]) / denom
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    if e <= 0:
        return np.zeros(order)
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / e if e != 0 else 0.0
        new_a = a.copy()
        for j in range(1, i):
            new_a[j] = a[j] + k * a[i - j]
        new_a[i] = k
        a = new_a
        e *= (1 - k * k)
        if e <= 0:
            e = 1e-12
    return -a[1:order + 1]


def hurst_rs(x):
    n = len(x)
    ns = [16, 32, 64, 128, 256, 512]
    rs_vals, log_ns = [], []
    for w in ns:
        if w >= n:
            continue
        k = n // w
        rss = []
        for i in range(k):
            seg = x[i * w:(i + 1) * w]
            m = seg.mean()
            z = np.cumsum(seg - m)
            R = z.max() - z.min()
            S = seg.std()
            if S > 0:
                rss.append(R / S)
        if rss:
            rs_vals.append(np.log(np.mean(rss)))
            log_ns.append(np.log(w))
    if len(rs_vals) < 2:
        return 0.5
    A = np.vstack([log_ns, np.ones(len(log_ns))]).T
    return np.linalg.lstsq(A, rs_vals, rcond=None)[0][0]


def welch_bands_setup(nperseg, nbands):
    nfreq = nperseg // 2 + 1
    edges = np.unique(np.round(np.logspace(np.log10(1), np.log10(nfreq - 1), nbands + 1)).astype(int))
    return edges, nfreq


WIN_PSD  = 512
PSD_EDGES, NFREQ = welch_bands_setup(WIN_PSD, WELCH_BANDS)
NB = len(PSD_EDGES) - 1
WIN_PSD2 = 1024
PSD_EDGES2, NFREQ2 = welch_bands_setup(WIN_PSD2, 40)
NB2 = len(PSD_EDGES2) - 1


def pacf_levinson(x, order):
    n = len(x)
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom <= 0:
        return np.zeros(order)
    r = np.array([np.dot(xc[:n - k], xc[k:]) / denom for k in range(order + 1)])
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    pac = np.zeros(order)
    if e <= 0:
        return pac
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / e if e != 0 else 0.0
        pac[i - 1] = -k
        new_a = a.copy()
        for j in range(1, i):
            new_a[j] = a[j] + k * a[i - j]
        new_a[i] = k
        a = new_a
        e *= (1 - k * k)
        if e <= 0:
            e = 1e-12
    return pac


def higuchi_fd(x, kmax=8):
    n = len(x)
    lk, lnk = [], []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            lmk = np.sum(np.abs(np.diff(x[idx]))) * (n - 1) / (((len(idx) - 1)) * k)
            lm.append(lmk)
        if lm:
            lk.append(np.log(np.mean(lm) + 1e-12))
            lnk.append(np.log(1.0 / k))
    if len(lk) < 2:
        return 1.5
    A = np.vstack([lnk, np.ones(len(lnk))]).T
    return np.linalg.lstsq(A, lk, rcond=None)[0][0]


def featurize(x):
    feats = []
    xm = x.mean()
    xs = x.std()
    xc = x - xm
    xv = np.dot(xc, xc)
    feats += [xm, xs, skew(x), kurtosis(x)]
    qs = np.quantile(x, QUANTS)
    feats += list(qs)
    feats += [qs[-1] - qs[0], qs[10] - qs[2]]
    a = acf(x, ACF_LAGS, xm, xv)
    feats += list(a)
    feats += list(yule_walker(x, AR_ORDER))
    feats += list(pacf_levinson(x, 16))
    ax = np.abs(x - xm)
    feats += list(acf(ax, ABS_ACF_LAGS))
    d1 = np.diff(x)
    d2 = np.diff(d1)
    feats += [d1.std(), np.mean(np.abs(d1)), skew(d1) if d1.std() > 0 else 0.0,
              np.mean(d1[:-1] * d1[1:] < 0),
              np.mean((d1[:-1] * d1[1:]) < 0)]
    feats += [d2.std(), np.mean(np.abs(d2))]
    feats += [np.mean((xc[:-1] * xc[1:]) < 0)]
    feats += list(acf(d1, [1, 2, 3, 4, 6, 8]))
    c  = np.cumsum(np.insert(x, 0, 0))
    c2 = np.cumsum(np.insert(x * x, 0, 0))
    for win in (32, 64, 128, 256):
        s   = (c[win:] - c[:-win]) / win
        s2  = (c2[win:] - c2[:-win]) / win
        var = np.maximum(s2 - s * s, 0)
        rstd = np.sqrt(var)
        feats += [rstd.mean(), rstd.std(), rstd.min(), rstd.max()]
        feats += [s.std()]
    feats += [higuchi_fd(x, 10)]
    f, pxx = signal.welch(x, nperseg=WIN_PSD, noverlap=WIN_PSD // 2, detrend='constant')
    pxx = np.maximum(pxx, 1e-12)
    bands = np.empty(NB)
    for i in range(NB):
        lo, hi = PSD_EDGES[i], PSD_EDGES[i + 1]
        bands[i] = np.log(pxx[lo:hi].mean() + 1e-12)
    feats += list(bands)
    psum = pxx.sum()
    pn   = pxx / psum
    spec_ent  = -np.sum(pn * np.log(pn + 1e-12)) / np.log(len(pn))
    spec_cent = np.sum(f * pn)
    cum     = np.cumsum(pn)
    rolloff = f[np.searchsorted(cum, 0.85)] if cum[-1] >= 0.85 else f[-1]
    gmean   = np.exp(np.mean(np.log(pxx + 1e-12)))
    flatness = gmean / (pxx.mean() + 1e-12)
    feats += [spec_ent, spec_cent, rolloff, flatness]
    third = len(pxx) // 3
    lo_p  = pxx[:third].sum(); mid_p = pxx[third:2 * third].sum(); hi_p = pxx[2 * third:].sum()
    feats += [lo_p / psum, mid_p / psum, hi_p / psum, np.log((hi_p + 1e-12) / (lo_p + 1e-12))]
    feats += [f[np.argmax(pxx)]]
    f2, pxx2 = signal.welch(x, nperseg=WIN_PSD2, noverlap=WIN_PSD2 // 2, detrend='constant')
    pxx2 = np.maximum(pxx2, 1e-12)
    bands2 = np.empty(NB2)
    for i in range(NB2):
        lo, hi = PSD_EDGES2[i], PSD_EDGES2[i + 1]
        bands2[i] = np.log(pxx2[lo:hi].mean() + 1e-12)
    feats += list(bands2)
    above = (x > xm).astype(int)
    def longest_run(b):
        mx = cur = 0
        for v in b:
            if v:
                cur += 1
                if cur > mx: mx = cur
            else:
                cur = 0
        return mx
    feats += [longest_run(above) / L, longest_run(1 - above) / L]
    if xs > 0:
        z = xc / xs
        feats += [np.mean(np.abs(z) > 2), np.mean(np.abs(z) > 3)]
    else:
        feats += [0.0, 0.0]
    feats += [hurst_rs(x)]
    return feats


t0 = time.time()
allX = np.vstack([X, Xt])
feat_list = []
for i in range(allX.shape[0]):
    feat_list.append(featurize(allX[i]))
    if (i + 1) % 2000 == 0:
        print("featurized", i + 1, "%.1fs" % (time.time() - t0))
F = np.array(feat_list, dtype=np.float64)
F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
Ftr = F[:N].astype(np.float32)
Fte = F[N:].astype(np.float32)
print("feature matrix", Ftr.shape, "time %.1fs" % (time.time() - t0))

import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

NCLASS  = 9
Ftr64   = Ftr.astype(np.float64)
Fte64   = Fte.astype(np.float64)

oof_lgb  = np.zeros((N, NCLASS), np.float64)
test_lgb = np.zeros((NT, NCLASS), np.float64)
oof_hgb  = np.zeros((N, NCLASS), np.float64)
test_hgb = np.zeros((NT, NCLASS), np.float64)

lgb_params = dict(
    objective="multiclass", num_class=NCLASS, learning_rate=0.03,
    num_leaves=63, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=25, lambda_l1=0.0, lambda_l2=1.0,
    n_jobs=4, verbosity=-1, max_depth=-1,
)

for f in range(5):
    tr = folds != f
    va = folds == f
    Xtr_all, ytr_all = Ftr64[tr], y[tr]
    Xva = Ftr64[va]
    rng = np.random.RandomState(100 + f)
    idx = np.arange(Xtr_all.shape[0])
    rng.shuffle(idx)
    ncar = int(0.12 * len(idx))
    car  = idx[:ncar]
    fit  = idx[ncar:]
    dtr  = lgb.Dataset(Xtr_all[fit], label=ytr_all[fit])
    dval = lgb.Dataset(Xtr_all[car], label=ytr_all[car])
    booster = lgb.train(
        lgb_params, dtr, num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[va]  = booster.predict(Xva)
    test_lgb    += booster.predict(Fte64) / 5.0

    hgb = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, max_depth=None, min_samples_leaf=25,
        validation_fraction=0.12, early_stopping=True, n_iter_no_change=30,
        random_state=200 + f,
    )
    hgb.fit(Xtr_all, ytr_all)
    oof_hgb[va]  = hgb.predict_proba(Xva)
    test_hgb    += hgb.predict_proba(Fte64) / 5.0

    acc_l = accuracy_score(y[va], oof_lgb[va].argmax(1))
    acc_h = accuracy_score(y[va], oof_hgb[va].argmax(1))
    print(f"fold {f}: lgb {acc_l:.4f}  hgb {acc_h:.4f}  (lgb best_iter {booster.best_iteration})")

acc_lgb = accuracy_score(y, oof_lgb.argmax(1))
acc_hgb = accuracy_score(y, oof_hgb.argmax(1))
print(f"\nOOF lgb {acc_lgb:.4f}  hgb {acc_hgb:.4f}")

# 나머지 4-fold OOF 기준으로 fold별 blend weight 선택 — 선택편향 방지
WGRID = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]


def variant_oof(w):
    return w * oof_lgb + (1 - w) * oof_hgb


oof_final    = np.zeros_like(oof_lgb)
test_w_accum = 0.0
chosen_ws    = []
for f in range(5):
    sel     = folds != f
    apply_m = folds == f
    best_w, best_a = None, -1
    for w in WGRID:
        bl = variant_oof(w)
        a  = accuracy_score(y[sel], bl[sel].argmax(1))
        if a > best_a:
            best_a, best_w = a, w
    oof_final[apply_m] = variant_oof(best_w)[apply_m]
    test_w_accum      += best_w
    chosen_ws.append(best_w)

mean_w     = test_w_accum / 5.0
test_final = mean_w * test_lgb + (1 - mean_w) * test_hgb
acc_final  = accuracy_score(y, oof_final.argmax(1))
oof_final  = oof_final / oof_final.sum(1, keepdims=True)
test_final = test_final / test_final.sum(1, keepdims=True)
print(f"per-fold chosen w(lgb)={chosen_ws}  mean_w={mean_w:.3f}")
print(f"OOF acc {acc_final:.4f}")

per_fold = [accuracy_score(y[folds == f], oof_final[folds == f].argmax(1)) for f in range(5)]
print("per-fold:", [f"{a:.4f}" for a in per_fold])

os.makedirs(os.path.join(BASE, "oof"),       exist_ok=True)
os.makedirs(os.path.join(BASE, "test_pred"), exist_ok=True)
np.save(os.path.join(BASE, "oof",       NAME + ".npy"), oof_final.astype(np.float32))
np.save(os.path.join(BASE, "test_pred", NAME + ".npy"), test_final.astype(np.float32))
print("저장 완료:", NAME, "  총 %.1fs" % (time.time() - t0))
