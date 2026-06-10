"""
Welch PSD 피처 (log power spectrum + spectral centroid/entropy/rolloff/flatness) → LogReg/LGBM/MLP. OOF ~0.74.
단일 모델 또는 3모델 평균 중 OOF 기준 최선을 저장.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import torch
import torch.nn as nn

torch.set_num_threads(5)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data")
NAME  = "spectral"
np.random.seed(0); torch.manual_seed(0)

X     = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
y     = np.load(os.path.join(DATA, "y_train.npy")).astype(np.int64)
folds = np.load(os.path.join(DATA, "folds.npy")).astype(np.int64)
Xt    = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)
N, L  = X.shape; NT = Xt.shape[0]; NC = 9


def spectral_features(M):
    Mc = M - M.mean(axis=1, keepdims=True)
    F  = np.fft.rfft(Mc, axis=1)
    P  = F.real**2 + F.imag**2
    logP = np.log1p(P).astype(np.float32)
    w = 8; pad = w // 2
    padded = np.pad(logP, ((0,0),(pad, w-pad)), mode="edge")
    cs = np.concatenate([np.zeros((padded.shape[0],1),np.float32),
                         np.cumsum(padded, axis=1)], axis=1)
    sm  = ((cs[:, w:] - cs[:, :-w]) / w).astype(np.float32)
    sub = sm[:, ::4].astype(np.float32)
    Pn  = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    bins = np.arange(P.shape[1], dtype=np.float32)
    centroid = (Pn * bins).sum(axis=1)
    spread   = np.sqrt((Pn * (bins - centroid[:,None])**2).sum(axis=1))
    entropy  = -(Pn * np.log(Pn + 1e-12)).sum(axis=1)
    csum     = np.cumsum(Pn, axis=1)
    rolloff  = (csum < 0.85).sum(axis=1).astype(np.float32)
    gm       = np.exp(np.log(P + 1e-12).mean(axis=1))
    flatness = gm / (P.mean(axis=1) + 1e-12)
    peakfreq = np.argmax(P[:,1:], axis=1).astype(np.float32) + 1
    logpeak  = np.log1p(P[np.arange(P.shape[0]), peakfreq.astype(int)])
    half = P.shape[1] // 2
    lowE = np.log1p(P[:,1:half].sum(axis=1)); highE = np.log1p(P[:,half:].sum(axis=1))
    stats = np.stack([centroid, spread, entropy, rolloff, flatness,
                      peakfreq, logpeak, lowE, highE], axis=1).astype(np.float32)
    return np.concatenate([sub, stats], axis=1).astype(np.float32), sub


t0 = time.time()
F_all, S_all   = spectral_features(X)
Ft_all, St_all = spectral_features(Xt)
print("features built", F_all.shape, "in", round(time.time()-t0,1), "s")


def run_oof(fit_predict, feat_train, feat_test):
    oof = np.zeros((N, NC), dtype=np.float32)
    tep = np.zeros((NT, NC), dtype=np.float32)
    accs = []
    for f in range(5):
        tr = folds != f; va = folds == f
        p_va, p_te = fit_predict(feat_train[tr], y[tr], feat_train[va], feat_test)
        oof[va] = p_va; tep += p_te / 5.0
        accs.append(accuracy_score(y[va], p_va.argmax(1)))
    return oof, tep, accs, accuracy_score(y, oof.argmax(1))


def make_logreg(C):
    def fp(Xtr, ytr, Xva, Xte):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000, n_jobs=4,
                                               multi_class="multinomial"))
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xva), clf.predict_proba(Xte)
    return fp


best_lr = None
for C in [0.05, 0.2, 0.5, 1.0]:
    oof, tep, accs, m = run_oof(make_logreg(C), S_all, St_all)
    print(f"[LogReg C={C}] OOF={m:.4f}")
    if best_lr is None or m > best_lr[3]:
        best_lr = (oof, tep, accs, m, f"logreg_C{C}")


def lgbm_fp(Xtr, ytr, Xva, Xte):
    dtr = lgb.Dataset(Xtr, label=ytr)
    params = dict(objective="multiclass", num_class=NC, learning_rate=0.05,
                  num_leaves=63, min_child_samples=30, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
    bst = lgb.train(params, dtr, num_boost_round=400)
    return bst.predict(Xva), bst.predict(Xte)


oof_lg, tep_lg, accs_lg, m_lg = run_oof(lgbm_fp, S_all, St_all)
print(f"[LGBM] OOF={m_lg:.4f}")


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, NC))
    def forward(self, x): return self.net(x)


def mlp_fp(Xtr, ytr, Xva, Xte):
    sc   = StandardScaler().fit(Xtr)
    Xtr2 = sc.transform(Xtr).astype(np.float32)
    Xva2 = sc.transform(Xva).astype(np.float32)
    Xte2 = sc.transform(Xte).astype(np.float32)
    model = MLP(Xtr2.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr2); ytr_t = torch.tensor(ytr)
    for ep in range(60):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 256):
            idx = perm[i:i+256]; opt.zero_grad()
            lossf(model(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pva = torch.softmax(model(torch.tensor(Xva2)), 1).numpy()
        pte = torch.softmax(model(torch.tensor(Xte2)), 1).numpy()
    return pva, pte


oof_mlp, tep_mlp, accs_mlp, m_mlp = run_oof(mlp_fp, S_all, St_all)
print(f"[MLP] OOF={m_mlp:.4f}")

cands = [
    (best_lr[3], best_lr[0], best_lr[1], best_lr[4]),
    (m_lg, oof_lg, tep_lg, "lgbm"),
    (m_mlp, oof_mlp, tep_mlp, "mlp"),
]
cands.sort(key=lambda c: -c[0])
best = cands[0]

def norm(p): return p / p.sum(1, keepdims=True)
avg_oof = norm((best_lr[0] + oof_lg + oof_mlp) / 3.0)
avg_tep = norm((best_lr[1] + tep_lg + tep_mlp) / 3.0)
m_avg   = accuracy_score(y, avg_oof.argmax(1))

if m_avg > best[0]:
    final_oof, final_tep = avg_oof, avg_tep
    print(f"SAVING avg3 OOF={m_avg:.4f}")
else:
    final_oof, final_tep = best[1], best[2]
    print(f"SAVING {best[3]} OOF={best[0]:.4f}")

os.makedirs(os.path.join(BASE, "oof"),       exist_ok=True)
os.makedirs(os.path.join(BASE, "test_pred"), exist_ok=True)
np.save(os.path.join(BASE, "oof",       f"{NAME}.npy"), final_oof.astype(np.float32))
np.save(os.path.join(BASE, "test_pred", f"{NAME}.npy"), final_tep.astype(np.float32))
print("DONE total", round(time.time()-t0, 1), "s")
