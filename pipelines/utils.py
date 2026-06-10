"""앙상블 공통 유틸: clip_log, EM 라벨-shift 보정 (Saerens 2002), Nested CV logstack."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def clip_log(p):
    p = np.clip(p, 1e-6, 1)
    p = p / p.sum(1, keepdims=True)
    return np.log(p)


def clip_norm(p):
    p = np.clip(p, 1e-9, None)
    return p / p.sum(1, keepdims=True)


def em_estimate_pi(P, pi_train, n_iter=2000, tol=1e-9):
    """EM 고정점 반복으로 test class prior 추정 (Saerens et al. 2002)."""
    P = np.clip(P, 1e-12, None)
    P = P / P.sum(1, keepdims=True)
    base = P / pi_train[None, :]
    pi = pi_train.copy()
    for _ in range(n_iter):
        num = base * pi[None, :]
        num = num / num.sum(1, keepdims=True)
        new_pi = num.mean(0)
        new_pi /= new_pi.sum()
        if np.abs(new_pi - pi).max() < tol:
            return new_pi
        pi = new_pi
    return pi


def prior_correct(P, pi_test, pi_train):
    """pi_test/pi_train으로 행별 재가중 — test에만 적용, train OOF 제외."""
    Q = P * (pi_test / pi_train)[None, :]
    return Q / Q.sum(1, keepdims=True)


def logstack_fit(OOF, TST, strong, y, folds, train_mask,
                 Cgrid=(0.01, 0.02, 0.05, 0.1), solver="saga", seed=42):
    """내부 fold CV로 C 선택 후 train_mask 행에서 LogisticRegression 적합. 반환: (oof_pred, test_pred, best_C)."""
    Xall = np.hstack([clip_log(OOF[m]) for m in strong])
    Xte  = np.hstack([clip_log(TST[m]) for m in strong])

    ifolds = np.unique(folds[train_mask])
    bestC, bestA = None, -1.0
    for C in Cgrid:
        ac = []
        for vf in ifolds:
            st = train_mask & (folds == vf)
            s2 = train_mask & (folds != vf)
            clf = LogisticRegression(C=C, max_iter=5000, solver=solver,
                                     random_state=seed, n_jobs=1)
            clf.fit(Xall[s2], y[s2])
            ac.append((clf.predict_proba(Xall[st]).argmax(1) == y[st]).mean())
        m = float(np.mean(ac))
        if m > bestA:
            bestA, bestC = m, C

    clf = LogisticRegression(C=bestC, max_iter=5000, solver=solver,
                             random_state=seed, n_jobs=1)
    clf.fit(Xall[train_mask], y[train_mask])
    return clf.predict_proba(Xall), clf.predict_proba(Xte), bestC


def nested_logstack(OOF, TST, strong, y, folds, **kwargs):
    """정직한 Nested CV 추정 — outer fold 라벨이 C 선택에 노출되지 않음."""
    accs = []
    for of in range(5):
        ot = folds == of
        po, _, _ = logstack_fit(OOF, TST, strong, y, folds, ~ot, **kwargs)
        accs.append(float((po[ot].argmax(1) == y[ot]).mean()))
    return float(np.mean(accs)), accs
