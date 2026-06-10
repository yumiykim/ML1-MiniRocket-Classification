"""
round6_pipeline.py — 시계열 10모델 스택 앙상블 (6/6 제출, 리더보드 0.8713)
================================================================
final_pipeline.py 비교 버전: 7모델 → 10모델 (스케일-불변 입력 변형 3종 추가).
models/_train.py 를 호출해 OOF/test_pred 캐시를 채운 뒤 Nested CV logstack → EM 라벨-shift 보정.

실행:
  python pipelines/round6_pipeline.py            # oof/ 캐시 재사용, 앙상블만 재실행
  python pipelines/round6_pipeline.py --rebuild  # 모든 베이스 모델 재학습
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
OOF_DIR    = os.path.join(ROOT, "oof")
TEST_DIR   = os.path.join(ROOT, "test_pred")
MODELS_DIR = os.path.join(ROOT, "models")
SEED = 42
NC   = 9

STRONG = ["minirocket", "ts_kitchen", "spectral", "lgbm_pca", "lightgbm",
          "ar_likelihood", "cnn1d_v2", "minirocket_std", "ts_kitchen_std",
          "cnn1d_std"]


# ── 1. CSV → npy + 고정 folds ────────────────────────────────
def build_arrays():
    need = not all(os.path.exists(os.path.join(DATA_DIR, f))
                   for f in ("X_train.npy", "y_train.npy", "X_test.npy", "folds.npy"))
    if not need:
        return
    print("[build] CSV -> npy 변환 ...")
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y  = tr["class_idx"].to_numpy().astype(np.int64)
    X  = tr.drop(columns=["class_idx"]).to_numpy().astype(np.float32)
    Xt = te.to_numpy().astype(np.float32)
    assert X.shape == (10000, 2048) and Xt.shape == (3590, 2048)
    folds = np.full(len(y), -1, dtype=np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for f, (_, va) in enumerate(skf.split(X, y)):
        folds[va] = f
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  Xt)
    np.save(os.path.join(DATA_DIR, "folds.npy"),   folds)


# ── 2. 베이스 모델 OOF/test_pred (캐시 재사용 또는 재학습) ──
def ensure_base(rebuild=False):
    for m in STRONG:
        op = os.path.join(OOF_DIR,  m + ".npy")
        tp = os.path.join(TEST_DIR, m + ".npy")
        if (not rebuild) and os.path.exists(op) and os.path.exists(tp):
            continue
        script = os.path.join(MODELS_DIR, m + "_train.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"base train 스크립트 없음: {script}")
        print(f"[base] 학습 실행: {m}_train.py")
        env = dict(os.environ, OMP_NUM_THREADS="4")
        subprocess.run([sys.executable, script], check=True, env=env, cwd=ROOT)


def load_strong():
    y     = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    folds = np.load(os.path.join(DATA_DIR, "folds.npy"))
    OOF, TST = {}, {}
    for m in STRONG:
        o = np.load(os.path.join(OOF_DIR,  m + ".npy")).astype(np.float64)
        t = np.load(os.path.join(TEST_DIR, m + ".npy")).astype(np.float64)
        assert o.shape == (len(y), NC) and t.shape == (3590, NC), (m, o.shape, t.shape)
        o = np.clip(o, 1e-9, None); o /= o.sum(1, keepdims=True)
        t = np.clip(t, 1e-9, None); t /= t.sum(1, keepdims=True)
        OOF[m], TST[m] = o, t
    return y, folds, OOF, TST


# ── 3. 앙상블: Nested CV logstack + EM 라벨-shift 보정 ──────
def clip_log(p):
    p = np.clip(p, 1e-6, 1); p /= p.sum(1, keepdims=True); return np.log(p)


def em_estimate_pi(P, pi_train, n_iter=2000, tol=1e-9):
    """EM 고정점 반복으로 test class prior 추정 (Saerens et al. 2002)."""
    P    = np.clip(P, 1e-12, None); P = P / P.sum(1, keepdims=True)
    base = P / pi_train[None, :]
    pi   = pi_train.copy()
    for _ in range(n_iter):
        num = base * pi[None, :]
        num = num / num.sum(1, keepdims=True)
        new = num.mean(0); new = new / new.sum()
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi


def prior_correct(P, pi_test, pi_train):
    """pi_test/pi_train으로 행별 재가중 — test에만 적용."""
    Q = P * (pi_test / pi_train)[None, :]
    return Q / Q.sum(1, keepdims=True)


def logstack_fit(OOF, TST, y, folds, train_mask, Cgrid=(0.02, 0.05, 0.1, 0.3)):
    """train_mask 행에서 내부 fold로 C 선택 후 적합. 반환: (oof_pred[all], test_pred, C)."""
    Xall   = np.hstack([clip_log(OOF[n]) for n in STRONG])
    Xte    = np.hstack([clip_log(TST[n]) for n in STRONG])
    ifolds = np.unique(folds[train_mask])
    bestC, bestA = None, -1
    for C in Cgrid:
        ac = []
        for vf in ifolds:
            st  = train_mask & (folds == vf)
            s2  = train_mask & (folds != vf)
            clf = LogisticRegression(C=C, max_iter=300, solver="lbfgs", n_jobs=1)
            clf.fit(Xall[s2], y[s2])
            ac.append((clf.predict_proba(Xall[st]).argmax(1) == y[st]).mean())
        m = np.mean(ac)
        if m > bestA:
            bestA, bestC = m, C
    clf = LogisticRegression(C=bestC, max_iter=300, solver="lbfgs", n_jobs=1)
    clf.fit(Xall[train_mask], y[train_mask])
    return clf.predict_proba(Xall), clf.predict_proba(Xte), bestC


def nested_logstack(OOF, TST, y, folds):
    accs = []
    for of in range(5):
        ot = folds == of
        po, _, _ = logstack_fit(OOF, TST, y, folds, ~ot)
        accs.append((po[ot].argmax(1) == y[ot]).mean())
    return float(np.mean(accs)), [round(a, 4) for a in accs]


def main(rebuild=False):
    build_arrays()
    ensure_base(rebuild=rebuild)
    y, folds, OOF, TST = load_strong()

    print("=== base 모델 full-OOF acc ===")
    for m in STRONG:
        print(f"  {m:20s} {accuracy_score(y, OOF[m].argmax(1)):.4f}")

    nested, fold_accs = nested_logstack(OOF, TST, y, folds)
    print(f"\n[NESTED CV] logstack_strong = {nested:.4f}  folds={fold_accs}")

    full = np.ones(len(y), bool)
    po, pt, C = logstack_fit(OOF, TST, y, folds, full)
    po = (po / po.sum(1, keepdims=True)).astype(np.float32)
    pt = (pt / pt.sum(1, keepdims=True)).astype(np.float32)
    print(f"refit C={C}  full-OOF acc(낙관)={accuracy_score(y, po.argmax(1)):.4f}")

    np.save(os.path.join(OOF_DIR,  "ensemble_round6.npy"), po)
    np.save(os.path.join(TEST_DIR, "ensemble_round6.npy"), pt)

    pi_train = np.bincount(y, minlength=NC).astype(np.float64) / len(y)
    pi_test  = em_estimate_pi(pt.astype(np.float64), pi_train)
    pt_corr  = prior_correct(pt.astype(np.float64), pi_test, pi_train).astype(np.float32)
    np.save(os.path.join(TEST_DIR, "ensemble_round6_final.npy"), pt_corr)
    changed = (pt_corr.argmax(1) != pt.argmax(1)).mean()
    print(f"[EM] pi_test={np.round(pi_test, 3).tolist()}  argmax변경 {changed*100:.1f}%")

    pred     = pt_corr.argmax(1).astype(int)
    sub      = pd.DataFrame({"class_idx": pred})
    sub_path = os.path.join(ROOT, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\n[저장] {sub_path}  rows={len(sub)}")
    print("[class dist]", np.bincount(pred, minlength=NC).tolist())
    print(f"[nested CV] {nested:.4f}  +EM 보정 (합성 shift 검증 +0.24pp)")


if __name__ == "__main__":
    main(rebuild=("--rebuild" in sys.argv))
