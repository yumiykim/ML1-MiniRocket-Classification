"""
ISE4045 9-클래스 분류 엔드투엔드 파이프라인 — 리더보드 1위, accuracy 0.8713.
7개 시계열 베이스 모델 (MiniRocket / TS-Kitchen / CNN1D / Spectral / AR / LightGBM / LGBM-PCA) → Nested CV logstack → EM 라벨-shift 보정.
실행: python pipelines/final_pipeline.py [--rebuild]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ── Config ───────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OOF_DIR  = os.path.join(ROOT, "oof")
TEST_DIR = os.path.join(ROOT, "test_pred")
MDL_DIR  = os.path.join(ROOT, "models")
SEED     = 42
NC       = 9

STRONG = [
    "minirocket", "ts_kitchen", "spectral",
    "lgbm_pca", "lightgbm", "ar_likelihood", "cnn1d_v2",
]

from pipelines.utils import clip_log, clip_norm, em_estimate_pi, prior_correct
from pipelines.utils import logstack_fit, nested_logstack


def build_arrays():
    targets = ["X_train.npy", "y_train.npy", "X_test.npy", "folds.npy"]
    if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in targets):
        return
    print("[build] CSV → npy 변환 ...")
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
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), Xt)
    np.save(os.path.join(DATA_DIR, "folds.npy"), folds)


def ensure_base(rebuild=False):
    os.makedirs(OOF_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    for m in STRONG:
        op = os.path.join(OOF_DIR, m + ".npy")
        tp = os.path.join(TEST_DIR, m + ".npy")
        if (not rebuild) and os.path.exists(op) and os.path.exists(tp):
            continue
        script = os.path.join(MDL_DIR, m + "_train.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"base 학습 스크립트 없음: {script}")
        print(f"[base] 학습 실행: {m}_train.py")
        subprocess.run([sys.executable, script], check=True, cwd=MDL_DIR)


def load_strong():
    y     = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    folds = np.load(os.path.join(DATA_DIR, "folds.npy"))
    OOF, TST = {}, {}
    for m in STRONG:
        o = np.load(os.path.join(OOF_DIR,  m + ".npy")).astype(np.float64)
        t = np.load(os.path.join(TEST_DIR, m + ".npy")).astype(np.float64)
        assert o.shape == (len(y), NC) and t.shape == (3590, NC), (m, o.shape)
        OOF[m] = clip_norm(o)
        TST[m] = clip_norm(t)
    return y, folds, OOF, TST


def main(rebuild=False):
    build_arrays()
    ensure_base(rebuild=rebuild)
    y, folds, OOF, TST = load_strong()

    print("=== Base 모델 OOF 정확도 ===")
    for m in STRONG:
        print(f"  {m:16s}: {accuracy_score(y, OOF[m].argmax(1)):.4f}")

    nested, fold_accs = nested_logstack(OOF, TST, STRONG, y, folds)
    print(f"\n[NESTED CV] logstack_strong = {nested:.4f}  folds={[round(a,4) for a in fold_accs]}")

    full = np.ones(len(y), bool)
    po, pt, C = logstack_fit(OOF, TST, STRONG, y, folds, full)
    print(f"[refit] C={C}  full-OOF(낙관)={accuracy_score(y, po.argmax(1)):.4f}")

    np.save(os.path.join(OOF_DIR,  "ensemble_final.npy"), po.astype(np.float32))
    np.save(os.path.join(TEST_DIR, "ensemble_final.npy"), pt.astype(np.float32))

    pi_train = np.bincount(y, minlength=NC).astype(np.float64) / len(y)
    pi_test  = em_estimate_pi(pt.astype(np.float64), pi_train)
    pt_corr  = prior_correct(pt.astype(np.float64), pi_test, pi_train)
    changed  = (pt_corr.argmax(1) != pt.argmax(1)).mean()
    print(f"[EM 보정] pi_test={np.round(pi_test,3).tolist()}  argmax 변경 {changed*100:.1f}%")

    pred     = pt_corr.argmax(1).astype(int)
    sub_path = os.path.join(ROOT, "submission.csv")
    pd.DataFrame({"class_idx": pred}).to_csv(sub_path, index=False)
    print(f"\n[저장] {sub_path}  rows={len(pred)}")
    print(f"[test 클래스 분포] {np.bincount(pred, minlength=NC).tolist()}")
    print(f"\n[Nested CV] honest={nested:.4f}")


if __name__ == "__main__":
    main(rebuild=("--rebuild" in sys.argv))
