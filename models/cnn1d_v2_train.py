"""
1D CNN, 3채널 입력 (raw / 행별 z-norm / first difference), 길이 2048 시계열. OOF ~0.82.
fold별 체크포인트 저장으로 재시작 가능, test 예측은 점진적으로 누적.
"""
import os, time, sys
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(5)
np.random.seed(42); torch.manual_seed(42)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data")
NAME  = "cnn1d_v2"
DEV   = "cpu"

X     = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
y     = np.load(os.path.join(DATA, "y_train.npy")).astype(np.int64)
folds = np.load(os.path.join(DATA, "folds.npy")).astype(np.int64)
Xt    = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)

N, L = X.shape; NT = Xt.shape[0]; NCLS = 9
AVGPOOL_PRE = int(os.environ.get("AVGPOOL_PRE", "2"))


def make_channels(arr):
    """행별 독립 변환 (raw / z-norm / diff). Cross-row 누설 없음."""
    raw   = arr
    rmean = arr.mean(1, keepdims=True)
    rstd  = arr.std(1, keepdims=True) + 1e-6
    std   = (arr - rmean) / rstd
    d1    = np.diff(arr, axis=1)
    d1    = np.concatenate([np.zeros((arr.shape[0], 1), np.float32), d1], axis=1)
    return np.stack([raw, std, d1], axis=1).astype(np.float32)


Xc  = make_channels(X)
Xtc = make_channels(Xt)
print("channels", Xc.shape, "avgpool_pre", AVGPOOL_PRE, flush=True)


class Net(nn.Module):
    def __init__(self, in_ch=3, ncls=9, p=0.3, pre=2):
        super().__init__()
        self.pre  = nn.AvgPool1d(pre)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.GELU())

        def block(ci, co):
            return nn.Sequential(
                nn.Conv1d(ci, co, kernel_size=5, padding=2),
                nn.BatchNorm1d(co), nn.GELU(), nn.MaxPool1d(2))

        self.b1 = block(32, 64); self.b2 = block(64, 96); self.b3 = block(96, 128)
        self.ap = nn.AdaptiveAvgPool1d(1); self.mp = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(p); self.fc = nn.Linear(256, ncls)

    def forward(self, x):
        x = self.pre(x); x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        z = torch.cat([self.ap(x).squeeze(-1), self.mp(x).squeeze(-1)], dim=1)
        return self.fc(self.drop(z))


def softmax_np(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


EPOCHS = 25; PATIENCE = 5; BS = 128; LR = 2e-3; WD = 1e-4

oof_path  = os.path.join(BASE, "oof",       f"{NAME}.npy")
test_path = os.path.join(BASE, "test_pred", f"{NAME}.npy")
done_path = os.path.join(BASE, "data",      f"{NAME}_folds_done.npy")

os.makedirs(os.path.join(BASE, "oof"),       exist_ok=True)
os.makedirs(os.path.join(BASE, "test_pred"), exist_ok=True)

oof      = np.zeros((N,  NCLS), np.float32)
test_acc = np.zeros((NT, NCLS), np.float32)
done     = np.zeros(5, np.int8)

if os.path.exists(done_path) and os.path.exists(oof_path) and os.path.exists(test_path):
    try:
        oof = np.load(oof_path); done = np.load(done_path)
        test_acc = np.load(test_path) * (int(done.sum()) or 1)
        print("resumed; folds done:", done.tolist(), flush=True)
    except Exception as e:
        print("resume failed:", e, flush=True)
        oof = np.zeros((N, NCLS), np.float32)
        test_acc = np.zeros((NT, NCLS), np.float32)
        done = np.zeros(5, np.int8)

fold_accs = []
Xt_t = torch.from_numpy(Xtc)

for f in range(5):
    if done[f]:
        facc = (oof[folds==f].argmax(1)==y[folds==f]).mean()
        fold_accs.append(float(facc))
        print(f"fold {f}: already done facc={facc:.4f}", flush=True); continue

    t0  = time.time()
    tr  = folds != f; va = folds == f
    Xtr = torch.from_numpy(Xc[tr]); ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(Xc[va]); yva_np = y[va]

    torch.manual_seed(100 + f)
    net   = Net(p=0.3, pre=AVGPOOL_PRE)
    opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    ntr = Xtr.shape[0]; best_acc = -1.0; best_state = None; no_imp = 0

    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, BS):
            idx = perm[i:i+BS]; opt.zero_grad()
            lossf(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vlog = torch.cat([net(Xva[i:i+256]) for i in range(0,Xva.shape[0],256)]).numpy()
        vacc = (vlog.argmax(1)==yva_np).mean()
        if vacc > best_acc + 1e-5:
            best_acc = vacc
            best_state = {k: v.clone() for k,v in net.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
        print(f"  fold {f} ep {ep+1}: val={vacc:.4f} best={best_acc:.4f}", flush=True)
        if no_imp >= PATIENCE: print(f"  early stop ep {ep+1}", flush=True); break

    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        vlog = torch.cat([net(Xva[i:i+256]) for i in range(0,Xva.shape[0],256)]).numpy()
        tlog = torch.cat([net(Xt_t[i:i+256]) for i in range(0,NT,256)]).numpy()

    oof[va]   = softmax_np(vlog)
    test_acc += softmax_np(tlog)
    done[f]   = 1
    facc = (oof[va].argmax(1)==y[va]).mean(); fold_accs.append(float(facc))
    print(f"fold {f}: best={best_acc:.4f} facc={facc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    nd = int(done.sum())
    np.save(oof_path, oof.astype(np.float32))
    np.save(test_path, (test_acc/nd).astype(np.float32))
    np.save(done_path, done)

oof_acc = (oof.argmax(1)==y).mean()
print("FOLD ACCS:", [round(a,4) for a in fold_accs], flush=True)
print(f"MEAN OOF ACC: {oof_acc:.4f}", flush=True)
nd = int(done.sum())
np.save(oof_path, oof.astype(np.float32))
np.save(test_path, (test_acc/nd).astype(np.float32))
print("DONE", flush=True)
