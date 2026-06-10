"""
per-row z-score 후 2채널 입력(std/diff) 1D CNN — 스케일-불변 변형.
raw 채널 제거로 진폭 정보를 완전 차단하고 형태·동역학만 학습. round6 다양성 기여 모델.
"""
import os, time
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(5)
np.random.seed(42); torch.manual_seed(42)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data")
NAME  = "cnn1d_std"
DEV   = "cpu"

X     = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
y     = np.load(os.path.join(DATA, "y_train.npy")).astype(np.int64)
folds = np.load(os.path.join(DATA, "folds.npy")).astype(np.int64)
Xt    = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)

N, L  = X.shape
NT    = Xt.shape[0]
NCLS  = 9

AVGPOOL_PRE = int(os.environ.get("AVGPOOL_PRE", "2"))


def make_channels(arr):
    """행별 z-score + diff(z) 2채널 — 진폭 정보 차단."""
    rmean = arr.mean(1, keepdims=True)
    rstd  = arr.std(1, keepdims=True) + 1e-6
    std   = (arr - rmean) / rstd
    d1    = np.diff(std, axis=1)
    d1    = np.concatenate([np.zeros((arr.shape[0], 1), np.float32), d1], axis=1)
    return np.stack([std, d1], axis=1).astype(np.float32)


Xc  = make_channels(X)
Xtc = make_channels(Xt)
print("channels", Xc.shape, Xtc.shape, "avgpool_pre", AVGPOOL_PRE, flush=True)


class Net(nn.Module):
    def __init__(self, in_ch=2, ncls=9, p=0.3, pre=2):
        super().__init__()
        self.pre  = nn.AvgPool1d(pre)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
        )

        def block(ci, co):
            return nn.Sequential(
                nn.Conv1d(ci, co, kernel_size=5, padding=2),
                nn.BatchNorm1d(co), nn.GELU(),
                nn.MaxPool1d(2),
            )
        self.b1   = block(32, 64)
        self.b2   = block(64, 96)
        self.b3   = block(96, 128)
        self.ap   = nn.AdaptiveAvgPool1d(1)
        self.mp   = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(p)
        self.fc   = nn.Linear(256, ncls)

    def forward(self, x):
        x = self.pre(x)
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        a = self.ap(x).squeeze(-1)
        m = self.mp(x).squeeze(-1)
        z = torch.cat([a, m], dim=1)
        z = self.drop(z)
        return self.fc(z)


def softmax_np(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


EPOCHS  = 25
PATIENCE = 5
BS = 128
LR = 2e-3
WD = 1e-4

os.makedirs(os.path.join(BASE, "oof"),       exist_ok=True)
os.makedirs(os.path.join(BASE, "test_pred"), exist_ok=True)

oof_path  = os.path.join(BASE, "oof",       f"{NAME}.npy")
test_path = os.path.join(BASE, "test_pred", f"{NAME}.npy")
done_path = os.path.join(BASE, "oof",       f"{NAME}_folds_done.npy")

oof      = np.zeros((N,  NCLS), np.float32)
test_acc = np.zeros((NT, NCLS), np.float32)
done     = np.zeros(5, np.int8)
if os.path.exists(done_path) and os.path.exists(oof_path) and os.path.exists(test_path):
    try:
        oof      = np.load(oof_path)
        test_acc = np.load(test_path) * (np.load(done_path).sum() or 1)
        done     = np.load(done_path)
        print("resumed; folds done:", done.tolist(), flush=True)
    except Exception as e:
        print("resume failed, fresh start:", e, flush=True)
        oof      = np.zeros((N,  NCLS), np.float32)
        test_acc = np.zeros((NT, NCLS), np.float32)
        done     = np.zeros(5, np.int8)

fold_accs = []
Xt_t      = torch.from_numpy(Xtc)

for f in range(5):
    if done[f]:
        facc = (oof[folds == f].argmax(1) == y[folds == f]).mean()
        fold_accs.append(float(facc))
        print(f"fold {f}: already done facc={facc:.4f}", flush=True)
        continue
    t0  = time.time()
    tr  = folds != f
    va  = folds == f
    Xtr = torch.from_numpy(Xc[tr]);  ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(Xc[va]);  yva_np = y[va]

    torch.manual_seed(100 + f)
    net   = Net(p=0.3, pre=AVGPOOL_PRE).to(DEV)
    opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    ntr      = Xtr.shape[0]
    best_acc = -1.0
    best_state = None
    no_imp   = 0
    ep_used  = 0

    for ep in range(EPOCHS):
        te = time.time()
        net.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, BS):
            idx = perm[i:i + BS]
            xb  = Xtr[idx]; yb = ytr[idx]
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vlog = []
            for i in range(0, Xva.shape[0], 256):
                vlog.append(net(Xva[i:i + 256]))
            vlog = torch.cat(vlog).numpy()
        vacc    = (vlog.argmax(1) == yva_np).mean()
        ep_used = ep + 1
        ept     = time.time() - te
        if vacc > best_acc + 1e-5:
            best_acc   = vacc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        print(f"  fold {f} ep {ep+1}: val_acc={vacc:.4f} best={best_acc:.4f} ep_time={ept:.0f}s", flush=True)
        if no_imp >= PATIENCE:
            print(f"  fold {f} early stop at ep {ep+1}", flush=True)
            break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        vlog = []
        for i in range(0, Xva.shape[0], 256):
            vlog.append(net(Xva[i:i + 256]))
        vlog = torch.cat(vlog).numpy()
        tlog = []
        for i in range(0, NT, 256):
            tlog.append(net(Xt_t[i:i + 256]))
        tlog = torch.cat(tlog).numpy()

    oof[va]   = softmax_np(vlog)
    test_acc += softmax_np(tlog)
    done[f]   = 1
    facc      = (oof[va].argmax(1) == y[va]).mean()
    fold_accs.append(float(facc))
    print(f"fold {f}: best_val_acc={best_acc:.4f} facc={facc:.4f} ep_used={ep_used} time={time.time()-t0:.0f}s", flush=True)

    nd = int(done.sum())
    np.save(oof_path,  oof.astype(np.float32))
    np.save(test_path, (test_acc / nd).astype(np.float32))
    np.save(done_path, done)
    print(f"  checkpoint saved ({nd}/5 folds)", flush=True)

oof_acc = (oof.argmax(1) == y).mean()
print("FOLD ACCS:", [round(a, 4) for a in fold_accs], flush=True)
print(f"OOF 전체 acc: {oof_acc:.4f}", flush=True)

nd = int(done.sum())
np.save(oof_path,  oof.astype(np.float32))
np.save(test_path, (test_acc / nd).astype(np.float32))
print("저장 완료:", NAME, flush=True)
