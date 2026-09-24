"""MLP training and CPU/GPU benchmark on the UCI HIGGS dataset."""
import argparse, time
import numpy as np, pandas as pd, torch, torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--batch-size", type=int, default=1024)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--data-path", default="/data/HIGGS.csv")
p.add_argument("--train-rows", type=int, default=10_000_000)
p.add_argument("--test-rows", type=int, default=1_000_000)
a = p.parse_args()

if a.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("--device cuda requested but CUDA is not available")
torch.set_num_threads(a.threads)
torch.manual_seed(0)
dev = torch.device(a.device)


def stamp():
    """CUDA kernels are async, so sync before reading the clock."""
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


T = {}
t_start = stamp()

# Data loading: CSV parse plus standardisation
t = stamp()
arr = pd.read_csv(a.data_path, header=None, dtype=np.float32, engine="pyarrow").to_numpy()
assert len(arr) >= a.train_rows + a.test_rows, f"{a.data_path} has only {len(arr):,} rows"
X, y = arr[:, 1:], arr[:, 0]
Xtr, ytr = X[: a.train_rows], y[: a.train_rows]
Xte, yte = X[-a.test_rows :], y[-a.test_rows :]
mean, std = Xtr.mean(0), Xtr.std(0)
std[std == 0] = 1  # a constant column would otherwise produce NaNs
Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std  # train statistics only
T["data loading"] = stamp() - t

# Host to device: the whole training set moves once and stays resident, so no
# batch ever crosses PCIe during training
t = stamp()
Xtr_t = torch.from_numpy(Xtr).to(dev)
ytr_t = torch.from_numpy(ytr).to(dev).unsqueeze(1)
Xte_t = torch.from_numpy(Xte).to(dev)
yte_t = torch.from_numpy(yte).to(dev).unsqueeze(1)
T["host-to-device"] = stamp() - t

t = stamp()
model = nn.Sequential(
    nn.Linear(28, 300), nn.ReLU(),
    nn.Linear(300, 300), nn.ReLU(),
    nn.Linear(300, 100), nn.ReLU(),
    nn.Linear(100, 1),
).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()
T["model init"] = stamp() - t
print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
print(f"trainable parameters: {sum(q.numel() for q in model.parameters() if q.requires_grad):,}")

t = stamp()
n = len(Xtr_t)
for ep in range(a.epochs):
    t_ep = stamp()
    perm = torch.randperm(n, device=dev)
    # Accumulate on the device: calling .item() per batch would sync every
    # iteration and distort exactly what this script is trying to measure
    run_loss = torch.zeros((), device=dev)
    run_hits = torch.zeros((), device=dev)
    for i in range(0, n, a.batch_size):
        idx = perm[i : i + a.batch_size]
        xb, yb = Xtr_t[idx], ytr_t[idx]
        logit = model(xb)
        loss = loss_fn(logit, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.detach() * len(idx)
        run_hits += ((logit > 0) == (yb > 0.5)).sum()
    print(f"epoch {ep + 1}/{a.epochs}  loss {run_loss.item() / n:.4f}  "
          f"acc {run_hits.item() / n:.4f}  ({stamp() - t_ep:.1f}s)")
T["training"] = stamp() - t

t = stamp()
with torch.no_grad():
    # Batched so a 1M-row forward pass cannot blow up GPU memory
    logits = torch.cat([model(Xte_t[i : i + a.batch_size]) for i in range(0, len(Xte_t), a.batch_size)])
    test_loss = loss_fn(logits, yte_t).item()
    test_acc = ((logits > 0) == (yte_t > 0.5)).float().mean().item()
T["evaluation"] = stamp() - t
T["total"] = stamp() - t_start

print(f"\ntest loss {test_loss:.4f}   test accuracy {test_acc:.4f}")
print(f"\n{'phase':<18}{'seconds':>10}")
for k, v in T.items():
    print(f"{k:<18}{v:>10.2f}")
