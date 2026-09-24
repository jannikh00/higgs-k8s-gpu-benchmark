"""MLP training and CPU/GPU benchmark on HIGGS (Phases 4-6).

TODO — not implemented yet. Per the project plan:

Flags
  --device {cpu,cuda}
  --threads N        (torch.set_num_threads(N), matched to the Job's CPU request)
  --batch-size       (default 1024)
  --data-path        (the CSV on the PVC)

Data
  pd.read_csv(path, header=None, dtype=np.float32, engine="pyarrow")
  column 0 = label, columns 1-28 = features
  train = rows [:10_000_000], test = rows [-1_000_000:], no shuffle before split
  standardize with train-set mean/std only, applied to both splits

Model
  nn.Sequential: 28 -> 300 (ReLU) -> 300 (ReLU) -> 100 (ReLU) -> 1 (raw logit)
  129,201 trainable parameters; print the count to corroborate the hand calc.

Batching
  Do NOT use DataLoader(TensorDataset(...)) over 10M rows.
  Move the full training tensor to the device once (~1.1 GB float32), then each
  epoch shuffle with torch.randperm and slice batches from the device tensor.
  Same logic on CPU so the comparison stays controlled.

Training
  Adam, lr=0.001, 5 epochs, BCEWithLogitsLoss, batch size 1024, fixed seed.

Timing (time.perf_counter, with torch.cuda.synchronize() before every
timestamp on GPU)
  1. data loading  2. host-to-device  3. model init
  4. training (per epoch and total)  5. evaluation  6. total pipeline
  Print a clean summary block at the end.

Metrics
  per epoch: train loss + accuracy
  final: test accuracy (sigmoid(logit) > 0.5), test loss, optionally AUC
  Sanity check: test accuracy should land in the mid-to-high 70s %.
"""

raise NotImplementedError("train.py is a stub; see the TODO above.")
