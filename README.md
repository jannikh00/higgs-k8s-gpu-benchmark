# higgs-k8s-gpu-benchmark

Containerized MLP pipeline on the Nautilus (NRP) Kubernetes cluster. One image,
built for `linux/amd64` by GitHub Actions and pushed to GHCR, serves both the GPU
verification task and the HIGGS training benchmark.

## Layout

```
.github/workflows/build.yml   build the amd64 image, push to GHCR
Dockerfile                    CUDA 12.4 base + Python + requirements
requirements.txt              torch (cu124), numpy, pandas, pyarrow
src/
  container_nautilus_gpu_test.py   provided GPU verification script (image default CMD)
  fetch_data.py                    download + decompress HIGGS.csv onto the PVC
  train.py                         MLP training, timing instrumentation, CPU/GPU benchmark
k8s/
  deployment.yaml             Task 1: GPU verification Job
  data-pvc.yaml               20Gi PVC for the dataset
  fetch-data.yaml             one-off data download Job
  train-gpu.yaml              Task 2 GPU run
  train-cpu.yaml              Task 2 CPU run
```

## Image

`ghcr.io/jannikh00/higgs-k8s-gpu-benchmark`

Pushes to `main` build and push both `:latest` and `:<git-sha>`. The Kubernetes
manifests reference the **SHA tag**, not `latest`, so each run is tied to an exact
image.

Do not build locally on Apple silicon with plain `docker build` — the result is
arm64 and the pod fails on Nautilus with `exec format error`. Use
`docker buildx build --platform linux/amd64` if you must build by hand.

## Setup

Create the GHCR pull secret in the namespace once (classic PAT, `read:packages`
scope only):

```bash
kubectl create secret docker-registry ghcr-pull \
  --docker-server=ghcr.io --docker-username=jannikh00 \
  --docker-password=<PAT> -n <namespace>
```

## Running

```bash
# Task 1: GPU verification
kubectl apply -f k8s/deployment.yaml
kubectl get jobs,pods
kubectl logs job/<netid>-gpu-check
kubectl describe pod <pod-name>     # shows the node and GPU model

# Data
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/fetch-data.yaml

# Task 2: benchmark (GPU first, it validates the code end to end)
kubectl apply -f k8s/train-gpu.yaml
kubectl apply -f k8s/train-cpu.yaml
```

Delete each Job when it is done. NRP polices idle GPUs — never leave a GPU pod
sleeping.

## Model

`nn.Sequential`: 28 → 300 (ReLU) → 300 (ReLU) → 100 (ReLU) → 1 (raw logit).

| Layer | Weights | Biases |
|---|---|---|
| 28 → 300 | 8,400 | 300 |
| 300 → 300 | 90,000 | 300 |
| 300 → 100 | 30,000 | 100 |
| 100 → 1 | 100 | 1 |
| **Total** | **128,500** | **701** |

Trainable parameters: **129,201**.

## Status

Scaffolding is in place. Still to do:

- [ ] `src/fetch_data.py` — stub
- [ ] `src/train.py` — stub
- [ ] Fill in `<netid>`, `<sha>`, `<storage-class>` in the manifests
- [ ] Confirm the CUDA 12.4 pin against the Nautilus driver version
