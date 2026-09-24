import time
import numpy as np
import torch

# =========================================================
# CONFIG
# =========================================================
MAT_SIZE = 600
VEC_SIZE = 10_000_000
SEED = 42

# =========================================================
# UTILITIES
# =========================================================
def timer(func, name):
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    print(f"{name:<22} {end-start:.6f} sec")
    return result

def sync_if_gpu(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

# =========================================================
# DOT PRODUCT METHODS
# =========================================================
def dot_native(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

def dot_numpy(a, b):
    return np.dot(a, b)

def dot_torch(a, b, device):
    def run():
        out = torch.dot(a, b)
        sync_if_gpu(device)
        return out
    return run

# =========================================================
# MATRIX MULTIPLICATION METHODS
# =========================================================
def matmul_native(A, B):
    n = len(A)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def matmul_numpy(A, B):
    return A @ B

def matmul_torch(A, B, device):
    def run():
        out = torch.matmul(A, B)
        sync_if_gpu(device)
        return out
    return run

# =========================================================
# DATA GENERATION
# =========================================================
def generate_data(device):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("\nGenerating data...\n")

    # vectors
    v1_np = np.random.rand(VEC_SIZE).astype(np.float32)
    v2_np = np.random.rand(VEC_SIZE).astype(np.float32)

    v1_list = v1_np.tolist()
    v2_list = v2_np.tolist()

    v1_torch = torch.from_numpy(v1_np).to(device)
    v2_torch = torch.from_numpy(v2_np).to(device)

    # matrices
    A_np = np.random.rand(MAT_SIZE, MAT_SIZE).astype(np.float32)
    B_np = np.random.rand(MAT_SIZE, MAT_SIZE).astype(np.float32)

    A_list = A_np.tolist()
    B_list = B_np.tolist()

    A_torch = torch.from_numpy(A_np).to(device)
    B_torch = torch.from_numpy(B_np).to(device)

    print("Data ready.\n")

    return (
        (v1_np, v2_np, v1_list, v2_list, v1_torch, v2_torch),
        (A_np, B_np, A_list, B_list, A_torch, B_torch),
    )

# =========================================================
# BENCHMARK RUNNERS
# =========================================================
def run_dot_bench(vdata, device):
    print("=== DOT PRODUCT ===\n")

    v1_np, v2_np, v1_list, v2_list, v1_torch, v2_torch = vdata

    d_native = timer(lambda: dot_native(v1_list, v2_list), "Native loop")
    d_np     = timer(lambda: dot_numpy(v1_np, v2_np), "NumPy")

    torch_func = dot_torch(v1_torch, v2_torch, device)
    _ = torch_func()  # warmup
    d_torch = timer(torch_func, f"PyTorch ({device})")


def run_matmul_bench(mdata, device):
    print("\n=== MATRIX MULTIPLICATION ===\n")

    A_np, B_np, A_list, B_list, A_torch, B_torch = mdata

    C_native = timer(lambda: matmul_native(A_list, B_list), "Native loops")
    C_np     = timer(lambda: matmul_numpy(A_np, B_np), "NumPy")

    torch_func = matmul_torch(A_torch, B_torch, device)
    _ = torch_func()
    C_torch = timer(torch_func, f"PyTorch ({device})")

# =========================================================
# MAIN
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vdata, mdata = generate_data(device)

    run_dot_bench(vdata, device)
    run_matmul_bench(mdata, device)

    print("\nDevice:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nDone.")


if __name__ == "__main__":
    main()
    