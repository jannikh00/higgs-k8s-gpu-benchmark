"""Download the UCI HIGGS dataset and decompress it onto the PVC."""
import argparse, gzip, os, shutil, urllib.request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="/data")
ap.add_argument("--timeout", type=float, default=120)
args = ap.parse_args()
out = os.path.join(args.out_dir, "HIGGS.csv")

if os.path.exists(out):
    print(f"{out} exists ({os.path.getsize(out) / 2**30:.1f} GiB), skipping")
    raise SystemExit

# Stream-decompress so the 2.6 GiB archive is never stored; .part guards against
# a half-written file being mistaken for a complete one on retry.
# Without the timeout a stalled socket hangs forever and backoffLimit never fires.
print(f"downloading {URL}", flush=True)
with urllib.request.urlopen(URL, timeout=args.timeout) as r, gzip.open(r) as gz:
    with open(out + ".part", "wb") as f:
        while chunk := gz.read(1 << 20):
            f.write(chunk)
            if f.tell() % (1 << 30) < (1 << 20):
                print(f"  {f.tell() / 2**30:.0f} GiB", flush=True)
os.replace(out + ".part", out)
print(f"wrote {out} ({os.path.getsize(out) / 2**30:.1f} GiB)")
