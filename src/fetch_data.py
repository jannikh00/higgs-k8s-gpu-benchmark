"""Download the UCI HIGGS dataset and decompress it onto the PVC."""
import argparse, gzip, os, shutil, urllib.request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="/data")
out = os.path.join(ap.parse_args().out_dir, "HIGGS.csv")

if os.path.exists(out):
    print(f"{out} exists ({os.path.getsize(out) / 2**30:.1f} GiB), skipping")
    raise SystemExit

# Stream-decompress so the 2.6 GiB archive is never stored; .part guards against
# a half-written file being mistaken for a complete one on retry.
print(f"downloading {URL}")
with urllib.request.urlopen(URL) as r, gzip.open(r) as gz, open(out + ".part", "wb") as f:
    shutil.copyfileobj(gz, f, 1 << 20)
os.replace(out + ".part", out)
print(f"wrote {out} ({os.path.getsize(out) / 2**30:.1f} GiB)")
