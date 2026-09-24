"""Download the UCI HIGGS dataset onto the PVC (Phase 3).

TODO — not implemented yet. Per the project plan:
  - Download the HIGGS archive (~2.6 GB compressed) from the UCI page.
  - Decompress it to a plain HIGGS.csv (~8 GB) on the mounted PVC.
  - Download and decompression stay outside the benchmark, so that "data
    loading" in train.py is purely CSV parsing, identical for CPU and GPU.
  - Skip the download if the CSV is already present, so the Job is re-runnable.
"""

raise NotImplementedError("fetch_data.py is a stub; see the TODO above.")
