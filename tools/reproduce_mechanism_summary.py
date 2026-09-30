#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "reproducibility" / "mechanism"

method = pd.read_csv(D/"reachability"/"b1_external_response_method_summary_v1.csv")
print("External-output method summary:")
print(method.to_string(index=False))

src = pd.read_csv(D/"sourcewise"/"sourcewise_join_v1.csv")
print("\nAvailable columns:", list(src.columns))

print(
    "\nThis script exposes the released per-source tables so that the "
    "paper's reachability-realization associations can be recomputed "
    "without rerunning external restoration methods."
)
