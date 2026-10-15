"""Plot bench.py CSV output: latency and TFLOP/s vs sequence length per backend."""
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default="bench.png")
    a = p.parse_args()
    data = defaultdict(list)
    with open(a.csv) as f:
        for row in csv.DictReader(f):
            data[row["backend"]].append((int(row["seqlen"]), float(row["ms"]), float(row["tflops"])))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for be, rows in data.items():
        rows.sort()
        ax1.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label=be)
        ax2.plot([r[0] for r in rows], [r[2] for r in rows], marker="o", label=be)
    ax1.set(xscale="log", yscale="log", xlabel="sequence length", ylabel="latency (ms)", title="Latency")
    ax2.set(xscale="log", xlabel="sequence length", ylabel="TFLOP/s", title="Throughput")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
