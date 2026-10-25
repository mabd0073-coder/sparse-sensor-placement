"""Make a sweep CSV from selected (e.g. still-pending) cases of an existing sweep CSV.

Use it to hand pending jobs to someone else: cancel them on the cluster
(scancel), then give them the new CSV. Cases are renumbered 1..n in the team
format (case,frh,rrh,r,s,cr,y,v,run). A mapping file *_mapping.csv records
the original case number, though the folder names also encode every parameter.

Example (sweep 50.42 was made from straight_rrf_batch2.csv, jobs 1-32 pending):
    python sweeps/split_pending.py sweeps/new/straight_rrf_batch2.csv 1-32 -o pending_50.42.csv
    python sweeps/split_pending.py sweeps/new/cornering_rrf_batch2.csv 17-32 -o pending_50.41.csv
"""
import argparse

import pandas as pd


def parse_cases(spec):
    """'17-32,5,9' -> [5, 9, 17, ..., 32]"""
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_csv")
    p.add_argument("cases", help="job/case numbers, e.g. 17-32 or 17,18")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    df = pd.read_csv(a.sweep_csv)
    pick = df[df["case"].isin(parse_cases(a.cases))].copy()
    if pick.empty:
        raise SystemExit("no matching cases")
    pick.insert(0, "orig_case", pick.pop("case"))
    pick.insert(0, "case", range(1, len(pick) + 1))
    pick.drop(columns="orig_case").to_csv(a.out, index=False)
    pick[["case", "orig_case"]].to_csv(a.out.replace(".csv", "_mapping.csv"), index=False)
    print(f"{a.out}: {len(pick)} cases (original numbers {pick.orig_case.min()}-{pick.orig_case.max()})")


if __name__ == "__main__":
    main()
