"""Reconcile the day coverage of the benchmark and the baseline runs.
 
The two runs report different headline revenues over what is nominally the same
sample. The stated explanation is that the threshold rule needs a prior day to
compute its trailing percentiles and therefore cannot trade the first day of the
sample, leaving the baseline comparison one day short at the start.
 
That explanation is plausible and was never checked. This script checks it. It
prints the days present in one run and absent from the other, in both
directions, so the claim either holds exactly or fails visibly.
 
The claim holds only if the benchmark has exactly one extra day, that day is the
first in the sample, and the baseline has no extra days at all. Anything else -
a missing day in the middle, a day the baseline solved and the benchmark did
not - means days are being dropped for a reason nobody has identified, and the
revenue difference between the two runs is measuring something other than the
optimiser against the rule.
 
Usage:
 
    python -m scripts.check_benchmark_coverage
"""
 
from __future__ import annotations
 
import argparse
 
import pandas as pd
 
 
def compare(benchmark: pd.DataFrame, baseline: pd.DataFrame) -> str:
    bench_days = set(pd.to_datetime(benchmark.index).date)
    base_days = set(pd.to_datetime(baseline.index).date)
 
    only_bench = sorted(bench_days - base_days)
    only_base = sorted(base_days - bench_days)
    shared = bench_days & base_days
 
    first_day = min(bench_days | base_days)
    claim_holds = (
        len(only_bench) == 1 and only_bench[0] == first_day and len(only_base) == 0
    )
 
    lines = [
        f"benchmark days              {len(bench_days)}",
        f"baseline days               {len(base_days)}",
        f"shared days                 {len(shared)}",
        f"sample starts               {first_day}",
        "",
        f"in benchmark only ({len(only_bench)}): {only_bench[:10]}",
        f"in baseline only  ({len(only_base)}): {only_base[:10]}",
        "",
        f"stated explanation holds:   {claim_holds}",
    ]
 
    if not claim_holds:
        lines += [
            "",
            "The difference is NOT a single missing first day. Until the extra",
            "days are explained, the two runs are not measured over the same",
            "sample and their revenues are not directly comparable. Restrict",
            "both to the shared days before quoting a capture percentage.",
        ]
 
    if shared:
        shared_index = sorted(shared)
        bench_shared = benchmark.loc[
            pd.to_datetime(benchmark.index).normalize().isin(pd.to_datetime(shared_index))
        ]
        base_shared = baseline.loc[
            pd.to_datetime(baseline.index).normalize().isin(pd.to_datetime(shared_index))
        ]
        lines += [
            "",
            "Restricted to shared days only:",
            f"  benchmark net revenue GBP {bench_shared['net_revenue_gbp'].sum():,.0f}",
            f"  baseline  net revenue GBP {base_shared['net_revenue_gbp'].sum():,.0f}",
        ]
 
    return "\n".join(lines)
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/processed/benchmark_daily.parquet")
    parser.add_argument("--baseline", default="data/processed/baseline_daily.parquet")
    args = parser.parse_args()
 
    print(compare(pd.read_parquet(args.benchmark), pd.read_parquet(args.baseline)))
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 