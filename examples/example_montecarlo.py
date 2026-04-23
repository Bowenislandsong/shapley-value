#!/usr/bin/env python3
"""
Monte Carlo Shapley Value Example

Demonstrates MonteCarloShapleyValue for approximating Shapley values in games
where the exact computation (all 2^n coalitions) is too expensive.

Topics covered:
  1. Basic usage with a simple evaluation function
  2. Choosing num_samples – comparing MC vs exact for a small game
  3. Parallel processing with n_jobs
  4. Diagnosing convergence with get_convergence_data()
  5. Inspecting raw per-permutation data with get_raw_data()
"""

from shapley_value import MonteCarloShapleyValue, ShapleyCombinations


# ---------------------------------------------------------------------------
# 1. Basic usage
# ---------------------------------------------------------------------------

def revenue(coalition):
    """Non-linear revenue: synergies make the grand coalition more valuable."""
    if not coalition:
        return 0.0
    total = float(sum(coalition))
    # Superadditive: scale by coalition size
    return total * (1 + 0.1 * (len(coalition) - 1))


print("=" * 60)
print("1. Basic Monte Carlo Shapley Value")
print("=" * 60)

players = [10, 20, 30, 40]

mc = MonteCarloShapleyValue(
    revenue,
    players=players,
    num_samples=2000,
    random_seed=42,
)
values = mc.calculate_shapley_values()

print(f"Players: {players}")
print("Shapley values (MC, 2 000 samples):")
for player, value in values.items():
    print(f"  Player {player:2d}: {value:.4f}")

grand = revenue(players)
print(f"\nGrand coalition value : {grand:.4f}")
print(f"Sum of Shapley values : {sum(values.values()):.4f}  (must equal grand coalition)")


# ---------------------------------------------------------------------------
# 2. Comparing MC vs exact for a small game
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Monte Carlo vs Exact – Alice / Bob / Charlie")
print("=" * 60)

abc_players = ["Alice", "Bob", "Charlie"]
coalition_values = {
    (): 0,
    ("Alice",): 10,
    ("Bob",): 15,
    ("Charlie",): 12,
    ("Alice", "Bob"): 35,
    ("Alice", "Charlie"): 30,
    ("Bob", "Charlie"): 32,
    ("Alice", "Bob", "Charlie"): 60,
}


def abc_game(coalition):
    key = tuple(sorted(coalition))
    return float(coalition_values.get(key, 0))


exact_calc = ShapleyCombinations(abc_players)
exact = exact_calc.calculate_shapley_values(coalition_values)

for n_samples in [100, 500, 2000, 10_000]:
    mc_abc = MonteCarloShapleyValue(
        abc_game, abc_players, num_samples=n_samples, random_seed=0
    )
    approx = mc_abc.calculate_shapley_values()
    max_err = max(abs(approx[p] - exact[p]) for p in abc_players)
    print(f"  {n_samples:>6} samples → max absolute error: {max_err:.4f}")

print("\nExact Shapley values:")
for player, value in exact.items():
    print(f"  {player}: {value:.4f}")


# ---------------------------------------------------------------------------
# 3. Parallel processing – performance benchmark
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Parallel Processing – Performance Benchmark")
print("=" * 60)

import os
import time


def slow_game(coalition):
    """Simulates an expensive evaluation (e.g. model inference, ~0.1 ms/call)."""
    if not coalition:
        return 0.0
    acc = 0.0
    for i in range(10_000):
        acc += i * 1e-10
    return float(sum(coalition)) + acc


cpu_count = os.cpu_count() or 1
print(f"\n  Machine: {cpu_count} logical CPU(s)")
print(f"  Game: slow_game (≈0.1 ms per evaluation)")

# ── table header ────────────────────────────────────────────────────────────
col_w = 14
header_players  = f"{'Players':>8}"
header_samples  = f"{'Samples':>8}"
sep = "-" * (8 + 2 + 8 + 3 * (col_w + 2))

print(f"\n  {header_players}  {header_samples}  "
      f"{'n_jobs=1':>{col_w}}  {'n_jobs=2':>{col_w}}  {'n_jobs=-1':>{col_w}}")
print("  " + sep)

bench_configs = [
    ([1, 2, 3, 4, 5], 100),
    ([1, 2, 3, 4, 5], 300),
    (list(range(1, 11)), 200),
]

for bench_players, bench_samples in bench_configs:
    row_times = {}
    for n_jobs in [1, 2, -1]:
        mc_b = MonteCarloShapleyValue(
            slow_game, bench_players, num_samples=bench_samples,
            random_seed=42, n_jobs=n_jobs,
        )
        t0 = time.perf_counter()
        mc_b.calculate_shapley_values()
        row_times[n_jobs] = time.perf_counter() - t0

    speedup_2    = row_times[1] / row_times[2]    if row_times[2]  > 0 else float("inf")
    speedup_all  = row_times[1] / row_times[-1]   if row_times[-1] > 0 else float("inf")

    cells = (
        f"  {len(bench_players):>8}  {bench_samples:>8}"
        f"  {row_times[1]:>{col_w-3}.3f}s     "
        f"  {row_times[2]:>{col_w-3}.3f}s {speedup_2:4.1f}×"
        f"  {row_times[-1]:>{col_w-3}.3f}s {speedup_all:4.1f}×"
    )
    print(cells)

print("  " + sep)
print("  Speedup shown as ×seq (e.g. 2.1× means 2.1× faster than n_jobs=1)")
print("\n  Tip: for cheap evaluation functions (< 1 µs) the joblib overhead")
print("  dominates and sequential is faster. For expensive functions")
print("  (ML models, simulations) set n_jobs=-1 for maximum throughput.")


# ---------------------------------------------------------------------------
# 4. Convergence data – how many samples do you really need?
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Convergence Diagnostics")
print("=" * 60)

mc_conv = MonteCarloShapleyValue(
    abc_game, abc_players, num_samples=1000, random_seed=7
)
conv_df = mc_conv.get_convergence_data()

print(f"Convergence DataFrame shape: {conv_df.shape}  (rows=iterations, cols=players)")
print("\nRunning estimates at selected iterations:")
print(f"  {'Iteration':>10}  " + "  ".join(f"{p:>10}" for p in abc_players))

for idx in [0, 9, 49, 99, 499, 999]:
    row = conv_df.iloc[idx]
    vals = "  ".join(f"{row[p]:>10.4f}" for p in abc_players)
    print(f"  {idx + 1:>10}  {vals}")

print("\nExact values:          " + "  ".join(f"{exact[p]:>10.4f}" for p in abc_players))

# Show when estimates have stabilised (std of last 100 rows)
print("\nStd-dev of last 100 running estimates (smaller = more stable):")
for player in abc_players:
    std = conv_df[player].iloc[-100:].std()
    print(f"  {player}: {std:.4f}")


# ---------------------------------------------------------------------------
# 5. Raw per-permutation data
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. Raw Per-Permutation Data")
print("=" * 60)

mc_raw = MonteCarloShapleyValue(
    abc_game, abc_players, num_samples=5, random_seed=0
)
raw_df = mc_raw.get_raw_data()

print(f"Shape: {raw_df.shape}  ({mc_raw.num_samples} samples × {len(abc_players)} players)")
print("\nFirst few rows:")
print(raw_df.to_string(index=False))

print("\nPer-permutation contribution sums (should all equal 60):")
for it, grp in raw_df.groupby("iteration"):
    total = grp["marginal_contribution"].sum()
    print(f"  Iteration {it}: {total:.4f}")
