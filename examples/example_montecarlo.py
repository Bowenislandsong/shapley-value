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
# 3. Parallel processing
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Parallel Processing with n_jobs")
print("=" * 60)

import time

for n_jobs in [1, 2, -1]:
    label = "all cores" if n_jobs == -1 else f"{n_jobs} job(s)"
    mc_par = MonteCarloShapleyValue(
        revenue, players=players, num_samples=3000, random_seed=42, n_jobs=n_jobs
    )
    t0 = time.time()
    mc_par.calculate_shapley_values()
    elapsed = time.time() - t0
    print(f"  n_jobs={n_jobs:>2} ({label:<10}): {elapsed:.3f}s")

print("\n  Note: parallel speedup is most visible for large num_samples and")
print("  expensive evaluation functions.")


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
