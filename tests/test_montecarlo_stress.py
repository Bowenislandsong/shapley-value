"""
Stress tests and performance benchmarks for MonteCarloShapleyValue.

Goals
-----
* Verify correctness is maintained at large scale (many players, many samples).
* Show that parallel execution (n_jobs > 1) is meaningfully faster than
  sequential when the evaluation function is expensive.
* Confirm the efficiency axiom (sum == grand coalition) is rock-solid
  regardless of scale.

These tests are intentionally heavier than unit tests; they are separated
here so the normal test run stays fast while this suite can be run on demand
or in CI with a dedicated "stress" marker.
"""

import time
import unittest

from shapley_value import MonteCarloShapleyValue


# ---------------------------------------------------------------------------
# Evaluation functions used across tests
# ---------------------------------------------------------------------------

def fast_additive(coalition):
    """Cheap additive game – good baseline for overhead measurement."""
    return float(sum(coalition)) if coalition else 0.0


def slow_game(coalition):
    """
    Simulates an expensive evaluation function (e.g. model inference).
    Uses a tight Python loop so that total sequential time dominates joblib
    process overhead; otherwise parallel can appear slower than sequential.
    """
    if not coalition:
        return 0.0
    # ~100k iterations: enough work per call that multi-core speedup is
    # measurable on CI runners; tune here if the speedup test becomes flaky.
    acc = 0.0
    for i in range(100_000):
        acc += i * 1e-10
    return float(sum(coalition)) + acc


def synergy_game(coalition):
    """Superadditive game: v(S) = |S|^1.5 * mean(S)."""
    if not coalition:
        return 0.0
    return float(len(coalition) ** 1.5) * (sum(coalition) / len(coalition))


# ---------------------------------------------------------------------------
# 1. Large-player correctness
# ---------------------------------------------------------------------------

class TestStressLargePlayers(unittest.TestCase):
    """Verify correctness properties scale to 20–100 players."""

    def _check_efficiency(self, players, num_samples, n_jobs=1, delta=None):
        """Helper: assert efficiency (sum == grand coalition value)."""
        mc = MonteCarloShapleyValue(
            fast_additive, players, num_samples=num_samples,
            random_seed=0, n_jobs=n_jobs,
        )
        result = mc.calculate_shapley_values()
        grand = fast_additive(players)
        total = sum(result.values())
        if delta is None:
            self.assertAlmostEqual(total, grand, places=6,
                msg=f"Efficiency failed: sum={total:.6f}, grand={grand:.6f}")
        else:
            self.assertAlmostEqual(total, grand, delta=delta,
                msg=f"Efficiency failed: sum={total:.6f}, grand={grand:.6f}")
        return result

    def test_20_players_efficiency(self):
        """20 players, 500 samples – efficiency must hold exactly."""
        players = list(range(1, 21))
        self._check_efficiency(players, num_samples=500)

    def test_50_players_efficiency(self):
        """50 players, 200 samples – efficiency must hold exactly."""
        players = list(range(1, 51))
        self._check_efficiency(players, num_samples=200)

    def test_100_players_efficiency(self):
        """100 players, 100 samples – efficiency must hold exactly."""
        players = list(range(1, 101))
        self._check_efficiency(players, num_samples=100)

    def test_20_players_all_keys_present(self):
        players = list(range(1, 21))
        mc = MonteCarloShapleyValue(fast_additive, players, num_samples=100, random_seed=0)
        result = mc.calculate_shapley_values()
        self.assertEqual(set(result.keys()), set(players))

    def test_20_players_additive_accuracy(self):
        """For additive game, each Shapley value ≈ player's value."""
        players = list(range(1, 21))
        mc = MonteCarloShapleyValue(
            fast_additive, players, num_samples=5000, random_seed=42
        )
        result = mc.calculate_shapley_values()
        for p in players:
            self.assertAlmostEqual(result[p], float(p), delta=3.0,
                msg=f"Player {p}: got {result[p]:.4f}, expected {float(p):.4f}")

    def test_50_players_synergy_efficiency(self):
        """Efficiency holds for a superadditive game with 50 players."""
        players = list(range(1, 51))
        mc = MonteCarloShapleyValue(
            synergy_game, players, num_samples=300, random_seed=0
        )
        result = mc.calculate_shapley_values()
        grand = synergy_game(players)
        self.assertAlmostEqual(sum(result.values()), grand, places=6)

    def test_large_sample_count(self):
        """10 000 samples with 5 players must complete and be accurate."""
        players = [10, 20, 30, 40, 50]
        mc = MonteCarloShapleyValue(
            fast_additive, players, num_samples=10_000, random_seed=0
        )
        result = mc.calculate_shapley_values()
        for p in players:
            self.assertAlmostEqual(result[p], float(p), delta=1.0,
                msg=f"Player {p}: got {result[p]:.4f}, expected {float(p):.4f}")


# ---------------------------------------------------------------------------
# 2. Performance – wall-clock timing
# ---------------------------------------------------------------------------

class TestStressPerformanceTiming(unittest.TestCase):
    """
    Wall-clock timing tests.

    These tests assert *upper bounds* on elapsed time so CI fails loudly if a
    regression makes the code unexpectedly slow.  The bounds are intentionally
    generous (10× the expected wall time on a slow CI runner) so the tests
    remain green on constrained hardware.
    """

    def test_sequential_100_players_200_samples_under_10s(self):
        """Sequential: 100 players × 200 samples must finish in < 10 s."""
        players = list(range(1, 101))
        mc = MonteCarloShapleyValue(
            fast_additive, players, num_samples=200, random_seed=0, n_jobs=1
        )
        t0 = time.perf_counter()
        mc.calculate_shapley_values()
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 10.0,
            f"Sequential 100-player run took {elapsed:.2f}s (limit 10s)")

    def test_sequential_50_players_1000_samples_under_10s(self):
        """Sequential: 50 players × 1 000 samples must finish in < 10 s."""
        players = list(range(1, 51))
        mc = MonteCarloShapleyValue(
            fast_additive, players, num_samples=1000, random_seed=0, n_jobs=1
        )
        t0 = time.perf_counter()
        mc.calculate_shapley_values()
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 10.0,
            f"Sequential 50-player 1k-sample run took {elapsed:.2f}s (limit 10s)")

    def test_parallel_all_cores_faster_than_sequential_for_expensive_game(self):
        """
        With a slow evaluation function, n_jobs=-1 must be faster than
        n_jobs=1.  We use a 300-sample run with slow_game so each permutation
        takes ~10 ms; sequential cost ≈ 3 s while parallel should halve it
        on any multi-core machine.

        The test is skipped (not failed) if only 1 logical CPU is available,
        since you cannot speed up with parallelism on a single core.
        """
        import os
        cpu_count = os.cpu_count() or 1
        if cpu_count < 2:
            self.skipTest("Parallel speedup test requires at least 2 CPU cores")

        players = list(range(1, 6))  # 5 players, slow function
        num_samples = 300

        # Sequential baseline
        seq = MonteCarloShapleyValue(
            slow_game, players, num_samples=num_samples, random_seed=0, n_jobs=1
        )
        t0 = time.perf_counter()
        seq.calculate_shapley_values()
        t_seq = time.perf_counter() - t0

        # Parallel
        par = MonteCarloShapleyValue(
            slow_game, players, num_samples=num_samples, random_seed=0, n_jobs=-1
        )
        t0 = time.perf_counter()
        par.calculate_shapley_values()
        t_par = time.perf_counter() - t0

        speedup = t_seq / t_par if t_par > 0 else float("inf")
        self.assertGreater(
            speedup, 1.2,
            f"Expected parallel speedup > 1.2×, got {speedup:.2f}× "
            f"(seq={t_seq:.2f}s, par={t_par:.2f}s)",
        )

    def test_parallel_correctness_large_scale(self):
        """
        n_jobs=-1, 50 players, 500 samples – efficiency property must hold.
        """
        players = list(range(1, 51))
        mc = MonteCarloShapleyValue(
            fast_additive, players, num_samples=500, random_seed=0, n_jobs=-1
        )
        result = mc.calculate_shapley_values()
        grand = fast_additive(players)
        self.assertAlmostEqual(sum(result.values()), grand, places=6)

    def test_caching_avoids_recompute(self):
        """Second call to calculate_shapley_values must be nearly instant."""
        players = list(range(1, 31))
        mc = MonteCarloShapleyValue(
            fast_additive, players, num_samples=500, random_seed=0
        )
        mc.calculate_shapley_values()  # warm cache

        t0 = time.perf_counter()
        mc.calculate_shapley_values()  # should use cache
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.1,
            f"Cached call took {elapsed:.4f}s (expected < 0.1s)")


# ---------------------------------------------------------------------------
# 3. Throughput benchmark (not a pass/fail test – prints a table)
# ---------------------------------------------------------------------------

class TestStressBenchmarkTable(unittest.TestCase):
    """
    Prints a human-readable benchmark table to stdout.

    This is always-passing; its purpose is to give developers a quick overview
    of throughput at different scales and n_jobs values.

    Run with:  python -m pytest tests/test_montecarlo_stress.py -v -s
    """

    def test_benchmark_table(self):
        """Print permutation-throughput table across player counts and n_jobs."""
        import os

        configs = [
            (10,  1_000),
            (20,  500),
            (50,  200),
            (100, 100),
        ]
        cpu_count = os.cpu_count() or 1
        job_values = [1, 2, -1] if cpu_count >= 2 else [1]

        header = (
            f"\n{'Players':>8}  {'Samples':>8}  "
            + "  ".join(f"{'n_jobs='+str(j):>14}" for j in job_values)
        )
        sep = "-" * (8 + 2 + 8 + (16 * len(job_values)))

        print("\n" + "=" * len(sep))
        print(" MonteCarloShapleyValue – throughput benchmark")
        print("=" * len(sep))
        print(f" Platform: {cpu_count} logical CPU(s)")
        print("=" * len(sep))
        print(header)
        print(sep)

        for n_players, n_samples in configs:
            players = list(range(1, n_players + 1))
            row = f"{n_players:>8}  {n_samples:>8}"
            times = {}
            for n_jobs in job_values:
                mc = MonteCarloShapleyValue(
                    fast_additive, players, num_samples=n_samples,
                    random_seed=0, n_jobs=n_jobs,
                )
                t0 = time.perf_counter()
                mc.calculate_shapley_values()
                elapsed = time.perf_counter() - t0
                times[n_jobs] = elapsed

                # perms/s = num_samples / elapsed
                pps = n_samples / elapsed if elapsed > 0 else float("inf")
                row += f"  {elapsed:>7.3f}s {pps:>5.0f}p/s"

            if 1 in times and -1 in times and times[-1] > 0:
                speedup = times[1] / times[-1]
                row += f"  (speedup all-cores: {speedup:.2f}×)"

            print(row)

        print(sep)
        print(" p/s = permutations per second")
        print("=" * len(sep) + "\n")

        # Always passes – this is a diagnostic / informational test
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
