"""
Test suite for MonteCarloShapleyValue
"""

import unittest

import pandas as pd

from shapley_value import MonteCarloShapleyValue
from shapley_value import ShapleyCombinations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def additive_game(coalition):
    """v(S) = sum of player values (trivially additive)."""
    return float(sum(coalition)) if coalition else 0.0


def simple_game(coalition):
    """
    Three-player game with known exact Shapley values:
      Players: A=1, B=2, C=3 (represented as integers for easy summation).
    Exact Shapley values equal each player's value (additive game).
    """
    return float(sum(coalition)) if coalition else 0.0


# Exact Shapley values for the Alice/Bob/Charlie coalition game used in
# test_calculator.py – precomputed for cross-validation.
_COALITION_VALUES = {
    (): 0,
    ("Alice",): 10,
    ("Bob",): 15,
    ("Charlie",): 12,
    ("Alice", "Bob"): 35,
    ("Alice", "Charlie"): 30,
    ("Bob", "Charlie"): 32,
    ("Alice", "Bob", "Charlie"): 60,
}


def _abc_game(coalition):
    """Evaluation function for the Alice/Bob/Charlie game."""
    key = tuple(sorted(coalition))
    return float(_COALITION_VALUES.get(key, 0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMonteCarloShapleyValueBasic(unittest.TestCase):
    """Basic correctness tests."""

    def test_returns_dict_with_all_players(self):
        players = [1, 2, 3]
        mc = MonteCarloShapleyValue(additive_game, players, num_samples=200, random_seed=0)
        result = mc.calculate_shapley_values()

        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(players))

    def test_efficiency_property(self):
        """Sum of Shapley values must equal v(grand coalition)."""
        players = [1, 2, 3]
        mc = MonteCarloShapleyValue(additive_game, players, num_samples=500, random_seed=42)
        result = mc.calculate_shapley_values()

        grand_value = additive_game(players)
        self.assertAlmostEqual(sum(result.values()), grand_value, places=6)

    def test_additive_game_converges_to_exact(self):
        """For an additive game the Shapley value equals each player's value."""
        players = [10, 20, 30]
        mc = MonteCarloShapleyValue(additive_game, players, num_samples=2000, random_seed=7)
        result = mc.calculate_shapley_values()

        for player in players:
            self.assertAlmostEqual(result[player], float(player), delta=1.5)

    def test_abc_game_converges_to_exact(self):
        """MC values should be close to exact values for the 3-player game."""
        players = ["Alice", "Bob", "Charlie"]

        # Compute exact values
        exact_calc = ShapleyCombinations(players)
        exact = exact_calc.calculate_shapley_values(_COALITION_VALUES)

        mc = MonteCarloShapleyValue(_abc_game, players, num_samples=5000, random_seed=99)
        approx = mc.calculate_shapley_values()

        for player in players:
            self.assertAlmostEqual(approx[player], exact[player], delta=2.0)

    def test_values_are_numeric(self):
        """All returned values must be floats."""
        players = [1, 2, 3]
        mc = MonteCarloShapleyValue(additive_game, players, num_samples=50, random_seed=0)
        result = mc.calculate_shapley_values()
        for v in result.values():
            self.assertIsInstance(v, float)

    def test_negative_values(self):
        """Calculator should handle games with negative values."""
        def loss_game(coalition):
            return -float(sum(coalition)) if coalition else 0.0

        players = [1, 2, 3]
        mc = MonteCarloShapleyValue(loss_game, players, num_samples=200, random_seed=0)
        result = mc.calculate_shapley_values()
        # Grand coalition sum must equal v(grand_coalition)
        self.assertAlmostEqual(sum(result.values()), loss_game(players), places=6)


class TestMonteCarloReproducibility(unittest.TestCase):
    """random_seed must produce identical results across runs."""

    def test_same_seed_same_result(self):
        players = [1, 2, 3]

        mc1 = MonteCarloShapleyValue(additive_game, players, num_samples=300, random_seed=123)
        mc2 = MonteCarloShapleyValue(additive_game, players, num_samples=300, random_seed=123)

        result1 = mc1.calculate_shapley_values()
        result2 = mc2.calculate_shapley_values()

        for player in players:
            self.assertEqual(result1[player], result2[player])

    def test_different_seeds_may_differ(self):
        """With a small sample count different seeds should (very likely) differ."""
        # Use a non-additive game so that permutation order actually matters.
        # v(S) = len(S)^2 is superadditive, making marginal contributions
        # depend on the permutation order.
        def synergy_game(coalition):
            return float(len(coalition) ** 2) if coalition else 0.0

        players = [1, 2, 3, 4]

        mc1 = MonteCarloShapleyValue(synergy_game, players, num_samples=10, random_seed=1)
        mc2 = MonteCarloShapleyValue(synergy_game, players, num_samples=10, random_seed=2)

        result1 = mc1.calculate_shapley_values()
        result2 = mc2.calculate_shapley_values()

        # At least one player's value should differ
        differs = any(result1[p] != result2[p] for p in players)
        self.assertTrue(differs)

    def test_seed_reproducible_with_parallel(self):
        """Same seed must give identical results regardless of n_jobs."""
        players = [1, 2, 3]

        seq = MonteCarloShapleyValue(
            additive_game, players, num_samples=300, random_seed=77, n_jobs=1
        )
        par = MonteCarloShapleyValue(
            additive_game, players, num_samples=300, random_seed=77, n_jobs=2
        )

        result_seq = seq.calculate_shapley_values()
        result_par = par.calculate_shapley_values()

        for player in players:
            self.assertAlmostEqual(result_seq[player], result_par[player], places=10)


class TestMonteCarloParallelProcessing(unittest.TestCase):
    """Tests for n_jobs / parallel processing."""

    def _make_mc(self, n_jobs, seed=42):
        return MonteCarloShapleyValue(
            additive_game, [1, 2, 3], num_samples=300, random_seed=seed, n_jobs=n_jobs
        )

    def test_sequential_vs_parallel_two_jobs(self):
        """n_jobs=1 and n_jobs=2 with same seed must give identical results."""
        seq = self._make_mc(n_jobs=1)
        par = self._make_mc(n_jobs=2)
        r_seq = seq.calculate_shapley_values()
        r_par = par.calculate_shapley_values()
        for player in [1, 2, 3]:
            self.assertAlmostEqual(r_seq[player], r_par[player], places=10)

    def test_all_cores(self):
        """n_jobs=-1 (all cores) must produce valid output."""
        mc = self._make_mc(n_jobs=-1)
        result = mc.calculate_shapley_values()
        self.assertEqual(set(result.keys()), {1, 2, 3})
        self.assertAlmostEqual(sum(result.values()), additive_game([1, 2, 3]), places=6)

    def test_parallel_efficiency_property(self):
        """Efficiency must hold for parallel runs too."""
        mc = self._make_mc(n_jobs=2)
        result = mc.calculate_shapley_values()
        self.assertAlmostEqual(sum(result.values()), additive_game([1, 2, 3]), places=6)

    def test_n_jobs_stored(self):
        mc = MonteCarloShapleyValue(additive_game, [1], num_samples=5, n_jobs=3)
        self.assertEqual(mc.n_jobs, 3)


class TestMonteCarloConvergenceData(unittest.TestCase):
    """Tests for get_convergence_data()."""

    def setUp(self):
        self.players = [1, 2, 3]
        self.num_samples = 100
        self.mc = MonteCarloShapleyValue(
            additive_game, self.players, num_samples=self.num_samples, random_seed=0
        )

    def test_returns_dataframe(self):
        df = self.mc.get_convergence_data()
        self.assertIsInstance(df, pd.DataFrame)

    def test_shape(self):
        df = self.mc.get_convergence_data()
        self.assertEqual(len(df), self.num_samples)
        self.assertEqual(set(df.columns), set(self.players))

    def test_last_row_matches_calculate(self):
        """The final running average must equal calculate_shapley_values()."""
        df = self.mc.get_convergence_data()
        final_row = df.iloc[-1].to_dict()
        calculated = self.mc.calculate_shapley_values()

        for player in self.players:
            self.assertAlmostEqual(final_row[player], calculated[player], places=10)

    def test_variance_decreases(self):
        """Running estimates should become more stable (lower std-dev) over time."""
        players = [1, 2, 3]
        mc = MonteCarloShapleyValue(additive_game, players, num_samples=500, random_seed=5)
        df = mc.get_convergence_data()

        # Compare std of values in the first 50 rows vs last 50 rows for player 1
        early_std = df[1].iloc[:50].std()
        late_std = df[1].iloc[-50:].std()
        self.assertLessEqual(late_std, early_std)

    def test_convergence_data_first_row_is_first_permutation(self):
        """Row 0 should be the marginal contribution from the first permutation."""
        df = self.mc.get_convergence_data()
        # After 1 sample the running mean equals the first permutation's contributions.
        first_row = df.iloc[0].to_dict()
        first_estimate = self.mc._iteration_estimates[0]
        for player in self.players:
            self.assertAlmostEqual(first_row[player], first_estimate[player], places=10)


class TestMonteCarloGetRawData(unittest.TestCase):
    """Tests for get_raw_data()."""

    def setUp(self):
        self.players = [1, 2, 3]
        self.num_samples = 50
        self.mc = MonteCarloShapleyValue(
            additive_game, self.players, num_samples=self.num_samples, random_seed=0
        )

    def test_returns_dataframe(self):
        df = self.mc.get_raw_data()
        self.assertIsInstance(df, pd.DataFrame)

    def test_columns(self):
        df = self.mc.get_raw_data()
        for col in ("iteration", "permutation", "player", "marginal_contribution"):
            self.assertIn(col, df.columns)

    def test_row_count(self):
        """One row per player per permutation."""
        df = self.mc.get_raw_data()
        self.assertEqual(len(df), self.num_samples * len(self.players))

    def test_marginal_contributions_sum_to_grand_value(self):
        """Within each permutation, contributions must sum to v(grand coalition)."""
        df = self.mc.get_raw_data()
        grand_value = additive_game(self.players)
        for iteration, group in df.groupby("iteration"):
            self.assertAlmostEqual(
                group["marginal_contribution"].sum(), grand_value, places=6
            )

    def test_iteration_column_values(self):
        """Iteration column must contain values 0 … num_samples-1."""
        df = self.mc.get_raw_data()
        self.assertEqual(set(df["iteration"].unique()), set(range(self.num_samples)))

    def test_all_players_appear_in_every_iteration(self):
        """Every player must appear exactly once per iteration."""
        df = self.mc.get_raw_data()
        for iteration, group in df.groupby("iteration"):
            self.assertEqual(set(group["player"].tolist()), set(self.players))


class TestMonteCarloEdgeCases(unittest.TestCase):
    """Edge cases."""

    def test_single_player(self):
        players = ["Solo"]
        mc = MonteCarloShapleyValue(
            lambda c: 42.0 if c else 0.0, players, num_samples=10, random_seed=0
        )
        result = mc.calculate_shapley_values()
        self.assertAlmostEqual(result["Solo"], 42.0, places=6)

    def test_zero_value_game(self):
        players = [1, 2, 3]
        mc = MonteCarloShapleyValue(
            lambda c: 0.0, players, num_samples=100, random_seed=0
        )
        result = mc.calculate_shapley_values()
        for v in result.values():
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_caching_consistency(self):
        """Calling calculate_shapley_values twice must return the same result."""
        players = [1, 2, 3]
        mc = MonteCarloShapleyValue(additive_game, players, num_samples=200, random_seed=0)
        first = mc.calculate_shapley_values()
        second = mc.calculate_shapley_values()
        for player in players:
            self.assertEqual(first[player], second[player])

    def test_two_players(self):
        """Two-player game: MC should give values near the exact result."""
        players = ["P1", "P2"]
        cv = {(): 0, ("P1",): 40, ("P2",): 30, ("P1", "P2"): 100}

        def game(c):
            return float(cv.get(tuple(sorted(c)), 0))

        mc = MonteCarloShapleyValue(game, players, num_samples=2000, random_seed=0)
        result = mc.calculate_shapley_values()
        # Exact: P1=55, P2=45
        self.assertAlmostEqual(result["P1"], 55.0, delta=2.0)
        self.assertAlmostEqual(result["P2"], 45.0, delta=2.0)

    def test_string_players(self):
        """Player identifiers can be strings."""
        players = ["Alice", "Bob"]
        mc = MonteCarloShapleyValue(
            lambda c: float(len(c)), players, num_samples=100, random_seed=0
        )
        result = mc.calculate_shapley_values()
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)


if __name__ == "__main__":
    unittest.main()
