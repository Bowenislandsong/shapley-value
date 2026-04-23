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
        players = [1, 2, 3]

        mc1 = MonteCarloShapleyValue(additive_game, players, num_samples=10, random_seed=1)
        mc2 = MonteCarloShapleyValue(additive_game, players, num_samples=10, random_seed=2)

        result1 = mc1.calculate_shapley_values()
        result2 = mc2.calculate_shapley_values()

        # At least one player's value should differ
        differs = any(result1[p] != result2[p] for p in players)
        self.assertTrue(differs)


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


if __name__ == "__main__":
    unittest.main()
