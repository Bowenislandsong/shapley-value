"""
Monte Carlo Shapley Value Calculator

This module provides the MonteCarloShapleyValue class for approximating
Shapley values via random permutation sampling. The exact computation
requires evaluating 2^n coalitions; this approach reduces the cost to
O(m * n) evaluations where m is the number of sampled permutations.

Algorithm (permutation sampling):
    For each of m iterations:
    1. Draw a random permutation of all players.
    2. For each player at position k in the permutation, compute their
       marginal contribution: v(players[0..k]) - v(players[0..k-1]).
    3. Accumulate contributions and divide by m for the final estimate.
"""

import random
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


class MonteCarloShapleyValue:
    """
    Approximate Shapley values using Monte Carlo permutation sampling.

    This is suitable for games with many players where the exact computation
    (which enumerates all 2^n coalitions) is computationally infeasible.
    """

    def __init__(
        self,
        evaluation_function: Callable[[List[Any]], float],
        players: List[Any],
        num_samples: int = 1000,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Monte Carlo Shapley value calculator.

        Args:
            evaluation_function: Function that takes a list of players (coalition)
                and returns the coalition's value. Must return 0 for an empty list.
            players: List of player identifiers.
            num_samples: Number of random permutations to sample. Higher values
                give more accurate estimates at the cost of more evaluations.
                Defaults to 1000.
            random_seed: Seed for the random number generator. Set this for
                reproducible results. Defaults to None (non-deterministic).
        """
        self.evaluation_function = evaluation_function
        self.players = players
        self.num_samples = num_samples
        self.random_seed = random_seed

        self._rng = random.Random(random_seed)

        # Cache for sampled permutations and per-iteration estimates, populated
        # lazily by _run_sampling().
        self._permutations: List[List[Any]] = []
        self._iteration_estimates: List[Dict[Any, float]] = []
        self._sampled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_shapley_values(self) -> Dict[Any, float]:
        """
        Approximate Shapley values for all players.

        Returns:
            Dictionary mapping each player to their estimated Shapley value.
        """
        self._run_sampling()

        totals: Dict[Any, float] = {player: 0.0 for player in self.players}
        for estimate in self._iteration_estimates:
            for player, value in estimate.items():
                totals[player] += value

        return {player: totals[player] / self.num_samples for player in self.players}

    def get_convergence_data(self) -> pd.DataFrame:
        """
        Return the running (cumulative) Shapley estimate after each sampled
        permutation.

        This is useful for diagnosing how many samples are needed: plot each
        player's column against the iteration index to see when the estimates
        stabilise.

        Returns:
            A DataFrame with one row per sampled permutation and one column per
            player. The value in row i for player p is the average marginal
            contribution of p over the first i+1 permutations.
        """
        self._run_sampling()

        running_totals: Dict[Any, float] = {player: 0.0 for player in self.players}
        rows = []
        for i, estimate in enumerate(self._iteration_estimates):
            for player, value in estimate.items():
                running_totals[player] += value
            rows.append(
                {player: running_totals[player] / (i + 1) for player in self.players}
            )

        return pd.DataFrame(rows)

    def get_raw_data(self) -> pd.DataFrame:
        """
        Return detailed per-permutation marginal contribution data.

        Returns:
            A DataFrame with columns:
            - ``iteration``: permutation index (0-based)
            - ``permutation``: string representation of the sampled order
            - ``player``: player identifier
            - ``marginal_contribution``: player's marginal contribution in this
              permutation
        """
        self._run_sampling()

        rows = []
        for i, (perm, estimate) in enumerate(
            zip(self._permutations, self._iteration_estimates)
        ):
            for player, contrib in estimate.items():
                rows.append(
                    {
                        "iteration": i,
                        "permutation": str(perm),
                        "player": player,
                        "marginal_contribution": contrib,
                    }
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_sampling(self) -> None:
        """Draw permutations and compute per-permutation marginal contributions.

        Results are cached so subsequent calls to the public methods are free.
        """
        if self._sampled:
            return

        for _ in range(self.num_samples):
            permutation = list(self.players)
            self._rng.shuffle(permutation)
            self._permutations.append(permutation)
            self._iteration_estimates.append(
                self._marginal_contributions_for_permutation(permutation)
            )

        self._sampled = True

    def _marginal_contributions_for_permutation(
        self, permutation: List[Any]
    ) -> Dict[Any, float]:
        """Compute each player's marginal contribution for one permutation.

        Args:
            permutation: An ordered list of all players.

        Returns:
            Dictionary mapping each player to their marginal contribution in
            this permutation.
        """
        contributions: Dict[Any, float] = {}
        coalition: List[Any] = []

        previous_value = self.evaluation_function([])

        for player in permutation:
            coalition.append(player)
            current_value = self.evaluation_function(list(coalition))
            contributions[player] = current_value - previous_value
            previous_value = current_value

        return contributions
