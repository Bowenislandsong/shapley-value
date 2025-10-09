"""
Shapley Value Calculator using Coalition Combinations

This module provides the ShapleyCombinations class for calculating Shapley values
when coalition values are predefined.
"""

import itertools
import math
from typing import List, Dict, Tuple, Any, Iterator, Union


class ShapleyCombinations:
    """
    Calculate Shapley values using predefined coalition values.
    
    This class is suitable when you have specific values for each possible
    coalition and want to compute fair allocations using Shapley values.
    """
    
    def __init__(self, players: List[Any]) -> None:
        """
        Initialize the ShapleyCombinations class.

        Args:
            players: List of player identifiers (can be any hashable type).
        """
        self.players = players

    def get_subcoalitions(self, player: Any) -> Iterator[Tuple[Any, ...]]:
        """
        Generate all subcoalitions containing a given player.

        Args:
            player: Player identifier.

        Yields:
            Subcoalition tuples containing the player.
        """
        for r in range(1, len(self.players) + 1):
            for coalition in itertools.combinations(self.players, r):
                if player in coalition:
                    yield tuple(sorted(coalition))

    def get_all_coalitions(self) -> Iterator[Tuple[Any, ...]]:
        """
        Generate all possible non-empty coalitions.

        Yields:
            Coalition tuples in sorted order.
        """
        for r in range(1, len(self.players) + 1):
            for coalition in itertools.combinations(self.players, r):
                yield tuple(sorted(coalition))

    def get_marginal_contributions(self, coalition_values: Dict[Tuple[Any, ...], float], player: Any) -> Iterator[float]:
        """
        Calculate marginal contributions for a given player across all coalitions.

        Args:
            coalition_values: Dictionary mapping coalition tuples to their values.
            player: Player identifier.

        Yields:
            Marginal contribution values.
        """
        for coalition in self.get_subcoalitions(player):
            parent_coalition = tuple(sorted(set(coalition) - {player}))
            marginal_contribution = coalition_values[coalition] - coalition_values.get(parent_coalition, 0)
            yield marginal_contribution

    def calculate_shapley_values(self, coalition_values: Dict[Tuple[Any, ...], float]) -> Dict[Any, float]:
        """
        Calculate Shapley values for all players.

        Args:
            coalition_values: Dictionary mapping coalition tuples to their values.
                             Must include values for all possible coalitions.

        Returns:
            Dictionary mapping each player to their Shapley value.
        """
        shapley_values = {player: 0.0 for player in self.players}
        
        for player in self.players:
            for coalition in self.get_subcoalitions(player):
                parent_coalition = tuple(sorted(set(coalition) - {player}))
                marginal_contribution = coalition_values[coalition] - coalition_values.get(parent_coalition, 0)
                weight = self.calculate_weight(len(coalition), len(self.players))
                shapley_values[player] += marginal_contribution * weight
                
        return shapley_values

    @staticmethod
    def calculate_weight(coalition_size: int, total_players: int) -> float:
        """
        Calculate weight for a given coalition in Shapley value calculation.

        The weight formula is: (|S|-1)!(n-|S|)!/n!
        where |S| is coalition size and n is total number of players.

        Args:
            coalition_size: Size of the coalition containing the player.
            total_players: Total number of players.

        Returns:
            Weight for this coalition size.
        """
        if coalition_size == 0 or coalition_size > total_players:
            return 0.0
        
        s = coalition_size
        n = total_players
        
        # Calculate (s-1)! * (n-s)! / n!
        numerator = math.factorial(s - 1) * math.factorial(n - s)
        denominator = math.factorial(n)
        
        return numerator / denominator
