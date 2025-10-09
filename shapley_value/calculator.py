
"""
Basic Shapley Value Calculator

This module provides the ShapleyValue class for calculating Shapley values
with predefined coalition values using a different algorithm approach.
"""

import itertools
import math
from typing import List, Dict, Tuple, Any, Set


class ShapleyValue:
    """
    Calculate Shapley values using predefined coalition values.
    
    This is an alternative implementation to ShapleyCombinations with a
    different algorithmic approach. For most use cases, prefer ShapleyCombinations.
    """
    
    def __init__(self, players: List[Any], coalition_values: Dict[Tuple[Any, ...], float]) -> None:
        """
        Initialize the Shapley value calculator.

        Args:
            players: List of player identifiers.
            coalition_values: Dictionary mapping coalition tuples to their values.
                            Must include empty tuple () with value 0.
        """
        self.players = players
        self.coalition_values = coalition_values
        
        # Ensure empty coalition is included
        if () not in self.coalition_values:
            self.coalition_values[()] = 0.0

    def calculate_shapley_values(self) -> Dict[Any, float]:
        """
        Calculate Shapley values for all players.

        Returns:
            Dictionary mapping each player to their Shapley value.
        """
        shapley_values = {player: 0.0 for player in self.players}

        for player in self.players:
            for coalition in self._get_subcoalitions(player):
                coalition_tuple = tuple(sorted(coalition))
                parent_coalition_tuple = tuple(sorted(coalition - {player}))
                
                coalition_value = self.coalition_values.get(coalition_tuple, 0.0)
                parent_value = self.coalition_values.get(parent_coalition_tuple, 0.0)
                marginal_contribution = coalition_value - parent_value
                
                weight = self._calculate_weight(len(coalition), len(self.players))
                shapley_values[player] += marginal_contribution * weight

        return shapley_values

    def _get_subcoalitions(self, player: Any) -> List[Set[Any]]:
        """
        Generate all subcoalitions containing a given player.

        Args:
            player: Player identifier.

        Returns:
            List of sets representing coalitions containing the player.
        """
        subcoalitions = []
        for r in range(1, len(self.players) + 1):  # Fixed: include all coalition sizes
            for coalition in itertools.combinations(self.players, r):
                if player in coalition:
                    subcoalitions.append(set(coalition))
        return subcoalitions

    def _calculate_weight(self, coalition_size: int, total_players: int) -> float:
        """
        Calculate weight for a coalition in Shapley value calculation.

        Args:
            coalition_size: Size of the coalition.
            total_players: Total number of players.

        Returns:
            Weight for this coalition size.
        """
        if coalition_size == 0 or coalition_size > total_players:
            return 0.0
        
        s = coalition_size
        n = total_players
        
        # Shapley weight formula: (s-1)!(n-s)!/n!
        numerator = math.factorial(s - 1) * math.factorial(n - s)
        denominator = math.factorial(n)
        
        return numerator / denominator
