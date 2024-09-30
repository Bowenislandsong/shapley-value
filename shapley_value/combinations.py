import itertools
import math

class ShapleyCombinations:
    def __init__(self, players):
        """
        Initialize the ShapleyCombinations class.

        Args:
            players (list): List of player names.
        """
        self.players = players

    def get_subcoalitions(self, player):
        """
        Generate all subcoalitions for a given player.

        Args:
            player (str): Player name.

        Yields:
            tuple: Subcoalition tuple.
        """
        for r in range(1, len(self.players) + 1):
            for coalition in itertools.combinations(self.players, r):
                if player in coalition:
                    yield tuple(sorted(coalition))

    def get_all_coalitions(self):
        """
        Generate all possible coalitions.

        Yields:
            tuple: Coalition tuple.
        """
        for r in range(1, len(self.players) + 1):
            for coalition in itertools.combinations(self.players, r):
                yield tuple(sorted(coalition))

    def get_marginal_contributions(self, coalition_values, player):
        """
        Calculate marginal contributions for a given player.

        Args:
            coalition_values (dict): Dictionary of coalition values.
            player (str): Player name.

        Yields:
            float: Marginal contribution.
        """
        for coalition in self.get_subcoalitions(player):
            parent_coalition = tuple(sorted(coalition - {player}))
            marginal_contribution = coalition_values[coalition] - coalition_values.get(parent_coalition, 0)
            yield marginal_contribution

    def calculate_shapley_values(self, coalition_values):
        """
        Calculate Shapley values for all players.

        Args:
            coalition_values (dict): Dictionary of coalition values.

        Returns:
            dict: Dictionary of Shapley values.
        """
        shapley_values = {player: 0 for player in self.players}
        for player in self.players:
            for coalition in self.get_subcoalitions(player):
                marginal_contribution = self.get_marginal_contributions(coalition_values, player)
                weight = self.calculate_weight(len(coalition), len(self.players))
                shapley_values[player] += marginal_contribution * weight
        return shapley_values

    @staticmethod
    def calculate_weight(coalition_size, total_players):
        """
        Calculate weight for a given coalition.

        Args:
            coalition_size (int): Coalition size.
            total_players (int): Total number of players.

        Returns:
            float: Weight.
        """
        return (coalition_size - 1) * (total_players - coalition_size) / (float(total_players) * math.factorial(total_players))
