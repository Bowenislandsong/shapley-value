
class ShapleyValue:
    def __init__(self, players, coalition_values):
        self.players = players
        self.coalition_values = coalition_values

    def calculate_shapley_values(self):
        shapley_values = {player: 0 for player in self.players}

        for player in self.players:
            for coalition in self._get_subcoalitions(player):
                coalition_value = self.coalition_values[tuple(sorted(coalition))]
                marginal_contribution = coalition_value - self.coalition_values[tuple(sorted(coalition - {player}))]
                weight = self._calculate_weight(len(coalition), len(self.players))
                shapley_values[player] += marginal_contribution * weight

        return shapley_values

    def _get_subcoalitions(self, player):
        for r in range(1, len(self.players)):
            for coalition in self._combinations(self.players, r):
                if player in coalition:
                    yield set(coalition)

    def _combinations(self, items, r):
        if r == 0:
            yield []
        else:
            for i in range(len(items)):
                for combination in self._combinations(items[i+1:], r-1):
                    yield [items[i]] + combination

    def _calculate_weight(self, coalition_size, total_players):
        return (coalition_size - 1) * (total_players - coalition_size) / (float(total_players) * self._factorial(total_players))

    def _factorial(self, n):
        if n == 0:
            return 1
        else:
            return n * self._factorial(n-1)
