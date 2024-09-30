import itertools
import math
import pandas as pd
from typing import Callable, List, Any
from joblib import Parallel, delayed

class ShapleyValueCalculator:
    def __init__(self, evaluation_function: Callable[[List[Any]], float], players: List[Any], num_jobs: int = -1):
        """
        Initialize the Shapley value calculator.

        Args:
        - evaluation_function: Function that takes a coalition (list of players) as input and returns its value.
        - players: List of players.
        - num_jobs: Number of parallel jobs. Default: -1 (use all available CPUs).
        """
        self.evaluation_function = evaluation_function
        self.players = players
        self.num_players = len(players)
        self.num_jobs = num_jobs

    def calculate_shapley_values(self) -> dict:
        """
        Calculate the Shapley value for each player.

        Returns:
        - A dictionary where keys are players and values are their corresponding Shapley values.
        """
        shapley_values = {player: 0 for player in self.players}
        
        if self.num_players < 10:  # Threshold for parallel processing
            results = [self.process_coalition(coalition) for coalition in self.generate_coalitions()]
        else:
            results = Parallel(n_jobs=self.num_jobs)(delayed(self.process_coalition)(coalition) for coalition in self.generate_coalitions())
        
        for result in results:
            for player, value in result.items():
                shapley_values[player] += value
        
        return shapley_values

    def generate_coalitions(self):
        """
        Generate all possible coalitions.

        Yields:
        - Coalition (tuple of players)
        """
        for i in range(self.num_players + 1):
            yield from itertools.combinations(self.players, i)

    def process_coalition(self, coalition):
        """
        Process a single coalition.

        Args:
        - coalition: Coalition to process.

        Returns:
        - A dictionary where keys are players and values are their marginal contributions.
        """
        try:
            coalition_value = self.evaluation_function(list(coalition))
            result = {}
            for player in coalition:
                marginal_contribution = coalition_value - self.evaluation_function(list(set(coalition) - {player}))
                result[player] = (marginal_contribution / (self.num_players * (self.num_players - 1))) * (
                    math.comb(self.num_players - 1, len(coalition) - 1) / math.comb(self.num_players, len(coalition))
                )
            return result
        except Exception as e:
            print(f"Error processing coalition {coalition}: {str(e)}")
            return {}

    def get_raw_data(self) -> pd.DataFrame:
        """
        Get the raw data used to calculate the Shapley values.

        Returns:
        - A Pandas DataFrame containing the coalition, player, marginal contribution, and Shapley value.
        """
        data = []
        for coalition in self.generate_coalitions():
            coalition_value = self.evaluation_function(list(coalition))
            for player in coalition:
                marginal_contribution = coalition_value - self.evaluation_function(list(set(coalition) - {player}))
                shapley_value = (marginal_contribution / (self.num_players * (self.num_players - 1))) * (
                    math.comb(self.num_players - 1, len(coalition) - 1) / math.comb(self.num_players, len(coalition))
                )
                data.append({
                    'Player Coalition': str(coalition),
                    'Marginal Contribution': marginal_contribution,
                    'Shapley Value': shapley_value
                })
        return pd.DataFrame(data)

    def save_raw_data(self, file_path: str) -> None:
        """
        Save the raw data to a CSV file.

        Args:
        - file_path: Path to save the CSV file.
        """
        raw_data = self.get_raw_data()
        raw_data.to_csv(file_path, index=False)


if __name__ == "__main__":
    # Example usage
    def evaluation_function(coalition):
        # Example evaluation function: sum of player values
        return sum(value for value in coalition)

    players = [10, 20, 30]
    calculator = ShapleyValueCalculator(evaluation_function, players, num_jobs=-1)
    shapley_values = calculator.calculate_shapley_values()
    print(shapley_values)

    raw_data = calculator.get_raw_data()
    print(raw_data)

    calculator.save_raw_data('sample_shapley_raw_data.csv')