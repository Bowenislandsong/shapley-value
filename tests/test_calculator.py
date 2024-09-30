
import unittest
from shapley_value import ShapleyValue
from shapley_value import ShapleyValueCalculator
import pandas as pd
import os

class TestShapleyValue(unittest.TestCase):
    def test_calculate_shapley_values(self):
        players = ['A', 'B', 'C']
        coalition_values = {
            ('A',): 10,
            ('B',): 20,
            ('C',): 30,
            ('A', 'B'): 50,
            ('A', 'C'): 60,
            ('B', 'C'): 70,
            ('A', 'B', 'C'): 100
        }
        shapley = ShapleyValue(players, coalition_values)
        shapley_values = shapley.calculate_shapley_values()
        self.assertIsInstance(shapley_values, dict)
        self.assertEqual(len(shapley_values), len(players))

class TestShapleyValue(unittest.TestCase):
    def setUp(self):
        # Example usage
        def evaluation_function(coalition):
            # Example evaluation function: sum of player values
            return sum(value for value in coalition)

        self.players = [10, 20, 30]
        self.calculator = ShapleyValueCalculator(evaluation_function, self.players, num_jobs=-1)

    def test_calculate_shapley_values(self):
        shapley_values = self.calculator.calculate_shapley_values()
        self.assertIsInstance(shapley_values, dict)
        self.assertEqual(len(shapley_values), len(self.players))

    def test_get_raw_data(self):
        raw_data = self.calculator.get_raw_data()
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)

    def test_save_raw_data(self):
        file_path = 'sample_shapley_raw_data.csv'
        self.calculator.save_raw_data(file_path)
        self.assertTrue(os.path.exists(file_path))

if __name__ == '__main__':
    unittest.main()