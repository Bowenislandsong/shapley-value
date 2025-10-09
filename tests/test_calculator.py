
"""
Test suite for Shapley Value Calculator

Tests all three main classes: ShapleyValue, ShapleyCombinations, and ShapleyValueCalculator
"""

import unittest
import os
import tempfile
from shapley_value import ShapleyValue, ShapleyCombinations, ShapleyValueCalculator
import pandas as pd


class TestShapleyValue(unittest.TestCase):
    """Test the basic ShapleyValue class"""
    
    def setUp(self):
        self.players = ['A', 'B', 'C']
        self.coalition_values = {
            ('A',): 10,
            ('B',): 20,
            ('C',): 30,
            ('A', 'B'): 50,
            ('A', 'C'): 60,
            ('B', 'C'): 70,
            ('A', 'B', 'C'): 100
        }
        self.shapley = ShapleyValue(self.players, self.coalition_values)

    def test_calculate_shapley_values(self):
        """Test basic Shapley value calculation"""
        shapley_values = self.shapley.calculate_shapley_values()
        
        # Check return type and structure
        self.assertIsInstance(shapley_values, dict)
        self.assertEqual(len(shapley_values), len(self.players))
        
        # Check all players are present
        for player in self.players:
            self.assertIn(player, shapley_values)
        
        # Check efficiency (sum equals grand coalition value)
        total_value = sum(shapley_values.values())
        self.assertAlmostEqual(total_value, self.coalition_values[('A', 'B', 'C')], places=2)


class TestShapleyCombinations(unittest.TestCase):
    """Test the ShapleyCombinations class"""
    
    def setUp(self):
        self.players = ['Alice', 'Bob', 'Charlie']
        self.coalition_values = {
            (): 0,
            ('Alice',): 10,
            ('Bob',): 15,
            ('Charlie',): 12,
            ('Alice', 'Bob'): 35,
            ('Alice', 'Charlie'): 30,
            ('Bob', 'Charlie'): 32,
            ('Alice', 'Bob', 'Charlie'): 60
        }
        self.calculator = ShapleyCombinations(self.players)

    def test_calculate_shapley_values(self):
        """Test Shapley value calculation with combinations"""
        shapley_values = self.calculator.calculate_shapley_values(self.coalition_values)
        
        # Check return type and structure
        self.assertIsInstance(shapley_values, dict)
        self.assertEqual(len(shapley_values), len(self.players))
        
        # Check efficiency property
        total_value = sum(shapley_values.values())
        expected_total = self.coalition_values[('Alice', 'Bob', 'Charlie')]
        self.assertAlmostEqual(total_value, expected_total, places=2)
        
        # Check individual rationality (each player gets at least their individual value)
        for player in self.players:
            individual_value = self.coalition_values[(player,)]
            self.assertGreaterEqual(shapley_values[player], individual_value)

    def test_get_all_coalitions(self):
        """Test coalition generation"""
        coalitions = list(self.calculator.get_all_coalitions())
        
        # Should generate 2^n - 1 non-empty coalitions for n players
        expected_count = 2**len(self.players) - 1
        self.assertEqual(len(coalitions), expected_count)
        
        # Check all coalitions are tuples and sorted
        for coalition in coalitions:
            self.assertIsInstance(coalition, tuple)
            self.assertEqual(coalition, tuple(sorted(coalition)))

    def test_simple_two_player_case(self):
        """Test with a simple two-player case for manual verification"""
        players = ['Player1', 'Player2']
        coalition_values = {
            (): 0,
            ('Player1',): 40,
            ('Player2',): 30,
            ('Player1', 'Player2'): 100
        }
        
        calculator = ShapleyCombinations(players)
        shapley_values = calculator.calculate_shapley_values(coalition_values)
        
        # Manual calculation: Player1 should get 55, Player2 should get 45
        self.assertAlmostEqual(shapley_values['Player1'], 55.0, places=1)
        self.assertAlmostEqual(shapley_values['Player2'], 45.0, places=1)


class TestShapleyValueCalculator(unittest.TestCase):
    """Test the ShapleyValueCalculator class with evaluation functions"""
    
    def setUp(self):
        # Simple evaluation function: sum of player values
        def evaluation_function(coalition):
            return sum(value for value in coalition) if coalition else 0

        self.players = [10, 20, 30]
        self.calculator = ShapleyValueCalculator(evaluation_function, self.players, num_jobs=1)

    def test_calculate_shapley_values(self):
        """Test Shapley value calculation with evaluation function"""
        shapley_values = self.calculator.calculate_shapley_values()
        
        # Check return type and structure
        self.assertIsInstance(shapley_values, dict)
        self.assertEqual(len(shapley_values), len(self.players))
        
        # Check all players are present
        for player in self.players:
            self.assertIn(player, shapley_values)
        
        # Check that all values are numeric and reasonable
        for value in shapley_values.values():
            self.assertIsInstance(value, (int, float))
            self.assertGreaterEqual(value, 0)  # For additive functions, should be non-negative

    def test_get_raw_data(self):
        """Test raw data generation"""
        raw_data = self.calculator.get_raw_data()
        
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)
        
        # Check DataFrame has expected columns
        expected_columns = ['Player Coalition', 'Marginal Contribution', 'Shapley Value']
        for col in expected_columns:
            self.assertIn(col, raw_data.columns)

    def test_save_raw_data(self):
        """Test saving raw data to CSV"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            file_path = tmp_file.name
        
        try:
            self.calculator.save_raw_data(file_path)
            self.assertTrue(os.path.exists(file_path))
            
            # Check file content
            df = pd.read_csv(file_path)
            self.assertGreater(len(df), 0)
            
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_parallel_vs_sequential(self):
        """Test that parallel and sequential processing give same results"""
        # Sequential calculator
        seq_calculator = ShapleyValueCalculator(
            self.calculator.evaluation_function, 
            self.players, 
            num_jobs=1
        )
        seq_values = seq_calculator.calculate_shapley_values()
        
        # Parallel calculator
        par_calculator = ShapleyValueCalculator(
            self.calculator.evaluation_function, 
            self.players, 
            num_jobs=2
        )
        par_values = par_calculator.calculate_shapley_values()
        
        # Results should be identical
        for player in self.players:
            self.assertAlmostEqual(seq_values[player], par_values[player], places=6)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_single_player(self):
        """Test with single player"""
        players = ['OnlyPlayer']
        coalition_values = {
            (): 0.0,
            ('OnlyPlayer',): 100.0
        }
        
        calculator = ShapleyCombinations(players)
        shapley_values = calculator.calculate_shapley_values(coalition_values)
        
        # Single player should get the full value
        self.assertEqual(shapley_values['OnlyPlayer'], 100)

    def test_zero_values(self):
        """Test with all zero coalition values"""
        players = ['A', 'B']
        coalition_values = {
            (): 0.0,
            ('A',): 0.0,
            ('B',): 0.0,
            ('A', 'B'): 0.0
        }
        
        calculator = ShapleyCombinations(players)
        shapley_values = calculator.calculate_shapley_values(coalition_values)
        
        # All Shapley values should be zero
        for value in shapley_values.values():
            self.assertEqual(value, 0)

    def test_negative_values(self):
        """Test with negative coalition values"""
        def negative_evaluation(coalition):
            return -sum(abs(x) for x in coalition) if coalition else 0
        
        players = [1, 2]
        calculator = ShapleyValueCalculator(negative_evaluation, players, num_jobs=1)
        shapley_values = calculator.calculate_shapley_values()
        
        # Should handle negative values correctly
        self.assertIsInstance(shapley_values, dict)
        # Just check that the calculation completes successfully with negative values
        for value in shapley_values.values():
            self.assertIsInstance(value, (int, float))


if __name__ == '__main__':
    unittest.main()
