#!/usr/bin/env python3
"""
Parallel Processing Example

This example demonstrates the performance benefits of parallel processing
when calculating Shapley values for large games (many players).

The ShapleyValueCalculator automatically switches to parallel processing
for games with more than 10 players, but you can control this behavior
explicitly using the num_jobs parameter.
"""

import time
import random
from shapley_value import ShapleyValueCalculator


def performance_comparison():
    """
    Compare sequential vs parallel processing performance
    """
    print("=" * 70)
    print("PERFORMANCE COMPARISON: Sequential vs Parallel")
    print("=" * 70)
    
    def compute_intensive_function(coalition):
        """
        Simulate a computationally intensive evaluation function.
        
        This function includes artificial delay to simulate real-world
        scenarios where coalition evaluation is expensive (e.g., running
        ML models, database queries, complex calculations).
        """
        if not coalition:
            return 0
        
        # Simulate some computation time
        time.sleep(0.001)  # 1ms delay per evaluation
        
        # Complex calculation with interactions
        base_value = sum(player ** 1.2 for player in coalition)
        
        # Interaction effects
        if len(coalition) > 1:
            synergy = len(coalition) * sum(coalition) * 0.1
            base_value += synergy
        
        # Diminishing returns for large coalitions
        if len(coalition) > 5:
            base_value *= 0.9
        
        return base_value
    
    # Test with different numbers of players
    player_counts = [8, 10, 12]
    
    for num_players in player_counts:
        print(f"\nTesting with {num_players} players:")
        print("-" * 40)
        
        # Generate random player values
        players = [random.randint(10, 100) for _ in range(num_players)]
        total_coalitions = 2 ** num_players
        
        print(f"Total coalitions to evaluate: {total_coalitions:,}")
        
        # Sequential processing (num_jobs=1)
        print("Sequential processing...")
        start_time = time.time()
        calculator_seq = ShapleyValueCalculator(
            compute_intensive_function, 
            players, 
            num_jobs=1
        )
        shapley_seq = calculator_seq.calculate_shapley_values()
        seq_time = time.time() - start_time
        
        # Parallel processing (num_jobs=-1, use all cores)
        print("Parallel processing...")
        start_time = time.time()
        calculator_par = ShapleyValueCalculator(
            compute_intensive_function, 
            players, 
            num_jobs=-1
        )
        shapley_par = calculator_par.calculate_shapley_values()
        par_time = time.time() - start_time
        
        # Results
        speedup = seq_time / par_time if par_time > 0 else 0
        
        print(f"Sequential time: {seq_time:.2f} seconds")
        print(f"Parallel time: {par_time:.2f} seconds")
        print(f"Speedup: {speedup:.1f}x")
        
        # Verify results are identical (within floating point precision)
        max_diff = max(abs(shapley_seq[player] - shapley_par[player]) 
                      for player in players)
        print(f"Maximum difference between methods: {max_diff:.10f}")
        print("✓ Results are identical" if max_diff < 1e-10 else "⚠ Results differ")


def scalability_analysis():
    """
    Analyze how computation time scales with number of players
    """
    print("\n" + "=" * 70)
    print("SCALABILITY ANALYSIS")
    print("=" * 70)
    
    def simple_function(coalition):
        """Simpler function for scalability testing"""
        if not coalition:
            return 0
        return sum(coalition) + len(coalition) * 5
    
    player_counts = [5, 8, 10, 12, 15]
    
    print("Analyzing computation time vs number of players:")
    print("(Using parallel processing)")
    print()
    
    results = []
    
    for num_players in player_counts:
        players = list(range(1, num_players + 1))
        total_coalitions = 2 ** num_players
        
        start_time = time.time()
        calculator = ShapleyValueCalculator(simple_function, players, num_jobs=-1)
        shapley_values = calculator.calculate_shapley_values()
        computation_time = time.time() - start_time
        
        results.append((num_players, total_coalitions, computation_time))
        
        print(f"{num_players:2d} players: {total_coalitions:6,} coalitions, "
              f"{computation_time:.3f}s")
    
    print("\nScaling characteristics:")
    for i in range(1, len(results)):
        prev_players, prev_coalitions, prev_time = results[i-1]
        curr_players, curr_coalitions, curr_time = results[i]
        
        player_ratio = curr_players / prev_players
        time_ratio = curr_time / prev_time if prev_time > 0 else 0
        coalition_ratio = curr_coalitions / prev_coalitions
        
        print(f"{prev_players}→{curr_players} players: "
              f"{player_ratio:.1f}x players, "
              f"{coalition_ratio:.1f}x coalitions, "
              f"{time_ratio:.1f}x time")


def memory_efficiency_demo():
    """
    Demonstrate memory-efficient coalition generation
    """
    print("\n" + "=" * 70)
    print("MEMORY EFFICIENCY DEMONSTRATION")
    print("=" * 70)
    
    def memory_test_function(coalition):
        """Function for testing memory usage"""
        return sum(x**2 for x in coalition) if coalition else 0
    
    num_players = 16  # 65,536 coalitions
    players = list(range(1, num_players + 1))
    
    print(f"Testing with {num_players} players ({2**num_players:,} coalitions)")
    print()
    
    # The ShapleyValueCalculator generates coalitions on-demand
    # rather than storing all coalitions in memory at once
    
    start_time = time.time()
    calculator = ShapleyValueCalculator(memory_test_function, players, num_jobs=-1)
    
    print("Calculating Shapley values...")
    shapley_values = calculator.calculate_shapley_values()
    computation_time = time.time() - start_time
    
    print(f"Computation completed in {computation_time:.2f} seconds")
    print("✓ Memory-efficient: coalitions generated on-demand")
    
    # Show some results
    total_value = sum(shapley_values.values())
    print(f"\nTotal Shapley value: {total_value:.2f}")
    print("Top 5 contributors:")
    
    sorted_players = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    for i, (player, value) in enumerate(sorted_players[:5]):
        percentage = (value / total_value) * 100
        print(f"  Player {player}: {value:.2f} ({percentage:.1f}%)")


def optimal_parallelization_demo():
    """
    Show how to optimize parallelization for different scenarios
    """
    print("\n" + "=" * 70)
    print("OPTIMAL PARALLELIZATION STRATEGIES")
    print("=" * 70)
    
    def variable_complexity_function(coalition):
        """Function with variable computational complexity"""
        if not coalition:
            return 0
        
        # Complexity varies with coalition size
        complexity = len(coalition)
        
        # Simulate variable computation time
        for _ in range(complexity * 100):
            _ = sum(x * 1.1 for x in coalition)
        
        return sum(coalition) * len(coalition)
    
    players = list(range(1, 13))  # 12 players
    
    print(f"Testing different parallelization strategies with {len(players)} players:")
    print()
    
    # Test different num_jobs values
    job_configs = [
        (1, "Sequential"),
        (2, "2 processes"),
        (4, "4 processes"),
        (-1, "All available cores")
    ]
    
    results = []
    
    for num_jobs, description in job_configs:
        start_time = time.time()
        
        calculator = ShapleyValueCalculator(
            variable_complexity_function, 
            players, 
            num_jobs=num_jobs
        )
        shapley_values = calculator.calculate_shapley_values()
        
        computation_time = time.time() - start_time
        results.append((description, computation_time))
        
        print(f"{description:20s}: {computation_time:.2f}s")
    
    # Find optimal configuration
    best_config = min(results, key=lambda x: x[1])
    print(f"\nOptimal configuration: {best_config[0]} ({best_config[1]:.2f}s)")


def practical_performance_tips():
    """
    Practical tips for optimizing Shapley value calculations
    """
    print("\n" + "=" * 70)
    print("PRACTICAL PERFORMANCE OPTIMIZATION TIPS")
    print("=" * 70)
    
    tips = [
        "1. CHOOSE THE RIGHT APPROACH:",
        "   - Use ShapleyCombinations for pre-computed coalition values",
        "   - Use ShapleyValueCalculator for dynamic evaluation functions",
        "",
        "2. PARALLELIZATION GUIDELINES:",
        "   - Games with ≤10 players: Sequential is often faster (less overhead)",
        "   - Games with >10 players: Use parallel processing",
        "   - CPU-bound evaluation functions: Use num_jobs=-1 (all cores)",
        "   - I/O-bound evaluation functions: Use fewer processes",
        "",
        "3. EVALUATION FUNCTION OPTIMIZATION:",
        "   - Cache expensive computations within your evaluation function",
        "   - Avoid unnecessary work for empty coalitions",
        "   - Use efficient data structures and algorithms",
        "",
        "4. MEMORY CONSIDERATIONS:",
        "   - The library generates coalitions on-demand (memory efficient)",
        "   - Large games (>20 players) may still require significant computation",
        "   - Consider approximation methods for very large games",
        "",
        "5. WHEN TO USE APPROXIMATIONS:",
        "   - Games with >20 players become computationally expensive",
        "   - Consider sampling-based approaches for very large games",
        "   - Monte Carlo Shapley values for approximate solutions",
        "",
        "6. MONITORING AND PROFILING:",
        "   - Profile your evaluation function separately",
        "   - Monitor memory usage for large games",
        "   - Consider timeout mechanisms for long-running calculations"
    ]
    
    for tip in tips:
        print(tip)


if __name__ == "__main__":
    print("Note: This example includes artificial delays to demonstrate")
    print("performance differences. Real-world performance will vary")
    print("based on your evaluation function complexity.")
    print()
    
    performance_comparison()
    scalability_analysis()
    memory_efficiency_demo()
    optimal_parallelization_demo()
    practical_performance_tips()