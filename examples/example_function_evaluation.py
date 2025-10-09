#!/usr/bin/env python3
"""
Function-based Evaluation Example

This example demonstrates how to use the ShapleyValueCalculator class
with custom evaluation functions. This approach is useful when coalition
values are computed dynamically rather than predefined.

Scenario: Software development team where each developer has different
productivity levels and certain combinations work better together.
"""

import time
from shapley_value import ShapleyValueCalculator


def productivity_function(coalition):
    """
    Evaluation function that calculates team productivity.
    
    Args:
        coalition: List of developer skill levels
        
    Returns:
        Total productivity score for the coalition
    """
    if not coalition:
        return 0
    
    # Base productivity is sum of individual skills
    base_productivity = sum(coalition)
    
    # Collaboration bonus increases with team size
    collaboration_bonus = len(coalition) * 5
    
    # Diminishing returns for very large teams
    if len(coalition) > 4:
        collaboration_bonus *= 0.8
    
    # Special synergy bonuses for specific combinations
    if len(coalition) >= 2:
        # High-skill pairs get extra bonus
        max_skill = max(coalition)
        min_skill = min(coalition)
        if max_skill >= 80 and min_skill >= 60:
            collaboration_bonus += 10
    
    return base_productivity + collaboration_bonus


def complex_evaluation_function(coalition):
    """
    More complex evaluation function demonstrating advanced features.
    
    This function simulates a scenario where coalition value depends on
    multiple factors: individual contributions, synergies, and overhead costs.
    """
    if not coalition:
        return 0
    
    # Individual contributions (exponential to reward high performers)
    individual_value = sum(skill ** 1.2 for skill in coalition)
    
    # Synergy effects (quadratic growth with team size)
    synergy = (len(coalition) ** 2) * 2
    
    # Communication overhead (negative effect for large teams)
    overhead = max(0, (len(coalition) - 3) * 5)
    
    # Skill diversity bonus
    if len(coalition) > 1:
        skill_range = max(coalition) - min(coalition)
        diversity_bonus = skill_range * 0.5
    else:
        diversity_bonus = 0
    
    total_value = individual_value + synergy + diversity_bonus - overhead
    return max(0, total_value)  # Ensure non-negative values


def main():
    print("=" * 70)
    print("Function-based Evaluation Example")
    print("=" * 70)
    
    # Define developers with their skill levels (0-100)
    developers = {
        'Alice': 85,   # Senior developer
        'Bob': 70,     # Mid-level developer
        'Charlie': 60, # Junior developer
        'Diana': 90    # Tech lead
    }
    
    players = list(developers.values())
    player_names = list(developers.keys())
    
    print("Development Team:")
    for name, skill in developers.items():
        print(f"  {name}: {skill} skill points")
    print()
    
    # Calculate Shapley values using productivity function
    print("Calculating Shapley values using productivity function...")
    start_time = time.time()
    
    calculator = ShapleyValueCalculator(
        evaluation_function=productivity_function,
        players=players,
        num_jobs=1  # Single-threaded for this example
    )
    
    shapley_values = calculator.calculate_shapley_values()
    calculation_time = time.time() - start_time
    
    print(f"Calculation completed in {calculation_time:.3f} seconds")
    print()
    
    # Display results with names
    print("Shapley Values (Fair Productivity Distribution):")
    total_value = sum(shapley_values.values())
    
    for i, (skill_level, shapley_value) in enumerate(shapley_values.items()):
        name = player_names[i]
        percentage = (shapley_value / total_value) * 100
        print(f"  {name} (skill {skill_level}): {shapley_value:.2f} ({percentage:.1f}%)")
    
    print(f"\nTotal productivity: {total_value:.2f}")
    print()
    
    # Show raw coalition data
    print("Raw Coalition Data (sample):")
    raw_data = calculator.get_raw_data()
    
    # Show a few interesting coalitions
    sample_coalitions = [
        (),
        (85,),
        (85, 90),
        (70, 60),
        tuple(players)
    ]
    
    for coalition in sample_coalitions:
        if coalition in raw_data:
            if not coalition:
                coalition_str = "∅ (empty)"
            elif len(coalition) == 1:
                name = player_names[players.index(coalition[0])]
                coalition_str = f"{name}"
            elif len(coalition) == len(players):
                coalition_str = "All developers"
            else:
                names = [player_names[players.index(skill)] for skill in coalition]
                coalition_str = ", ".join(names)
            
            print(f"  {{{coalition_str}}}: {raw_data[coalition]:.2f}")
    
    print("  ... (and more coalitions)")
    print()
    
    # Save raw data for analysis
    output_file = "team_productivity_analysis.csv"
    calculator.save_raw_data(output_file)
    print(f"Raw data saved to: {output_file}")


def compare_evaluation_functions():
    """
    Compare results from different evaluation functions
    """
    print("\n" + "=" * 70)
    print("Comparing Different Evaluation Functions")
    print("=" * 70)
    
    # Simple team for clearer comparison
    players = [60, 80, 100]  # Junior, Mid, Senior
    names = ['Junior', 'Mid', 'Senior']
    
    print(f"Team: {dict(zip(names, players))}")
    print()
    
    # Function 1: Simple additive
    def simple_function(coalition):
        return sum(coalition) if coalition else 0
    
    # Function 2: With collaboration bonus
    def collaboration_function(coalition):
        if not coalition:
            return 0
        return sum(coalition) + len(coalition) * 10
    
    # Function 3: Complex with synergies
    evaluation_functions = [
        ("Simple (additive)", simple_function),
        ("With collaboration bonus", collaboration_function),
        ("Complex with synergies", complex_evaluation_function)
    ]
    
    for func_name, func in evaluation_functions:
        calculator = ShapleyValueCalculator(func, players)
        shapley_values = calculator.calculate_shapley_values()
        
        print(f"{func_name}:")
        total = sum(shapley_values.values())
        for i, (skill, value) in enumerate(shapley_values.items()):
            name = names[i]
            percentage = (value / total) * 100
            print(f"  {name}: {value:.2f} ({percentage:.1f}%)")
        print(f"  Total: {total:.2f}")
        print()


def demonstrate_marginal_contributions():
    """
    Show how marginal contributions work in practice
    """
    print("\n" + "=" * 70)
    print("Understanding Marginal Contributions")
    print("=" * 70)
    
    players = [50, 80]  # Two players for simplicity
    names = ['Alice', 'Bob']
    
    def demo_function(coalition):
        if not coalition:
            return 0
        base = sum(coalition)
        # Synergy bonus for working together
        bonus = 20 if len(coalition) > 1 else 0
        return base + bonus
    
    print(f"Players: Alice (50), Bob (80)")
    print("Evaluation function: sum of skills + 20 bonus if working together")
    print()
    
    # Show all coalition values
    from itertools import combinations
    all_coalitions = []
    for r in range(len(players) + 1):
        all_coalitions.extend(combinations(players, r))
    
    print("All Coalition Values:")
    for coalition in all_coalitions:
        value = demo_function(coalition)
        if not coalition:
            coalition_str = "∅"
        elif len(coalition) == 1:
            idx = players.index(coalition[0])
            coalition_str = names[idx]
        else:
            coalition_names = [names[players.index(p)] for p in coalition]
            coalition_str = ", ".join(coalition_names)
        print(f"  {{{coalition_str}}}: {value}")
    
    print()
    print("Marginal Contributions:")
    
    # Alice's marginal contributions
    print("Alice's contributions:")
    print("  - Joining empty coalition: 50 - 0 = 50")
    print("  - Joining {Bob}: 150 - 80 = 70")
    print("  - Average: (50 + 70) / 2 = 60")
    
    print("Bob's contributions:")
    print("  - Joining empty coalition: 80 - 0 = 80")
    print("  - Joining {Alice}: 150 - 50 = 100")
    print("  - Average: (80 + 100) / 2 = 90")
    
    # Verify with calculator
    calculator = ShapleyValueCalculator(demo_function, players)
    shapley_values = calculator.calculate_shapley_values()
    
    print("\nCalculated Shapley Values:")
    for i, (skill, value) in enumerate(shapley_values.items()):
        name = names[i]
        print(f"  {name}: {value:.2f}")


if __name__ == "__main__":
    main()
    compare_evaluation_functions()
    demonstrate_marginal_contributions()