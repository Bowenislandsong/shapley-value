#!/usr/bin/env python3
"""
Basic Coalition Values Example

This example demonstrates how to use the ShapleyCombinations class
when you have predefined values for each possible coalition.

Scenario: Three friends (Alice, Bob, Charlie) are planning a business venture.
Each person brings different skills and resources, and certain combinations
work better together than others.
"""

from shapley_value import ShapleyCombinations


def main():
    print("=" * 60)
    print("Basic Coalition Values Example")
    print("=" * 60)
    
    # Define the players
    players = ['Alice', 'Bob', 'Charlie']
    
    print(f"Players: {players}")
    print()
    
    # Define coalition values (expected profit in thousands)
    # Each coalition has a specific value based on synergies
    coalition_values = {
        (): 0,                           # Empty coalition has no value
        ('Alice',): 10,                  # Alice alone: $10k (marketing skills)
        ('Bob',): 15,                    # Bob alone: $15k (technical skills)
        ('Charlie',): 12,                # Charlie alone: $12k (business connections)
        ('Alice', 'Bob'): 35,            # Alice + Bob: $35k (marketing + tech)
        ('Alice', 'Charlie'): 30,        # Alice + Charlie: $30k (marketing + connections)
        ('Bob', 'Charlie'): 32,          # Bob + Charlie: $32k (tech + connections)
        ('Alice', 'Bob', 'Charlie'): 60  # All three: $60k (complete team synergy)
    }
    
    print("Coalition Values (in thousands $):")
    for coalition, value in coalition_values.items():
        if not coalition:
            coalition_str = "∅ (empty)"
        else:
            coalition_str = ", ".join(coalition)
        print(f"  {{{coalition_str}}}: ${value}k")
    print()
    
    # Calculate Shapley values
    calculator = ShapleyCombinations(players)
    shapley_values = calculator.calculate_shapley_values(coalition_values)
    
    print("Shapley Values (Fair Distribution):")
    total_value = sum(shapley_values.values())
    
    for player, value in shapley_values.items():
        percentage = (value / total_value) * 100
        print(f"  {player}: ${value:.2f}k ({percentage:.1f}%)")
    
    print(f"\nTotal distributed: ${total_value:.2f}k")
    print(f"Grand coalition value: ${coalition_values[('Alice', 'Bob', 'Charlie')]}k")
    print()
    
    # Interpretation
    print("Interpretation:")
    print("- The Shapley value represents each player's fair share based on")
    print("  their marginal contributions across all possible coalitions")
    print("- Bob receives the highest share due to high individual value")
    print("  and strong synergies with others")
    print("- All values sum to the grand coalition total (efficiency)")
    print("- Each player receives at least their individual contribution")


def example_with_explanation():
    """
    Detailed example showing step-by-step calculation explanation
    """
    print("\n" + "=" * 60)
    print("Detailed Calculation Example")
    print("=" * 60)
    
    # Simpler example for clearer explanation
    players = ['Player1', 'Player2']
    coalition_values = {
        (): 0,
        ('Player1',): 40,
        ('Player2',): 30,
        ('Player1', 'Player2'): 100
    }
    
    print(f"Players: {players}")
    print("Coalition Values:")
    for coalition, value in coalition_values.items():
        coalition_str = ", ".join(coalition) if coalition else "∅"
        print(f"  {{{coalition_str}}}: {value}")
    
    calculator = ShapleyCombinations(players)
    shapley_values = calculator.calculate_shapley_values(coalition_values)
    
    print("\nShapley Values:")
    for player, value in shapley_values.items():
        print(f"  {player}: {value}")
    
    print("\nManual Calculation for Player1:")
    print("- When Player1 joins empty coalition: 40 - 0 = 40")
    print("- When Player1 joins {Player2}: 100 - 30 = 70")
    print("- Average marginal contribution: (40 + 70) / 2 = 55")
    print("\nManual Calculation for Player2:")
    print("- When Player2 joins empty coalition: 30 - 0 = 30")
    print("- When Player2 joins {Player1}: 100 - 40 = 60")
    print("- Average marginal contribution: (30 + 60) / 2 = 45")
    print("\nVerification: 55 + 45 = 100 ✓")


if __name__ == "__main__":
    main()
    example_with_explanation()