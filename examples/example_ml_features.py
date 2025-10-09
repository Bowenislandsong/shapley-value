#!/usr/bin/env python3
"""
Machine Learning Feature Importance Example

This example demonstrates how to use Shapley values for feature importance
analysis in machine learning models. Shapley values provide a principled
way to understand which features contribute most to model predictions.

Note: This is a conceptual example showing how Shapley values work for
feature importance. For production ML feature importance, consider using
specialized libraries like SHAP (SHapley Additive exPlanations).
"""

import random
from shapley_value import ShapleyValueCalculator


def simple_ml_example():
    """
    Simple linear model example to demonstrate feature importance
    """
    print("=" * 70)
    print("FEATURE IMPORTANCE: Simple Linear Model")
    print("=" * 70)
    
    # Simulate a simple house price prediction model
    # Features: [square_feet, bedrooms, location_score, age]
    feature_names = ['Square Feet', 'Bedrooms', 'Location Score', 'Age']
    
    # Example house: 2000 sq ft, 3 bedrooms, location score 8, age 10 years
    house_features = [2000, 3, 8, 10]
    
    print("House Price Prediction Model")
    print("Features for analysis:")
    for i, (name, value) in enumerate(zip(feature_names, house_features)):
        print(f"  {name}: {value}")
    print()
    
    def house_price_model(feature_subset_indices):
        """
        Simplified house price model.
        
        Args:
            feature_subset_indices: List of feature indices to include
        
        Returns:
            Predicted house price in thousands
        """
        if not feature_subset_indices:
            return 200  # Base price with no features (market average)
        
        price = 200  # Base price
        
        # Feature contributions (simplified linear model)
        feature_weights = [0.1, 15, 10, -2]  # Price per sq ft, per bedroom, location, age penalty
        
        for idx in feature_subset_indices:
            if idx < len(house_features):
                contribution = house_features[idx] * feature_weights[idx]
                price += contribution
        
        # Feature interactions (synergies)
        if 0 in feature_subset_indices and 1 in feature_subset_indices:
            # Square feet and bedrooms together
            price += 20  # Good layout bonus
        
        if 2 in feature_subset_indices and len(feature_subset_indices) > 1:
            # Location is more valuable with other good features
            price += 15
        
        return max(0, price)
    
    # Calculate feature importance using Shapley values
    feature_indices = list(range(len(house_features)))
    calculator = ShapleyValueCalculator(house_price_model, feature_indices)
    shapley_values = calculator.calculate_shapley_values()
    
    # Display results
    print("Feature Importance (Shapley Values):")
    total_contribution = sum(shapley_values.values())
    base_price = house_price_model([])
    prediction = house_price_model(feature_indices)
    
    print(f"Base price (no features): ${base_price:.0f}K")
    print(f"Full prediction: ${prediction:.0f}K")
    print(f"Total feature contribution: ${total_contribution:.0f}K")
    print()
    
    # Sort by importance
    feature_importance = []
    for idx, importance in shapley_values.items():
        feature_importance.append((feature_names[idx], importance, idx))
    
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, importance, idx in feature_importance:
        percentage = (importance / total_contribution) * 100 if total_contribution != 0 else 0
        feature_value = house_features[idx]
        print(f"  {name} ({feature_value}):")
        print(f"    - Contribution: ${importance:.1f}K ({percentage:.1f}%)")
        print(f"    - Impact: {'Positive' if importance > 0 else 'Negative'}")


def classification_example():
    """
    Binary classification example with feature importance
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE: Binary Classification Model")
    print("=" * 70)
    
    # Email spam detection features
    feature_names = [
        'Exclamation Marks',
        'Capital Letters %',
        'Suspicious Words',
        'External Links',
        'Sender Reputation'
    ]
    
    # Example email features
    email_features = [5, 30, 3, 2, 7]  # Feature values
    
    print("Email Spam Detection Model")
    print("Email features:")
    for name, value in zip(feature_names, email_features):
        print(f"  {name}: {value}")
    print()
    
    def spam_probability(feature_subset_indices):
        """
        Calculate probability of email being spam based on features.
        
        Returns:
            Spam probability (0-100%)
        """
        if not feature_subset_indices:
            return 10  # Base spam rate (10% of all emails)
        
        spam_prob = 10  # Base probability
        
        # Feature contributions to spam probability
        feature_weights = [2, 1.5, 8, 5, -3]  # Impact on spam probability
        
        for idx in feature_subset_indices:
            if idx < len(email_features):
                contribution = email_features[idx] * feature_weights[idx]
                spam_prob += contribution
        
        # Feature interactions
        if 0 in feature_subset_indices and 1 in feature_subset_indices:
            # Exclamation marks + capital letters = very spammy
            spam_prob += 15
        
        if 2 in feature_subset_indices and 3 in feature_subset_indices:
            # Suspicious words + external links = phishing attempt
            spam_prob += 20
        
        return max(0, min(100, spam_prob))  # Clamp to 0-100%
    
    # Calculate feature importance
    feature_indices = list(range(len(email_features)))
    calculator = ShapleyValueCalculator(spam_probability, feature_indices)
    shapley_values = calculator.calculate_shapley_values()
    
    # Results
    base_prob = spam_probability([])
    full_prob = spam_probability(feature_indices)
    
    print("Feature Importance for Spam Detection:")
    print(f"Base spam probability: {base_prob:.1f}%")
    print(f"Final prediction: {full_prob:.1f}%")
    print(f"Classification: {'SPAM' if full_prob > 50 else 'NOT SPAM'}")
    print()
    
    # Sort features by absolute importance
    feature_importance = []
    for idx, importance in shapley_values.items():
        feature_importance.append((feature_names[idx], importance, idx))
    
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, importance, idx in feature_importance:
        feature_value = email_features[idx]
        impact_type = "Increases spam probability" if importance > 0 else "Decreases spam probability"
        
        print(f"  {name} ({feature_value}):")
        print(f"    - Impact: {importance:+.1f}% probability points")
        print(f"    - Effect: {impact_type}")


def feature_interaction_analysis():
    """
    Analyze feature interactions using Shapley values
    """
    print("\n" + "=" * 70)
    print("FEATURE INTERACTION ANALYSIS")
    print("=" * 70)
    
    # Simple 3-feature example for clear interaction analysis
    features = ['Feature A', 'Feature B', 'Feature C']
    feature_values = [1, 1, 1]  # Binary features for simplicity
    
    print("Analyzing feature interactions in a 3-feature model")
    print("All features have value 1 (present)")
    print()
    
    def interaction_model(feature_indices):
        """
        Model with strong feature interactions
        """
        if not feature_indices:
            return 0
        
        value = 0
        
        # Individual contributions
        individual_weights = [10, 15, 12]
        for idx in feature_indices:
            value += individual_weights[idx]
        
        # Pairwise interactions
        if 0 in feature_indices and 1 in feature_indices:
            value += 25  # Strong A-B interaction
        
        if 0 in feature_indices and 2 in feature_indices:
            value += 8   # Moderate A-C interaction
        
        if 1 in feature_indices and 2 in feature_indices:
            value += -5  # Negative B-C interaction
        
        # Three-way interaction
        if len(feature_indices) == 3:
            value += 15  # Strong three-way synergy
        
        return value
    
    # Calculate Shapley values
    feature_indices = list(range(len(features)))
    calculator = ShapleyValueCalculator(interaction_model, feature_indices)
    shapley_values = calculator.calculate_shapley_values()
    
    # Show all coalition values to understand interactions
    print("Coalition Values (showing interactions):")
    from itertools import combinations
    
    all_coalitions = []
    for r in range(len(features) + 1):
        all_coalitions.extend(combinations(feature_indices, r))
    
    for coalition in all_coalitions:
        value = interaction_model(coalition)
        if not coalition:
            coalition_str = "∅ (baseline)"
        elif len(coalition) == 1:
            coalition_str = features[coalition[0]]
        else:
            coalition_names = [features[i] for i in coalition]
            coalition_str = " + ".join(coalition_names)
        
        print(f"  {{{coalition_str}}}: {value}")
    
    print("\nShapley Values (accounting for all interactions):")
    total_value = sum(shapley_values.values())
    
    for idx, importance in shapley_values.items():
        feature_name = features[idx]
        percentage = (importance / total_value) * 100
        print(f"  {feature_name}: {importance:.2f} ({percentage:.1f}%)")
    
    print(f"\nTotal model value: {total_value:.2f}")
    
    # Compare with simple individual contributions
    print("\nComparison with simple individual weights:")
    individual_weights = [10, 15, 12]
    simple_total = sum(individual_weights)
    
    for i, weight in enumerate(individual_weights):
        feature_name = features[i]
        simple_pct = (weight / simple_total) * 100
        shapley_pct = (shapley_values[i] / total_value) * 100
        difference = shapley_pct - simple_pct
        
        print(f"  {feature_name}:")
        print(f"    - Simple weight: {weight} ({simple_pct:.1f}%)")
        print(f"    - Shapley value: {shapley_values[i]:.2f} ({shapley_pct:.1f}%)")
        print(f"    - Difference: {difference:+.1f}% (due to interactions)")


def practical_ml_tips():
    """
    Practical tips for using Shapley values in ML
    """
    print("\n" + "=" * 70)
    print("PRACTICAL ML TIPS FOR SHAPLEY VALUES")
    print("=" * 70)
    
    tips = [
        "1. COMPUTATIONAL COST:",
        "   - Shapley values require evaluating 2^n coalitions",
        "   - For large feature sets, use approximation methods",
        "   - Consider SHAP library for production use",
        "",
        "2. MODEL INTERPRETATION:",
        "   - Shapley values show average marginal contributions",
        "   - Useful for understanding feature importance",
        "   - Helps identify redundant or harmful features",
        "",
        "3. FEATURE SELECTION:",
        "   - Features with low/negative Shapley values are candidates for removal",
        "   - Consider feature interactions, not just individual importance",
        "   - Use for dimensionality reduction guidance",
        "",
        "4. MODEL DEBUGGING:",
        "   - Unexpected feature importance may indicate data leakage",
        "   - Helps identify model biases and overfitting",
        "   - Useful for validating domain expertise",
        "",
        "5. STAKEHOLDER COMMUNICATION:",
        "   - Provides mathematically principled explanations",
        "   - Helps build trust in model decisions",
        "   - Useful for regulatory compliance and auditing"
    ]
    
    for tip in tips:
        print(tip)


if __name__ == "__main__":
    simple_ml_example()
    classification_example()
    feature_interaction_analysis()
    practical_ml_tips()