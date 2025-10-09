# Shapley Value Calculator - Examples

This directory contains comprehensive examples demonstrating various applications and use cases of the Shapley Value Calculator package. Each example is self-contained and includes detailed explanations and practical scenarios.

## 📚 Available Examples

### 1. **Basic Coalition Values** (`example_basic_coalition.py`)
**Difficulty:** Beginner  
**Use Case:** When you have predefined values for each coalition

Learn the fundamentals of Shapley value calculation using the `ShapleyCombinations` class. This example covers:
- Setting up coalition values for a simple 3-player game
- Understanding fair distribution principles
- Step-by-step calculation explanation
- Business scenario: Three friends planning a venture

**Key Concepts:**
- Coalition value dictionaries
- Marginal contributions
- Fair allocation principles

```bash
python examples/example_basic_coalition.py
```

### 2. **Function-based Evaluation** (`example_function_evaluation.py`)
**Difficulty:** Intermediate  
**Use Case:** When coalition values are computed dynamically

Explore dynamic coalition evaluation using the `ShapleyValueCalculator` class. This example demonstrates:
- Custom evaluation functions
- Multiple evaluation strategies
- Team productivity modeling
- Marginal contribution analysis

**Key Concepts:**
- Evaluation functions
- Synergy effects
- Performance optimization
- Raw data export

```bash
python examples/example_function_evaluation.py
```

### 3. **Real-World Business Cases** (`example_business_case.py`)
**Difficulty:** Intermediate to Advanced  
**Use Case:** Practical business applications

Discover how Shapley values solve real business problems:

#### Scenarios Covered:
- **Joint Venture Profit Sharing**: Fair distribution among partner companies
- **Shared Service Cost Allocation**: IT infrastructure cost distribution
- **Sales Team Commission**: Performance-based team compensation

**Key Concepts:**
- Profit sharing mechanisms
- Cost allocation strategies
- Performance-based compensation
- Negotiation support tools

```bash
python examples/example_business_case.py
```

### 4. **Machine Learning Feature Importance** (`example_ml_features.py`)
**Difficulty:** Advanced  
**Use Case:** ML model interpretation and feature analysis

Apply Shapley values to understand machine learning models:

#### Applications:
- **House Price Prediction**: Feature importance in regression models
- **Spam Detection**: Binary classification feature analysis
- **Feature Interactions**: Understanding complex model relationships

**Key Concepts:**
- Model interpretability
- Feature contribution analysis
- Interaction effects
- ML debugging techniques

```bash
python examples/example_ml_features.py
```

### 5. **Parallel Processing & Performance** (`example_parallel_processing.py`)
**Difficulty:** Advanced  
**Use Case:** Large-scale computations and optimization

Optimize performance for large games and complex evaluations:

#### Topics Covered:
- Sequential vs parallel processing comparison
- Scalability analysis
- Memory efficiency
- Performance optimization strategies

**Key Concepts:**
- Computational complexity
- Parallel processing benefits
- Memory management
- Performance profiling

```bash
python examples/example_parallel_processing.py
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install shapley-value
```

### Running Examples

1. **Individual Examples:**
   ```bash
   cd /path/to/shapley-value
   python examples/example_basic_coalition.py
   ```

2. **All Examples:**
   ```bash
   # Run all examples sequentially
   for example in examples/example_*.py; do
       echo "Running $example..."
       python "$example"
       echo "---"
   done
   ```

## 📊 Example Complexity Guide

| Example | Players | Coalitions | Runtime | Complexity |
|---------|---------|------------|---------|------------|
| Basic Coalition | 3 | 8 | <1s | Beginner |
| Function Evaluation | 4 | 16 | <1s | Intermediate |
| Business Cases | 3-4 | 8-16 | <1s | Intermediate |
| ML Features | 5 | 32 | <1s | Advanced |
| Parallel Processing | 8-16 | 256-65,536 | 1-30s | Advanced |

## 🎯 Choosing the Right Example

### For Learning:
1. Start with **Basic Coalition Values** to understand fundamentals
2. Progress to **Function-based Evaluation** for dynamic scenarios
3. Explore **Business Cases** for practical applications

### For Specific Use Cases:
- **Business Applications**: Business Cases example
- **Model Interpretation**: ML Features example  
- **Performance Optimization**: Parallel Processing example
- **Custom Scenarios**: Function-based Evaluation example

### For Different Experience Levels:
- **Beginners**: Basic Coalition Values
- **Intermediate Users**: Function Evaluation, Business Cases
- **Advanced Users**: ML Features, Parallel Processing

## 💡 Tips for Using Examples

### 1. **Modify and Experiment**
- Change player values and coalition structures
- Try different evaluation functions
- Experiment with various business scenarios

### 2. **Performance Considerations**
- Small examples run instantly
- Large examples (>15 players) may take several minutes
- Use parallel processing for better performance

### 3. **Real-World Adaptation**
- Use examples as templates for your scenarios
- Adapt evaluation functions to your specific needs
- Consider data sources and integration requirements

### 4. **Understanding Output**
- Pay attention to Shapley value interpretations
- Compare different allocation methods
- Analyze fairness and efficiency properties

## 🔧 Customization Guide

### Creating Your Own Examples

1. **Choose the Right Class:**
   ```python
   # For predefined coalition values
   from shapley_value import ShapleyCombinations
   
   # For dynamic evaluation
   from shapley_value import ShapleyValueCalculator
   ```

2. **Define Your Scenario:**
   ```python
   # Example: Custom evaluation function
   def my_evaluation_function(coalition):
       # Your custom logic here
       return coalition_value
   
   players = [...]  # Your players
   calculator = ShapleyValueCalculator(my_evaluation_function, players)
   shapley_values = calculator.calculate_shapley_values()
   ```

3. **Add Performance Optimization:**
   ```python
   # For large games
   calculator = ShapleyValueCalculator(
       evaluation_function=my_function,
       players=players,
       num_jobs=-1  # Use all CPU cores
   )
   ```

## 📈 Expected Outputs

All examples generate detailed output including:
- **Shapley Values**: Fair allocation for each player
- **Percentage Distributions**: Relative importance/contribution
- **Total Values**: Verification of efficiency property
- **Interpretations**: Business insights and implications
- **Performance Metrics**: Timing and scalability information (where applicable)

## 🤝 Contributing Examples

We welcome contributions of new examples! Please consider:
- Real-world scenarios and use cases
- Different industries and applications
- Novel evaluation functions
- Performance optimizations
- Educational content

## 📚 Additional Resources

- [Main README](../README.md) - Package overview and installation
- [API Documentation](../README.md#api-reference) - Detailed API reference
- [Shapley Value Theory](https://en.wikipedia.org/wiki/Shapley_value) - Mathematical background
- [Cooperative Game Theory](https://en.wikipedia.org/wiki/Cooperative_game_theory) - Theoretical foundation

---

*These examples demonstrate the versatility and power of Shapley values across various domains. Each example is designed to be educational, practical, and adaptable to your specific needs.*