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

```bash
python examples/example_function_evaluation.py
```

### 3. **Real-World Business Cases** (`example_business_case.py`)
**Difficulty:** Intermediate to Advanced  
**Use Case:** Practical business applications

Discover how Shapley values solve real business problems:
- **Joint Venture Profit Sharing**: Fair distribution among partner companies
- **Shared Service Cost Allocation**: IT infrastructure cost distribution
- **Sales Team Commission**: Performance-based team compensation

```bash
python examples/example_business_case.py
```

### 4. **Machine Learning Feature Importance** (`example_ml_features.py`)
**Difficulty:** Advanced  
**Use Case:** ML model interpretation and feature analysis

Apply Shapley values to understand machine learning models:
- **House Price Prediction**: Feature importance in regression models
- **Spam Detection**: Binary classification feature analysis
- **Feature Interactions**: Understanding complex model relationships

```bash
python examples/example_ml_features.py
```

### 5. **Parallel Processing & Performance** (`example_parallel_processing.py`)
**Difficulty:** Advanced  
**Use Case:** Large-scale exact computations and optimization

Optimise performance for large exact games using `ShapleyValueCalculator`:
- Sequential vs parallel processing comparison (`n_jobs`)
- Scalability analysis across player counts
- Memory efficiency

```bash
python examples/example_parallel_processing.py
```

### 6. **Monte Carlo Approximation** (`example_montecarlo.py`)
**Difficulty:** Advanced  
**Use Case:** Games with many players (20+) where exact computation is infeasible

Approximate Shapley values for large games using `MonteCarloShapleyValue`:
- **Basic usage** – any callable evaluation function, any player types
- **MC vs exact comparison** – error vs sample count for a 3-player game
- **Parallel processing benchmark** – `n_jobs` timing table with speedup ratios
- **Convergence diagnostics** – `get_convergence_data()` running estimates
- **Raw data inspection** – `get_raw_data()` per-permutation marginal contributions

```bash
python examples/example_montecarlo.py
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install shapley-value
```

### Running Examples

```bash
# Individual example
python examples/example_basic_coalition.py

# All examples
for example in examples/example_*.py; do
    echo "Running $example..."
    python "$example"
    echo "---"
done
```

## 📊 Example Complexity Guide

| Example | Class used | Players | Complexity | Runtime |
|---------|-----------|---------|------------|---------|
| Basic Coalition | `ShapleyCombinations` | 3 | Beginner | < 1 s |
| Function Evaluation | `ShapleyValueCalculator` | 4 | Intermediate | < 1 s |
| Business Cases | `ShapleyCombinations` | 3–4 | Intermediate | < 1 s |
| ML Features | `ShapleyValueCalculator` | 5 | Advanced | < 1 s |
| Parallel Processing | `ShapleyValueCalculator` | 8–16 | Advanced | 1–30 s |
| Monte Carlo | `MonteCarloShapleyValue` | 4–50 | Advanced | 1–30 s |

## 🎯 Choosing the Right Example

### By number of players / scale

| Players | Recommended approach | Example |
|---------|---------------------|---------|
| ≤ 10 | `ShapleyCombinations` (exact, pre-defined values) | Basic Coalition |
| ≤ 20 | `ShapleyValueCalculator` (exact, evaluation function) | Function Evaluation / Parallel |
| 20+ | `MonteCarloShapleyValue` (approximate, scalable) | Monte Carlo |

### By use case

- **Business Applications** → Business Cases example
- **Model Interpretation** → ML Features example
- **Large-scale / approximate** → Monte Carlo example
- **Custom scenarios** → Function-based Evaluation example

## 🔧 Customisation Guide

### Choosing the right class

```python
# Exact – when every coalition value is known in advance
from shapley_value import ShapleyCombinations

# Exact – when values are computed by a function (≤ ~20 players)
from shapley_value import ShapleyValueCalculator

# Approximate – for 20+ players or expensive evaluation functions
from shapley_value import MonteCarloShapleyValue
```

### Adding parallelism

Both `ShapleyValueCalculator` and `MonteCarloShapleyValue` follow the
**scikit-learn `n_jobs` convention**:

```python
# Sequential (no overhead)
ShapleyValueCalculator(f, players, n_jobs=1)
MonteCarloShapleyValue(f, players, n_jobs=1)

# All available CPU cores
ShapleyValueCalculator(f, players, n_jobs=-1)
MonteCarloShapleyValue(f, players, n_jobs=-1)

# Exactly 4 cores
ShapleyValueCalculator(f, players, n_jobs=4)
MonteCarloShapleyValue(f, players, n_jobs=4)
```

> **Tip:** For cheap evaluation functions (< 10 µs), `n_jobs=1` avoids
> process-spawn overhead and is faster. For expensive functions (ML models,
> simulations), `n_jobs=-1` gives the best throughput.

### Diagnosing Monte Carlo convergence

```python
mc = MonteCarloShapleyValue(f, players, num_samples=5000, random_seed=0)
convergence_df = mc.get_convergence_data()  # shape: (5000, n_players)

# Plot with your favourite library to see when estimates stabilise:
# convergence_df.plot(title="Running Shapley estimates")
```

## 📚 Additional Resources

- [Main README](../README.md) – Package overview, installation, and API reference
- [Shapley Value Theory](https://en.wikipedia.org/wiki/Shapley_value) – Mathematical background
- [Cooperative Game Theory](https://en.wikipedia.org/wiki/Cooperative_game_theory) – Theoretical foundation

---

*These examples demonstrate the versatility and power of Shapley values across various domains. Each example is designed to be educational, practical, and adaptable to your specific needs.*
