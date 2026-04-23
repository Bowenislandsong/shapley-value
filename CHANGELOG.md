# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.9] - 2026-04-25

### Changed
- CI: publishing to PyPI runs automatically when a **`v*`** version tag is
  pushed (e.g. `git push origin v0.0.9`); no GitHub Release required for upload.

### Documentation
- README, `CITATION.cff`, and `docs/index.html` updated to v0.0.9.

## [0.0.8] - 2026-04-24

### Changed
- **Breaking:** `ShapleyValueCalculator` constructor parameter `num_jobs` was
  renamed to `n_jobs` (and the instance attribute is now `n_jobs`) to match
  scikit-learn and `MonteCarloShapleyValue`.

### Fixed
- `tests/test_montecarlo_stress.py`: increased work in the expensive evaluation
  stub so the parallel speedup test is stable on CI (avoids false failures when
  joblib overhead dominates).

### Added
- `tests/test_examples.py` now runs `example_montecarlo` with the other examples.

### Documentation
- README, `examples/README`, and `docs/index.html`: citations to v0.0.8, GitHub
  Pages copy aligned with optional `n_jobs` parallelism and Monte Carlo features.

## [0.0.7] - 2026-04-23

### Added
- `MonteCarloShapleyValue` class for approximating Shapley values via random
  permutation sampling (O(m × n) cost vs O(2ⁿ) for exact methods), enabling
  games with 100+ players.
- `n_jobs` parameter on `MonteCarloShapleyValue` following the scikit-learn
  convention (`1` = sequential, `-1` = all cores, `k` = exactly k cores).
  Permutations are generated sequentially before the parallel step so
  `random_seed` always produces bit-identical results regardless of `n_jobs`.
- `get_convergence_data()` method returning a DataFrame of running per-player
  estimates after each sampled permutation – useful for diagnosing convergence
  and tuning `num_samples`.
- `get_raw_data()` method on `MonteCarloShapleyValue` returning per-permutation
  marginal contributions (columns: `iteration`, `permutation`, `player`,
  `marginal_contribution`).
- `MonteCarloShapleyValue` exported from the top-level `shapley_value` package.
- `tests/test_montecarlo.py` – 29 tests covering correctness, reproducibility,
  parallel consistency (`n_jobs` with identical seeds), convergence diagnostics,
  raw-data structure, and edge cases (single player, zero game, negative values,
  string players, two-player exact comparison).
- `tests/test_montecarlo_stress.py` – 13 stress / performance tests:
  - Correctness at scale: 20, 50, and 100 players; additive and synergy games
  - Wall-clock timing bounds ensuring regression is caught by CI
  - Parallel speedup assertion for expensive evaluation functions
  - `TestStressBenchmarkTable` – always-passing test that prints a formatted
    throughput table (permutations/second by player count and `n_jobs`) when
    run with `-s` / `--capture=no`.
- `examples/example_montecarlo.py` – runnable script demonstrating all five
  features including a benchmark table with speedup ratios.

### Changed
- `montecarlo._run_sampling`: empty-coalition value `v([])` is now computed
  once and passed to each permutation worker, eliminating `num_samples`
  redundant calls.
- `montecarlo._marginal_contributions_for_permutation`: coalition list is
  mutated in place (no intermediate `list()` copy) for reduced allocation
  overhead in the hot path.
- README: added `MonteCarloShapleyValue` to the overview table, quick-start
  section, full usage example, API reference, features list, updated performance
  section with honest two-tier benchmarks, and a new Testing section.
- `examples/README.md`: added entry #6 for `example_montecarlo.py`, updated
  complexity guide and customisation section.

## [0.0.6] - 2024-12-01

### Added
- CITATION.cff file for standardized citation information
- Citation section in README.md with BibTeX, APA, and MLA formats
- Citation section in project website (docs/index.html)
- Support for academic citations in multiple formats

### Changed
- Version bump from 0.0.5 to 0.0.6

## [0.0.5] - Previous Release

### Added
- CI workflow now automatically tests against future Python versions (3.13, 3.x)
- Support for testing pre-release Python versions to catch compatibility issues early
- Python 3.13 classifier in package metadata

### Changed
- CI workflow now uses `fail-fast: false` to test all Python versions even if one fails
- Updated GitHub Actions workflow to adapt to all future Python releases automatically

## [0.0.4] - 2025-10-09

### Added
- Comprehensive example suite with 5 detailed examples
- Parallel processing support with automatic optimization
- Performance benchmarking and optimization guidance
- ML feature importance examples
- Real-world business case examples
- Memory-efficient coalition generation
- Comprehensive documentation and README updates

### Fixed
- Critical bug in ShapleyCombinations calculation algorithm
- Shapley value weight calculation formula
- Coalition tuple handling and sorting issues

### Improved
- Enhanced setup.py with comprehensive metadata
- Better package structure and imports
- Performance optimization for large games
- Documentation clarity and examples

## [0.0.3] - Previous Release

### Added
- Enhanced parallel processing
- Performance optimizations

## [0.0.2] - Previous Release

### Added
- Function-based evaluation
- Data export features

## [0.0.1] - Initial Release

### Added
- Basic Shapley value calculation
- Core functionality implementation