# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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