# Contributing

Thank you for your interest in contributing to nmoe!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/nether-soup.git
cd nether-soup/nmoe

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
pip install ruff pytest pytest-cov
```

## Code Style

We use Ruff for linting and formatting:

```bash
# Check style
ruff check nmoe/ tests/

# Format code
ruff format nmoe/ tests/
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run GPU tests only
pytest tests/ -v -m gpu

# Run without GPU tests
pytest tests/ -v -m "not gpu"

# Run with coverage
pytest tests/ --cov=nmoe --cov-report=html
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests and linting
5. Commit with descriptive message
6. Push and create PR

## Commit Messages

Use conventional commits:

```
feat: Add NVFP4 support for MoE layer
fix: Resolve RDEP buffer overflow
docs: Update SGLang integration guide
test: Add distributed tests for EP=4
```

## Code Review

All PRs require:

- Passing CI (lint, tests)
- Code review from maintainer
- Updated documentation if needed
- Tests for new features
