# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python 3.11+ ML project managed with `uv` and `pyproject.toml`.

## Commands

```bash
# Install dependencies
uv sync

# Run the main script
uv run python main.py

# Run tests
uv run pytest

# Run a single test
uv run pytest path/to/test_file.py::test_name
```

## Adding Dependencies

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev dependency
```

## Project structure 
### Titanic  
