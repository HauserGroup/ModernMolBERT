# Contributing

## Setup

```bash
uv sync --group dev --group eval
uv run pre-commit install
```

## Tests

```bash
uv run pytest -q -m "not smoke and not model and not mps and not cuda and not molformer"
```

## Code style

Ruff handles formatting and linting. Pre-commit runs both automatically on commit.
