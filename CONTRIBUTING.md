# Contributing to Simon

Thanks for your interest in contributing.

## Getting Started

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Layout

- `simon.py` — main application entry point
- `models/` — local model assets (Piper/Whisper)

## Development Workflow

1. Create a branch from `main`:

```bash
git checkout -b feat/short-description
```

2. Make focused, minimal changes.
3. Run the project and validate your changes locally.
4. Stage and commit with a clear message:

```bash
git add .
git commit -m "feat: add concise change description"
```

## Code Style

- Keep changes small and task-focused.
- Follow existing style and naming conventions.
- Avoid unrelated refactors in the same pull request.

## Pull Requests

When opening a PR, include:

- What changed
- Why it changed
- How it was tested
- Any follow-up work needed

## Issues

If you find a bug or want to request a feature, please open an issue with:

- Clear title and summary
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python version)

## Security and Secrets

- Do not commit credentials, tokens, or private keys.
- Do not commit large generated files unless required.
- Treat model binaries/configs as project assets and avoid unnecessary churn.
