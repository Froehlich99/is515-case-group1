# Group 1 - Process Management Project

Group project for IS 515 - Process Management & Analytics (HWS 2025).

## Setup

1. Clone the repo: `https://github.com/Froehlich99/is515-case-group1`
2. Install UV: `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Create virtual environment and sync deps: `uv venv` then `uv sync`
4. For Jupyter notebooks: `uv add jupyter` then `python -m ipykernel install`

## Dependencies

Managed via UV in pyproject.toml. Core: PM4Py for discovery/conformance, pandas for data handling, graphviz for visuals. Add extras with `uv add <pkg>` and commit updates.

## Notes

- PM4Py requires Graphviz: Install system-wide (e.g., `brew install graphviz` on macOS).
