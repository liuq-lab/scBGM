# scbgm: the Python package

Install from the repository root as described in the README and run the `scbgm`
command. Keep the `scbgm` package and subpackage names so installation and imports work.

| Folder or file | Purpose |
| --- | --- |
| `data/` | Data splitting, highly variable gene selection and downstream data preparation |
| `training/` | AE, unconditional, conditional, trajectory and counterfactual scBGM training |
| `generation/` | Loads trained models and generates cells |
| `models/` | Model implementations and helpers |
| `cli.py`, `registry.py` | The `scbgm` command and the script behind each subcommand |
| `environment.py` | Environment check behind `scbgm setup` |
| `audit.py` | Installation checks behind `scbgm doctor` |
| `utils.py` | Small file helpers used by generation and `scbgm setup` |
| `__init__.py`, `__main__.py` | Package entry points, including `python -m scbgm` |
