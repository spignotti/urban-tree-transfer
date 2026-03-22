---
name: python-setup
description: Bootstrap a Python portfolio repository from PROJECT.md — clean, presentable setup with uv, ruff, pyright, pytest, and nox, adapted to the specific project type
---

## Purpose

Use this skill to initialize a new Python portfolio project from `PROJECT.md`.

Repository category: `portfolio` — public-facing, presentable, polished. Strong README, clean structure, type checking included by default.

## PROJECT.md Format

```markdown
# [Project Name]

## Identity
- **What**: [1 sentence — what does the project do]
- **Why**: [1 sentence — what problem does it solve]
- **Type**: [cli | library | data-pipeline | notebook]
- **Python**: 3.12

## Architecture
- **Framework**: [Click | Typer | FastAPI | Streamlit | plain | none]
- **Dependencies**: [package: why, or "none"]
- **Secrets**: [ENV_VAR_NAME: purpose, or "none"]

## Objectives
### MVP
- [ ] [Concrete, testable goal]

### Non-Goals
- [What this does not do]

## Setup
- **Category**: portfolio
- **Git Remote**: [https://github.com/user/repo or "local only"]
```

Fields the skill reads:
- `name` → package name (kebab-case display, snake_case for Python identifier)
- `type` → directory structure
- `python` → version to pin (default `3.12`)
- `framework` → production framework dep
- `dependencies` → additional production deps
- `secrets` → `.env.example` variables
- `git_remote` → remote URL or "local only"

## Toolchain

- `uv` — package and environment management
- `ruff` — linting and formatting
- `pyright` — type checking (always included for portfolio repos)
- `pytest` — tests
- `nox` — validation entrypoint

## Bootstrap Process

### 1. Read PROJECT.md

Parse all fields. Stop immediately if `PROJECT.md` is missing.

### 2. Initialize uv

```bash
uv init
uv python pin <python-version>
```

### 3. Create Package Structure

Use `src/` layout for all types except notebook.

**cli**
```
src/<package>/
├── __init__.py
├── __main__.py
├── cli.py
└── config.py
tests/
└── test_cli.py
```

**library**
```
src/<package>/
├── __init__.py
├── core.py
└── models.py
tests/
└── test_core.py
```

**data-pipeline**
```
src/<package>/
├── __init__.py
├── pipeline.py
├── models.py
└── config.py
tests/
└── test_pipeline.py
data/               (.gitkeep)
```

**notebook**
```
notebooks/
└── main.ipynb  (+ main.py Jupytext pair)
```

### 4. Install Dependencies

```bash
uv add --dev ruff pyright pytest nox
```

Framework deps:
- Click → `uv add click`
- Typer → `uv add typer`
- FastAPI → `uv add fastapi uvicorn`
- Streamlit → `uv add streamlit`

Any additional deps from PROJECT.md `dependencies`:
```bash
uv add <dep>
```

Add pydantic for cli/api/data-pipeline types that have config or data models:
```bash
uv add pydantic pydantic-settings
```

### 5. Generate pyproject.toml

```toml
[project]
name = "<project-name>"
version = "0.1.0"
description = "<Identity.What from PROJECT.md>"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[dependency-groups]
dev = [
  "nox",
  "pyright",
  "pytest",
  "ruff",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "S"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101", "S105", "S106", "S108", "S113"]

[tool.pyright]
venvPath = "."
venv = ".venv"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Fill `name`, `description`, `requires-python` from PROJECT.md. Production deps go into `dependencies` after `uv add`.

### 6. Generate noxfile.py

```python
import nox


nox.options.sessions = ["lint", "typecheck", "test"]


@nox.session
def lint(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "check", ".")


@nox.session
def format(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "format", ".")


@nox.session
def fix(session: nox.Session) -> None:
    session.run("uv", "run", "ruff", "check", "--fix", ".")
    session.run("uv", "run", "ruff", "format", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.run("uv", "run", "pyright")


@nox.session
def test(session: nox.Session) -> None:
    session.run("uv", "run", "pytest", "-v")
```

### 7. Generate .gitignore

Generate based on tools in use. Always include:
```
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.ruff_cache/
.pyright/
.nox/
.coverage
htmlcov/
build/
dist/
*.egg-info/
```

Add `.env` if secrets are configured in PROJECT.md.
Add `.ipynb_checkpoints/` if project type is notebook.
Nothing else.

### 8. Generate .env.example

Only if `secrets` is defined in PROJECT.md:
```
# ENV_VAR_NAME: purpose
ENV_VAR_NAME=
```

Skip entirely if secrets is "none".

### 9. Generate README.md

Portfolio README quality matters — this is public-facing work.

```markdown
# <project-name>

> <Identity.What from PROJECT.md>

<Identity.Why — why this project exists, what problem it solves>

## Setup

\`\`\`bash
uv sync
\`\`\`

## Usage

<clear usage instructions based on project type and framework>

## Development

\`\`\`bash
uv run nox -s fix        # lint and format
uv run nox -s test       # run tests
uv run nox               # full validation
\`\`\`
```

The README should explain what the project does, why it exists, and how to run it. Keep it honest and direct — no hype, no filler.

### 10. Fill AGENTS.md

The `AGENTS.md` template is already present in the project root — it was copied from the portfolio template. Fill in the placeholders from PROJECT.md:

- Replace `<Project Name>` with the project name
- Replace the description placeholder with `Identity.What` and `Identity.Why` from PROJECT.md
- Replace `<version>` in Tech Stack with the Python version
- Add the framework to Tech Stack if one is configured
- Replace `<type>` in Project Type with the actual project type
- Fill in the Structure section with the actual `src/<package>/` directory name
- Add any project-specific conventions from PROJECT.md

Leave `## Known Constraints` empty — it gets filled over time.

### 11. Write Starter Files

`src/<package>/__init__.py`:
```python
"""<Identity.What from PROJECT.md>."""

__version__ = "0.1.0"
```

Entrypoint by type:

**cli** — `src/<package>/cli.py`:
```python
"""<package> CLI."""

import typer

app = typer.Typer()


@app.command()
def main() -> None:
    """<brief description>."""
    pass
```

**data-pipeline** — `src/<package>/pipeline.py`:
```python
"""<package> pipeline."""


def run() -> None:
    """Run the pipeline."""
    pass
```

**library** — `src/<package>/core.py`:
```python
"""<package> core."""
```

Smoke test — `tests/test_<package>.py`:
```python
"""Smoke tests."""


def test_import() -> None:
    import <package>
    assert <package>.__version__ == "0.1.0"
```

### 12. Initialize Git

```bash
git init
git add .
git commit -m "chore: initial project setup"
```

If `git_remote` is configured (not "local only"):
```bash
git remote add origin <url>
git push -u origin main
```

If push fails: stop and report init is incomplete — likely causes: remote does not exist, wrong URL, auth issue.

### 13. Verify

```bash
OPENCODE_CONFIG=./opencode.json opencode --help >/dev/null
uv sync
uv run nox
```

Both checks must be green. Fix any failures before declaring success.

### 14. Summary

Report:
- project name, type, Python version, tools installed
- git status: committed on main; remote push result if configured
- nox result
- next step: start `auto` with `PROJECT.md`, or use `build`/`plan` for manual work

## Guiding principles

Scaffold only — the purpose of init is a working skeleton, not a product. Implementing business logic from PROJECT.md at this stage conflates setup with delivery and makes the scaffold harder to verify.

Create `AGENTS.md` — it's the persistent context file every agent reads at the start of every session. Without it the project has no local context and the agent has to rediscover conventions every time.

Create at least one passing smoke test — a scaffold that doesn't run a test can't prove `nox` works end-to-end. Even a trivial import test gives confidence the structure is correct.

Use `src/` layout (except notebooks) — it prevents accidentally importing from the project root rather than the installed package, which causes subtle bugs in testing.

Use `uv add` rather than editing `pyproject.toml` directly — uv resolves and locks dependencies correctly; hand-editing can produce inconsistent lock state.

Don't create `CHANGELOG.md` — changelogs are release history and premature at scaffold time.

Don't add deps beyond what PROJECT.md specifies — keep the scaffold honest about what the project actually needs.
