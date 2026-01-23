import nox

nox.options.sessions = ["lint", "typecheck"]


@nox.session
def lint(session: nox.Session) -> None:
    """Check code with ruff."""
    session.run("uv", "run", "--active", "ruff", "check", ".", external=True)


@nox.session
def format(session: nox.Session) -> None:
    """Format code with ruff."""
    session.run("uv", "run", "--active", "ruff", "format", ".", external=True)


@nox.session
def typecheck(session: nox.Session) -> None:
    """Type check with pyright."""
    session.run("uv", "run", "--active", "pyright", external=True)


@nox.session
def fix(session: nox.Session) -> None:
    """Auto-fix all issues."""
    session.run("uv", "run", "--active", "ruff", "check", "--fix", ".", external=True)
    session.run("uv", "run", "--active", "ruff", "format", ".", external=True)


@nox.session
def pre_commit(session: nox.Session) -> None:
    """Run before commit."""
    session.run("uv", "run", "--active", "ruff", "check", "--fix", ".", external=True)
    session.run("uv", "run", "--active", "ruff", "format", ".", external=True)
    session.run("uv", "run", "--active", "pyright", external=True)


@nox.session
def ci(session: nox.Session) -> None:
    """Full CI pipeline."""
    session.run("uv", "run", "--active", "ruff", "check", ".", external=True)
    session.run("uv", "run", "--active", "ruff", "format", "--check", ".", external=True)
    session.run("uv", "run", "--active", "pyright", external=True)
