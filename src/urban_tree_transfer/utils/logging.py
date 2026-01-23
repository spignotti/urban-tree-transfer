"""Execution logging for notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log_step(step_name: str) -> None:
    """Print formatted step header with timestamp."""
    timestamp = _timestamp()
    print(f"\n{'=' * 70}")
    print(f"[{timestamp}] {step_name}")
    print(f"{'=' * 70}")


def log_success(message: str) -> None:
    """Print success message."""
    print(f"[OK] {message}")


def log_warning(message: str) -> None:
    """Print warning message."""
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    """Print error message."""
    print(f"[ERROR] {message}")


@dataclass
class StepResult:
    """Result of a processing step."""

    name: str
    status: str  # "success", "warning", "error"
    start_time: str
    end_time: str | None = None
    records: dict | int | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ExecutionLog:
    """Track notebook execution for metadata export."""

    notebook: str
    execution_start: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_end: str | None = None
    steps: list[StepResult] = field(default_factory=list)
    _current_step: StepResult | None = field(default=None, repr=False)

    def start_step(self, name: str) -> None:
        """Start tracking a new step."""
        log_step(name)
        self._current_step = StepResult(
            name=name,
            status="in_progress",
            start_time=datetime.now().isoformat(),
        )

    def end_step(
        self,
        status: str = "success",
        records: dict | int | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Complete current step."""
        if self._current_step is None:
            return

        self._current_step.status = status
        self._current_step.end_time = datetime.now().isoformat()
        self._current_step.records = records
        self._current_step.warnings = warnings or []
        self._current_step.errors = errors or []

        self.steps.append(self._current_step)

        if status == "success":
            log_success(f"{self._current_step.name} complete")
        elif status == "warning":
            log_warning(f"{self._current_step.name} complete with warnings")
        else:
            log_error(f"{self._current_step.name} failed")

        self._current_step = None

    def save(self, path: Path) -> None:
        """Save execution log to JSON."""
        self.execution_end = datetime.now().isoformat()

        data = {
            "notebook": self.notebook,
            "execution_start": self.execution_start,
            "execution_end": self.execution_end,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "records": step.records,
                    "warnings": step.warnings,
                    "errors": step.errors,
                }
                for step in self.steps
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

        log_success(f"Execution log saved to {path}")

    def summary(self) -> None:
        """Print execution summary."""
        print(f"\n{'=' * 70}")
        print("EXECUTION SUMMARY")
        print(f"{'=' * 70}")

        for step in self.steps:
            status = step.status.upper()
            records_str = ""
            if step.records:
                if isinstance(step.records, dict):
                    records_str = f" ({sum(step.records.values()):,} total)"
                else:
                    records_str = f" ({step.records:,} records)"
            print(f"  [{status}] {step.name}{records_str}")

        print(f"{'=' * 70}")
