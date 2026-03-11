# Project Lessons — urban-tree-transfer

Project-specific decisions, constraints, and one-off workarounds.

Format: `- [YYYY-MM-DD] <what was tricky / what went wrong> → <what worked / correct approach>`

## Emergency Recovery

- Last known-good commit (full pipeline run-through complete): `ca8eaa4`
  `git checkout ca8eaa4` to restore this state.

## Setup & Environment

- [2026-03-11] Running the full nox suite including tests is excessive for routine verification in this repo because tests take a long time → default to nox sessions without tests (lint/typecheck/format or pre_commit-equivalent checks), and only run tests when they are actually necessary.

## Data Pipeline

## Notebooks & Colab
