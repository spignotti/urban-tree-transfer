# Report Sources

Sources of the project report *Cross-City Transfer of Urban Tree Genus Classification Using
Multitemporal Sentinel-2 Data: Methodology, Workflow, and Evaluation* (M.Sc. Geoinformation,
Berliner Hochschule für Technik (BHT), submitted 31 March 2026). The submitted PDF is
[in the repository root](../Pignotti_Cross-City%20Transfer%20of%20Urban%20Tree%20Genus%20Classification.pdf).

## Files

| Path | Content |
|------|---------|
| `paper-draft.md` | Markdown manuscript source |
| `paper-draft.tex` | Final LaTeX source of the submitted PDF, self-contained including bibliography |
| `tables.md` | Table definitions used in the manuscript |
| `references.bib` | BibTeX bibliography |
| `references.md` | Harvard reference list with DOIs |
| `figures/` | Report figures (PNG) |
| `data/` | CSV data inputs for the figures and tables |
| `plots/generate_plots.py` | Plot generation script |
| `plots/style.py` | Shared plotting style |
| `template/template.tex` | KOMA-Script `scrartcl` template used to render the manuscript |

`paper-draft.tex` is the version that was actually compiled and contains the last edits made
directly in LaTeX; `paper-draft.md` is the Markdown source it was derived from.

## Regenerate plots

From the repository root:

```bash
uv run python paper/plots/generate_plots.py
```

The script uses `matplotlib`, `numpy` and `pandas`. Inputs are read from `paper/data/`; a few
figures additionally read metadata from `outputs/phase_3_experiments/metadata/`. Outputs are
written to `paper/figures/`.

## Build the PDF

The preamble uses T1 font encoding and the `libertinus` package without `fontspec`, so the
standard `pdflatex` engine works:

```bash
cd paper
latexmk -pdf paper-draft.tex
```

The bibliography is part of `paper-draft.tex`, so no separate BibTeX run or additional input
files are required.
