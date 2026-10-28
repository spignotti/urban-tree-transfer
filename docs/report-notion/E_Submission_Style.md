# E. Submission / Style

> Migriert aus Notion am 2026-10-28 (Seite `83ed4403eb5b44a89afe66b2253f6dbb`, zuletzt bearbeitet 2026-03-30). Text unverändert übernommen.

Formatierungs- und Submission-Constraints fuer den Export.

---

## Template

- **Template Name:** Custom LaTeX (KOMA-Script `scrartcl`)
- **Venue:** Projektbericht, [M.Sc](http://M.Sc). Geoinformation, Berliner Hochschule fuer Technik (BHT). Kein spezifisches Journal/Konferenz-Venue.
- **Document Class:** `scrartcl` (KOMA-Script) -- A4, single-column, 11pt
- **Fonts:** Libertinus Serif (Body) + Libertinus Sans (Headings) + Inconsolata (Code)
- **Packages:** `microtype`, `booktabs`, `siunitx`, `hyperref`, `cleveref`, `natbib` (Harvard)
- **Margins:** KOMA-Script default (\~3.5 cm) oder leicht angepasst via `geometry`

## Citation Style

- **Format:** Harvard (Author, Year) -- im Draft als `[@key]` Marker
- **Bibliographie:** Alphabetisch nach Erstautor
- **39 Quellen** in D. References definiert
- **Besonderheiten:**
  - Narrative Form: `[@key]` am Satzanfang wird zu "Author (Year)"
  - Parenthetische Form: `[@key]` in Klammern wird zu "(Author, Year)"
  - Multi-Zitationen: `[@key1]; [@key2]` werden zusammengefasst

## Figure Style

- **18 Figures** in B. Visualizations definiert (14 Main + 4 Appendix)
- **Marker im Draft:** `[FIGURE: fig-slug]`
- **Erwarteter Output:** PNG/PDF, min. 300 dpi
- **Style:** Wissenschaftlich-minimal, konsistente Farbpalette (siehe Color Palette unten)
- **Captions:** Unter der Figure, beschreibend
- **Datenverzeichnis:** `outputs/report/`
- **Default Backend:** matplotlib + seaborn, `savefig` als PDF (vektorisiert)
- **Font in Plots:** Libertinus Serif (matching Document Body) oder sans-serif fallback

## Table Style

- **4 Tabellen** in C. Tables definiert
- **Marker im Draft:** `[TABLE: tbl-slug]`
- **Captions:** Ueber der Tabelle
- **Format:** Booktabs-Stil (keine vertikalen Linien)

## Structure

- **Appendix Required:** Ja (A.1 Hypothesis Tests, A.2 Feature List, A.3 Confusion Matrices, A.4 Hyperparameters)
- **List of Figures Required:** Nein
- **List of Tables Required:** Nein
- **Page Limits:** 15-20 Seiten (exkl. Appendix)
- **Required Sections:** Abstract, Introduction, Related Work, Study Area & Data, Methods, Results, Discussion, Conclusion, References, Appendix

## Extra Constraints

- Sprache: Englisch (British spelling: behaviour, generalise, etc.)
- Keine Em-Dashes (en-dash `--` fuer Ranges, Komma/Semikolon sonst)
- Kein "we" im Fliesstext (Passiv oder "this study")
- Kein Rhetorical Questioning
- Tone: faktisch, komprimiert, facts-first
- Mathematische Notation: LaTeX-konform (\$F\_1\$, weighted \$F\_1\$, \$p\$-Werte, \$alpha = 0.05\$)
- Abkuerzungen: beim ersten Auftreten ausschreiben (z.B. "Canopy Height Model (CHM)")
- Zahlen: Dezimalpunkt, Tausender-Separator optional, 3 Dezimalstellen fuer Metriken
- Figures/Tables: Referenz via `\cref{}` (cleveref)

## Post-Submission

- Abgabe: Ende Maerz 2026
- Weiterleitung an Professor (Statistik-Expertise) zur Pruefung einer gemeinsamen Veroeffentlichung
- Ziel-Venue fuer Publication: offen (allgemeines Uni-Paper, kein spezifisches Journal)

---

## Color Palette

Thematisch abgestimmt auf Vegetation / Remote Sensing. Colorblind-safe (Okabe-Ito-inspiriert).

### Primaerfarben (Dokument + Plots)

- **Forest** `#2D6A4F` -- Ueberschriften, Links, primaere Datenreihe (XGBoost)
- **Teal** `#40916C` -- sekundaere Datenreihe (1D-CNN), Hervorhebungen
- **Amber** `#E07A2F` -- Akzent, Warnungen, dritte Datenreihe (Random Forest)
- **Steel** `#457B9D` -- Referenzlinien, Baselines, vierte Datenreihe
- **Sand** `#D4A276` -- Hintergruende, Fuellflaechen, fuenfte Datenreihe

### Neutralfarben

- **Ink** `#1D3557` -- Body Text in Plots (dunkler als schwarz, waermer)
- **Slate** `#6C757D` -- Achsen, Gridlines, Annotations
- **Cloud** `#F8F9FA` -- Plot-Hintergrund (leicht off-white)

### Erweiterte Palette (17 Klassen)

Fuer Genus-spezifische Visualisierungen (Confusion Matrix, Per-Genus Bars) wird die Palette auf 17 Farben erweitert. Basis: Primaerfarben + Zwischenstufen + muted Varianten. Definition erfolgt im Visualization-Skript als `GENUS_CMAP`.

### Anwendung

- **Bar Charts:** Forest (primaer), Teal (sekundaer), Amber (highlight)
- **Line Plots:** Forest = XGBoost, Teal = 1D-CNN, Amber = RF, Steel = Baseline
- **Heatmaps:** Sequential `Forest`-Palette (hell zu dunkel) oder diverging mit Amber
- **Karten:** Genus-CMAP fuer Baumstandorte, neutraler Kartenhintergrund
