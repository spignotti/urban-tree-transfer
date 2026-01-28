# Notebook Templates für Phase 2

**Phase:** 2 - Feature Engineering  
**Created:** 2026-01-28  
**Purpose:** Standardisierte Strukturen für Runner und Exploratory Notebooks

---

## Overview

Alle Notebooks in Phase 2 folgen standardisierten Templates basierend auf dem bewährten Runner-Notebook 1. Dies gewährleistet:

- **Konsistente Struktur** → Einfache Wartung und Nachvollziehbarkeit
- **Wiederverwendbares Setup** → Git-Integration, Plotting, Logging
- **Reproduzierbarkeit** → Gleiche Konfiguration, gleiche Umgebung

---

## 1. Runner-Notebook Template (Phase 2)

**Zweck:** Datenverarbeitung ausführen (Feature Extraction, Quality Control, Final Preparation)  
**Basis:** `notebooks/runners/01_data_processing.ipynb`  
**Ausgabe:** Prozessierte Datasets, Metadaten, Logs

### Zell-Struktur

#### **Zelle 1: Runtime Settings & Package Installation**

```python
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard) / GPU (if needed)
# High-RAM: Recommended for large datasets
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================

import subprocess
from google.colab import userdata

# Get GitHub token from Colab Secrets (for private repo access)
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        "GITHUB_TOKEN not found in Colab Secrets.\n"
        "1. Click the key icon in the left sidebar\n"
        "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
        "3. Toggle 'Notebook access' ON"
    )

# Install package from private GitHub repo
repo_url = f"git+https://{token}@github.com/SilasPignotti/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

print("OK: Package installed and imports complete")
```

**Anpassungen:**

- GPU-Anforderung je nach Bedarf (`Required: GPU (Standard)`)
- High-RAM je nach Datenmenge

---

#### **Zelle 2: Mount Google Drive**

```python
# Mount Google Drive for data files
from google.colab import drive
drive.mount("/content/drive")

print("Google Drive mounted")
```

**Keine Anpassungen nötig** → Immer gleich

---

#### **Zelle 3: Package Imports**

```python
# Package imports
from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
from urban_tree_transfer.config.loader import load_city_config, load_feature_config
from urban_tree_transfer.feature_engineering import (
    # Funktionen je nach PRD (002a/b/c)
    # Beispiel für 002a:
    correct_tree_positions,
    extract_chm_features,
    extract_sentinel_features,
)
from urban_tree_transfer.utils import (
    ExecutionLog,
    generate_validation_report,
    setup_plotting,
    validate_dataset,
)

setup_plotting()
log = ExecutionLog("02a_feature_extraction")  # Notebook-spezifischer Name

print("OK: Package imports complete")
```

**Anpassungen:**

- `ExecutionLog("02a_...")` → Notebook-spezifischer Name
- Import-Liste → Funktionen aus entsprechendem PRD-Modul

---

#### **Zelle 4: Configuration**

```python
# ============================================================
# CONFIGURATION
# ============================================================

# Large data files → Google Drive (not in repo)
DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")

# Phase 2 data directories
INPUT_DIR = DRIVE_DIR / "data" / "phase_1_processing"  # Input von Phase 1
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"   # Output für Phase 2

# Metadata & logs → Google Drive (download manually and commit to repo)
METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

# Cities to process
CITIES = ["berlin", "leipzig"]

# Create directories
for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load feature configuration (static defaults from YAML)
feature_config = load_feature_config()

print(f"Input (Phase 1):   {INPUT_DIR}")
print(f"Output (Phase 2):  {OUTPUT_DIR}")
print(f"Metadata (Drive):  {METADATA_DIR}")
print(f"Logs (Drive):      {LOGS_DIR}")
print(f"Cities:            {CITIES}")
print(f"Random seed:       {RANDOM_SEED}")
```

**Anpassungen:**

- `INPUT_DIR` → Quelle der Eingabedaten (meist Phase 1)
- `OUTPUT_DIR` → Phase 2 Subdirectory (z.B. `phase_2_features`, `phase_2_quality`, `phase_2_splits`)
- `feature_config` → Nur laden, wenn benötigt (in 002a/b/c)
- Weitere Config-Loads je nach PRD (z.B. `temporal_selection.json` in 002b)

---

#### **Zelle 5+: Processing Sections**

```python
# ============================================================
# SECTION: [Section Name] (z.B. "Feature Extraction - CHM")
# ============================================================

log.start_step("[Section Name]")

try:
    # Check if outputs already exist (skip if yes)
    output_path = OUTPUT_DIR / "chm_features.parquet"

    if output_path.exists():
        print(f"Found existing output: {output_path}")
        df = pd.read_parquet(output_path)
        validation = validate_dataset(df, expected_columns=["tree_id", "chm_height", ...])
        print(f"Validation: {validation['schema']['valid']}")
        log.end_step(status="success", records=len(df))
        print(f"Loaded: {output_path}")
    else:
        # Load inputs
        print("Loading inputs...")
        trees = gpd.read_file(INPUT_DIR / "trees" / "trees_filtered_viable.gpkg")
        chm_paths = {
            "berlin": INPUT_DIR / "chm" / "CHM_1m_berlin.tif",
            "leipzig": INPUT_DIR / "chm" / "CHM_1m_leipzig.tif",
        }

        # Process data
        print("Processing...")
        features = extract_chm_features(trees, chm_paths, feature_config)

        # Save outputs
        features.to_parquet(output_path)
        print(f"Saved: {output_path}")

        # Validate
        validation = validate_dataset(features, expected_columns=["tree_id", "chm_height", ...])
        print(f"Validation: {validation['schema']['valid']}")

        log.end_step(status="success", records=len(features))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise
```

**Anpassungen:**

- Section name → Beschreibend je nach Verarbeitungsschritt
- Input/Output paths → PRD-spezifisch
- Processing logic → Funktionsaufrufe aus entsprechendem Modul

**Pattern:**

- Immer `log.start_step()` und `log.end_step()` verwenden
- Immer prüfen, ob Output existiert (Skip-Logik)
- Immer validieren nach dem Speichern
- Immer `try/except` für Fehlerbehandlung

---

#### **Letzte Zelle: Summary & Validation**

```python
# ============================================================
# SUMMARY & VALIDATION
# ============================================================

# Generate validation report for all outputs
print("Generating validation report...")

datasets_to_validate = {
    "chm_features": OUTPUT_DIR / "chm_features.parquet",
    "sentinel_features": OUTPUT_DIR / "sentinel_features.parquet",
    # ... weitere Outputs
}

validation_report = generate_validation_report(datasets_to_validate)
print(f"Validation summary: {validation_report['summary']}")

# Save validation report
validation_path = METADATA_DIR / "validation_report.json"
validation_path.write_text(json.dumps(validation_report, indent=2, default=str), encoding="utf-8")
print(f"Validation report saved: {validation_path}")

# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

# ============================================================
# DETAILED OUTPUT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("OUTPUT SUMMARY")
print("=" * 60)

# Print statistics about outputs
chm_df = pd.read_parquet(OUTPUT_DIR / "chm_features.parquet")
print(f"\n--- CHM FEATURES ---")
print(f"Total trees: {len(chm_df):,}")
print(f"Columns: {list(chm_df.columns)}")
print(f"NaN counts:\n{chm_df.isna().sum()}")

# ... weitere Statistiken

print("\n" + "=" * 60)
print("NOTEBOOK COMPLETE")
print("=" * 60)
```

**Anpassungen:**

- `datasets_to_validate` → Alle Output-Dateien des Notebooks
- Output summary → Statistiken je nach Datensatz

---

## 2. Exploratory-Notebook Template (Phase 2)

**Zweck:** Parameter bestimmen, Analysen durchführen, Plots erstellen  
**Basis:** Runner-Template + Visualisierung  
**Ausgabe:** JSON-Konfigurationen, Publication-Quality Plots, Metadaten

### Zell-Struktur

#### **Zellen 1-4: Identisch zu Runner-Notebooks**

Setup, Drive mount, imports, configuration → **Exakt gleiche Struktur**

**Unterschied in Zelle 3 (Imports):**

```python
# Package imports
from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
from urban_tree_transfer.config.loader import load_city_config, load_feature_config
from urban_tree_transfer.feature_engineering import (
    # Funktionen je nach Analyse
    compute_jm_distance,
    compute_anova_eta_squared,
    compute_morans_i,
)
from urban_tree_transfer.utils import (
    ExecutionLog,
    setup_plotting,
    save_figure,
    PUBLICATION_STYLE,
)
import matplotlib.pyplot as plt
import seaborn as sns

setup_plotting()
log = ExecutionLog("exp_01_temporal_analysis")

print("OK: Package imports complete")
```

**Zusätzlich:**

- `save_figure, PUBLICATION_STYLE` → Für Visualisierung
- `matplotlib.pyplot`, `seaborn` → Plotting-Bibliotheken

---

#### **Zelle 4 (Configuration) - Erweitert:**

```python
# ============================================================
# CONFIGURATION
# ============================================================

# Large data files → Google Drive (not in repo)
DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")

# Phase 2 data directories
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"     # Input von 002a
OUTPUT_DIR = DRIVE_DIR / "outputs" / "phase_2"          # Outputs (JSONs + Plots)

# Output subdirectories
METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures" / "exp_01_temporal"  # Notebook-spezifisch

# Cities to process
CITIES = ["berlin", "leipzig"]

# Create directories
for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load feature configuration
feature_config = load_feature_config()

print(f"Input (Phase 2):    {INPUT_DIR}")
print(f"Output (JSONs):     {METADATA_DIR}")
print(f"Figures:            {FIGURES_DIR}")
print(f"Logs (Drive):       {LOGS_DIR}")
print(f"Cities:             {CITIES}")
print(f"Random seed:        {RANDOM_SEED}")
```

**Unterschied:**

- `FIGURES_DIR` → Separates Verzeichnis für Plots (notebook-spezifisch)
- `OUTPUT_DIR` → `outputs/phase_2` (nicht `data`)

---

#### **Zelle 5+: Analysis Sections**

```python
# ============================================================
# SECTION: [Analysis Name] (z.B. "JM Distance Analysis")
# ============================================================

log.start_step("[Analysis Name]")

try:
    # Load data
    print("Loading data...")
    features = pd.read_parquet(INPUT_DIR / "sentinel_features.parquet")

    # Perform analysis
    print("Computing JM distances...")
    jm_results = {}
    for city in CITIES:
        city_data = features[features["city"] == city]
        jm_matrix = compute_jm_distance(city_data, feature_cols=...)
        jm_results[city] = jm_matrix
        print(f"  {city}: {jm_matrix.shape}")

    # --- VISUALIZATION 1: JM Distance Heatmap ---
    print("Creating JM distance heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, city in enumerate(CITIES):
        ax = axes[i]
        sns.heatmap(jm_results[city], annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
        ax.set_title(f"JM Distance - {city}")

    plt.tight_layout()
    save_figure(fig, FIGURES_DIR / "jm_heatmap.png")
    print(f"  Saved: jm_heatmap.png")

    # --- VISUALIZATION 2: JM Distribution ---
    print("Creating JM distance distribution...")
    fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])

    for city in CITIES:
        jm_values = jm_results[city].values.flatten()
        ax.hist(jm_values, bins=30, alpha=0.5, label=city)

    ax.set_xlabel("JM Distance")
    ax.set_ylabel("Frequency")
    ax.set_title("JM Distance Distribution")
    ax.legend()

    save_figure(fig, FIGURES_DIR / "jm_distribution.png")
    print(f"  Saved: jm_distribution.png")

    # Save results to JSON
    output_json = {
        "analysis": "jm_distance",
        "cities": CITIES,
        "threshold": 1.8,  # Example threshold
        "selected_features": ["NDVI_06", "EVI_06", ...],
        "results": {city: jm_results[city].tolist() for city in CITIES},
    }

    json_path = METADATA_DIR / "temporal_selection.json"
    json_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    print(f"Configuration saved: {json_path}")

    log.end_step(status="success", records=len(output_json["selected_features"]))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise
```

**Pattern:**

1. **Load data** → Input von vorherigen Runner-Notebooks
2. **Perform analysis** → Statistische Berechnungen
3. **Create visualizations** → 2-4 Plots pro Section
4. **Save plots** → `save_figure(fig, FIGURES_DIR / "name.png")`
5. **Save JSON config** → Parameter/Schwellenwerte für nächste Runner
6. **Log results** → `log.end_step()`

**Visualisierung-Best-Practices:**

- Immer `setup_plotting()` in Zelle 3
- Immer `PUBLICATION_STYLE["figsize"]` für figure size
- Immer `save_figure()` statt `plt.savefig()`
- Immer descriptive filenames (`jm_heatmap.png`, nicht `plot1.png`)
- Immer `tight_layout()` vor dem Speichern

---

#### **Letzte Zelle: Summary & Manual Sync Instructions**

```python
# ============================================================
# SUMMARY & MANUAL SYNC INSTRUCTIONS
# ============================================================

# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

# ============================================================
# OUTPUT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("OUTPUT SUMMARY")
print("=" * 60)

print("\n--- JSON CONFIGURATIONS ---")
json_files = list(METADATA_DIR.glob("*.json"))
for f in sorted(json_files):
    print(f"  {f.name}")

print("\n--- PLOTS CREATED ---")
plot_files = list(FIGURES_DIR.glob("*.png"))
for f in sorted(plot_files):
    print(f"  {f.name}")

print(f"\nTotal plots: {len(plot_files)}")

# ============================================================
# MANUAL SYNC REQUIRED
# ============================================================
print("\n" + "=" * 60)
print("⚠️  MANUAL SYNC REQUIRED")
print("=" * 60)
print("\nJSON configurations must be synced to Git repo:")
print("1. Download from Google Drive:")
for f in json_files:
    print(f"   - {f.relative_to(DRIVE_DIR)}")

print("\n2. Copy to local repo:")
print(f"   - Destination: outputs/phase_2/metadata/")

print("\n3. Commit and push to Git")
print("   - git add outputs/phase_2/metadata/*.json")
print("   - git commit -m 'Add exploratory analysis configs'")
print("   - git push")

print("\n4. (Optional) Commit plots for documentation:")
print(f"   - Source: {FIGURES_DIR}")
print(f"   - Destination: outputs/phase_2/figures/exp_XX_*/")

print("\n" + "=" * 60)
print("NOTEBOOK COMPLETE")
print("=" * 60)
```

**Wichtig:**

- Explizite "Manual Sync Required" Warnung
- Liste aller JSON-Files zum Download
- Schritt-für-Schritt Anleitung für Git-Commit
- Optional: Plots committen (für Dokumentation)

---

## 3. Zusammenfassung der Unterschiede

| Aspekt                 | Runner-Notebook                            | Exploratory-Notebook                        |
| ---------------------- | ------------------------------------------ | ------------------------------------------- |
| **Zweck**              | Datenverarbeitung ausführen                | Parameter bestimmen, Analysen               |
| **Output**             | Prozessierte Datasets (Parquet/GeoPackage) | JSONs + Plots                               |
| **Output-Verzeichnis** | `data/phase_2_*`                           | `outputs/phase_2`                           |
| **Visualisierung**     | Nur summary plots (optional)               | 4-6 publication-quality plots               |
| **JSON-Output**        | Metadaten (optional)                       | Konfigurationen (erforderlich)              |
| **Manual Sync**        | Optional (nur Metadaten)                   | **Erforderlich** (JSONs für nächste Runner) |
| **Figures Directory**  | Nicht verwendet                            | `outputs/phase_2/figures/exp_XX_*`          |
| **Logging**            | Standard (`ExecutionLog`)                  | Standard + Plot-Liste                       |
| **End Summary**        | Dataset-Statistiken                        | JSON + Plot Liste + Sync-Anleitung          |

---

## 4. Setup-Checklist für Neue Notebooks

### Runner-Notebook

- [ ] Zelle 1: Runtime Settings anpassen (CPU/GPU, RAM)
- [ ] Zelle 3: Notebook-spezifischen Namen in `ExecutionLog("XX_name")`
- [ ] Zelle 3: Imports aus entsprechendem Modul (`extraction`, `quality`, `splits`)
- [ ] Zelle 4: `INPUT_DIR` und `OUTPUT_DIR` anpassen
- [ ] Zelle 4: Config-Loads hinzufügen (falls benötigt, z.B. `temporal_selection.json`)
- [ ] Zellen 5+: Processing Sections mit `log.start_step()` / `log.end_step()`
- [ ] Letzte Zelle: `datasets_to_validate` anpassen
- [ ] Letzte Zelle: Output summary mit Dataset-Statistiken

### Exploratory-Notebook

- [ ] Zelle 1-2: Identisch zu Runner (keine Änderungen)
- [ ] Zelle 3: `save_figure, PUBLICATION_STYLE` importieren
- [ ] Zelle 3: `matplotlib.pyplot, seaborn` importieren
- [ ] Zelle 3: Notebook-spezifischen Namen in `ExecutionLog("exp_XX_name")`
- [ ] Zelle 4: `FIGURES_DIR` mit notebook-spezifischem Subdirectory
- [ ] Zelle 4: `OUTPUT_DIR = outputs/phase_2`
- [ ] Zellen 5+: Analysis Sections mit Visualisierung
- [ ] Visualisierung: `setup_plotting()` bereits in Zelle 3
- [ ] Visualisierung: `save_figure(fig, FIGURES_DIR / "name.png")`
- [ ] JSON-Output: Konfigurationen in `METADATA_DIR`
- [ ] Letzte Zelle: Plot-Liste in Summary
- [ ] Letzte Zelle: "Manual Sync Required" Warnung mit Anleitung

---

## 5. Beispiel-Verwendung in PRDs

In jedem PRD (002a/b/c und exploratory) sollte ein **"Notebook Structure"** Abschnitt enthalten sein:

```markdown
### Notebook Structure

**Template:** [Runner-Notebook Template](../../docs/documentation/02_Feature_Engineering/02_Notebook_Templates.md#1-runner-notebook-template-phase-2)

**Anpassungen:**

- **Zelle 3 (Imports):** `extraction.py` Funktionen
- **Zelle 4 (Config):** `INPUT_DIR = phase_1_processing`, `OUTPUT_DIR = phase_2_features`
- **Zelle 5:** Feature Extraction - CHM
- **Zelle 6:** Feature Extraction - Sentinel-2
- **Letzte Zelle:** Validate `chm_features.parquet`, `sentinel_features.parquet`

**Siehe:** [Complete template documentation](../../docs/documentation/02_Feature_Engineering/02_Notebook_Templates.md)
```

---

**Letzte Aktualisierung:** 2026-01-28
