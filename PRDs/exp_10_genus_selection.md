# PRD: exp_10 Genus Selection Validation & exp_10→exp_11 Renumbering

**PRD ID:** exp_10_genus_selection
**Status:** Draft - Pending Review
**Created:** 2026-02-09
**Priority:** 🔴 **CRITICAL** - Blocks Berlin Optimization (03b)
**Dependencies:** Phase 2 outputs, exp_07-09 completed
**Estimated Effort:** 2-3 hours implementation + 1-2 hours execution

---

## 🎯 Goal

**Was soll gebaut werden:**
Ein exploratives Notebook zur **Validierung und Finalisierung der Genus-Auswahl** nach Phase 2c Filterung, mit gleichzeitiger Renummerierung des bestehenden Algorithm Comparison Notebooks.

**Erfolgskriterium:**

1. **Neue exp_10:** Finale Genus-Liste mit ≥500 Samples pro Genus in beiden Städten
2. **Umbenannt exp_11:** Algorithm Comparison nutzt finale Genus-Konfiguration
3. **Config-Integration:** `genus_selection_final.json` wird von allen Runner Notebooks gelesen
4. **Dokumentation komplett:** Alle Referenzen auf exp_10/exp_11 aktualisiert

**Warum jetzt:**

- Phase 1 filtert auf 500 Samples → 30 Genera
- Phase 2c (Proximity + Outlier) reduziert Samples weiter → **Filter muss erneut angewendet werden**
- **Lösung:** MIN_SAMPLES_PER_GENUS Filter wird am Ende von Phase 2c integriert
- exp_10 fokussiert auf Separabilität/Gruppierung (nicht Sample-Counting, das macht Phase 2c)
- exp_11 Algorithm Comparison nutzt finale, validierte Genus-Liste

---

## 🧑‍💻 User & Use Case

**Zielgruppe:**

- Forschende, die Genus-Auswahl-Entscheidung nachvollziehen wollen
- Code-Reviewer, die Reproduzierbarkeit sicherstellen
- Zukünftige Studies, die Pipeline auf andere Städte anwenden

**Hauptanwendung:**
Nach Phase 2c Filter:** Review der Genus-Reduktion durch Phase 2c (30 → N Genera)
2. **Vor Berlin-Optimierung (03b):** Separabilitäts-basierte finale Genus-Selektion
3. **In Publikation:** Transparente Dokumentation: Filter-Kaskade + Separabilitäts-Analyse
4. **Methodische Rechtfertigung:** Warum bestimmte Genera ausgeschlossen/gruppiert wurden

**Gelöstes Problem:**

- **Problem 1:** Sample-Count-Filter wird in Phase 2c angewendet, aber Entscheidung nicht im Experiment-Kontext dokumentiert
- **Problem 2:** Keine methodische Rechtfertigung warum finale N Genera (states haben
- **Problem 2:** Keine methodische Rechtfertigung warum 16 Genera (nicht 12 oder 30)
- **Problem 3:** Separabilität zwischen Genera ist nicht quantifiziert
- **Problem 4:** exp_10 läuft mit unvalidierten Genera → potenzielle Neuberechnung nötig

---

## ✅ Success Criteria

### Phase 0: Phase 2c Filter Integration (PREREQUISITE)

- [ ] **Phase 2c Updated:** MIN_SAMPLES_PER_GENUS Filter am Ende von `02c_final_preparation.ipynb` hinzugefügt
- [ ] **Metadata exportiert:** `genus_filter_phase2c.json` mit finaler Genus-Liste + excluded genera
- [ ] **Datensätze gefiltert:** Nur noch viable genera in exported splits

### Phase 1: exp_10 Notebook Creation

- [ ] **Neue Datei:** `notebooks/exploratory/exp_10_genus_selection_validation.ipynb` erstellt
- [ ] **4 Analysen implementiert:** Phase 2c Filter Review, JM-Separability Matrix, Genus Grouping Options, Final Recommendation
- [ ] **Config exportiert:** `outputs/phase_3_experiments/metadata/genus_selection_final.json`
- [ ] **Visualisierungen:** Dendrogram, JM Heatmap, Phase 2c Filter Review Plot
- [ ] **Notebook ausführbar in Colab:** Package Installation, Data Loading funktioniert

### Phase 2: exp_10→exp_11 Renaming

- [ ] **Datei umbenannt:** `exp_10_algorithm_comparison.ipynb` → `exp_11_algorithm_comparison.ipynb`
- [ ] **Config-Import hinzugefügt:** Notebook liest `genus_selection_final.json`
- [ ] **Filter implementiert:** Datensätze werden auf finale Genus-Liste gefiltert
- [ ] **Outputs verschoben:** `exp_10_algorithm_comparison/` → `exp_11_algorithm_comparison/`

### Phase 3: Config System Integration

- [ ] **Neue Config:** `configs/experiments/genus_selection.yaml` erstellt (Template)
- [ ] **JSON-Schema:** `genus_selection_final.json` dokumentiert
- [ ] **Runner-Integration:** 03b, 03c, 03d Notebooks importieren finale Genus-Liste
- [ ] **Backwards-Kompatibilität:** Falls JSON fehlt, Fallback auf alle Genera

### Phase 4: Documentation Updates

- [ ] **00_Experiment_Overview.md:** exp_10/exp_11 Referenzen aktualisiert, Dependency-Graph korrigiert
- [ ] **01_Setup_Fixierung.md:** Neue Sektion "Genus Selection Validation (exp_10)" hinzugefügt
- [ ] **05_Ergebnisse.md:** exp_10 Ergebnisse dokumentiert, exp_11 als "pending" markiert
- [ ] **Phase 2 Methodische Erweiterungen:** Genus-Filter-Cascade dokumentiert
- [ ] **FIGURE_DOCUMENTATION.md:** exp_10/exp_11 Ordner-Referenzen aktualisiert

---

## 🧩 Context & References

### Critical Files to Read

```yaml
- file: "src/urban_tree_transfer/config/constants.py"
  why: "MIN_SAMPLES_PER_GENUS = 500 threshold definition"
  lines: 9

- file: "src/urban_tree_transfer/data_processing/trees.py"
  why: "filter_viable_genera() implementation - Phase 1 logic"
  lines: 446-480

- file: "outputs/phase_2_features/metadata/temporal_selection.json"
  why: "30 viable_genera from Phase 1, JM-distance data for separability"
  critical: "JM values aggregated across months → genus-genus similarity"

- file: "outputs/phase_2_splits/metadata/phase_2_final_summary.json"
  why: "Actual sample counts after Phase 2c filtering"
  critical: "berlin_filtered_train: 471,815 samples, leipzig_filtered_finetune: 91,080"

- file: "docs/documentation/03_Experiments/00_Experiment_Overview.md"
  why: "Experiment dependency chain, all exp_10 references"

- file: "notebooks/exploratory/exp_07_cross_city_baseline.ipynb"
  why: "Pattern for cross-city comparison, Cohen's d calculations"
```

### Existing Patterns to Follow

**Exploratory Notebook Structure:**

- Follows: `exp_07_cross_city_baseline.ipynb`, `exp_09_feature_reduction.ipynb`
- Pattern: Markdown overview → Config → Analysis sections → Export JSON

**Config Export Pattern:**

```python
# From exp_07
config = {
    "version": "1.0",
    "created": datetime.now(timezone.utc).isoformat(),
    "analysis": {...},
    "decision": {...}
}
with open("outputs/.../metadata/name.json", "w") as f:
    json.dump(config, f, indent=2)
```

**Runner Notebook Config Import Pattern:**

```python
# From 03a_setup_fixation.ipynb
import json
with open("outputs/phase_3_experiments/metadata/setup_decisions.json") as f:
    setup_config = json.load(f)
selected_features = setup_config["feature_selection"]["selected_features"]
```

### Known Constraints

```python
# CRITICAL: JM-Distance calculation is expensive
# Solution: Use pre-computed values from temporal_selection.json
# DO NOT re-calculate JM distances from scratch

# CRITICAL: Genus filtering must maintain class balance
# Solution: Check KL-divergence before/after filtering
# Threshold: KL(filtered_dist || original_dist) < 0.1

# CRITICAL: Some genera may have fallen below 500 in Phase 2c
# Solution: Log excluded genera, validate against expected 16-30 range
```

---

## 🏗️ Technical Design

### Component 1: exp_10_genus_selection_validation.ipynb

**Location:** `notebooks/exploratory/exp_10_genus_selection_validation.ipynb`

**IMPORTANT NOTE:** 
Sample-count filtering (MIN_SAMPLES_PER_GENUS ≥500) is performed **in Phase 2c** (`02c_final_preparation.ipynb`).
This notebook focuses on:
1. **Review** of Phase 2c genus filtering results
2. **Separability analysis** (JM-distance matrix)
3. **Grouping recommendations** (optional genus clustering)
4. **Final genus selection** (exclude low-separability vs. group similar)

**Structure (13 cells):**

```markdown
## Cell 1: Markdown - Overview

# Genus Selection Validation

Purpose: Validate genus sample counts after Phase 2c filtering and determine final genus list

## Cell 2: Python - Setup & Installation

!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

## Cell 3: Markdown - Configuration

Define paths, thresholds, strategies

## Cell 4: Python - Configuration

BASE_PATH = Path("outputs")
MIN_SAMPLES = 500
JM_THRESHOLD = 1.0
STRATEGIES = ["exclude_low_sample", "group_similar", "hybrid"]

## Cell 5: Markdown - Data Loading

Load Phase 2 outputs and metadata

## Cell 6: Python - Load Data

berlin_train = pd.read_parquet(BASE_PATH / "phase_2_splits/berlin_filtered_train.parquet")
leipzig_finetune = pd.read_parquet(BASE_PATH / "phase_2_splits/leipzig_filtered_finetune.parquet")
temporal_metadata = json.load(...)

## Cell 7: Markdown - Analysis 1: Sample Count Validation

## Cell 8: Python - Sample Count Analysis

berlin_counts = berlin_train["genus_latin"].value_counts()
leipzig_counts = leipzig_finetune["genus_latin"].value_counts()

# Identify genera below threshold

# Visualize: bar chart with threshold line

## Cell 9: Markdown - Analysis 2: Sample Sufficiency Assessment

## Cell 10: Python - Sufficiency Check

# Literature: RF needs ~100-200 samples/class

# Calculate: total_samples / n_genera

# Validate: meets minimum for reliable training

## Cell 11: Markdown - Analysis 3: Genus Separability Matrix

## Cell 12: Python - JM-Distance Aggregation

# Extract JM values from temporal_selection.json

# Aggregate across months (mean JM distance per genus pair)

# Create symmetric matrix

# Heatmap visualization

## Cell 13: Markdown - Analysis 4: Genus Grouping Exploration

## Cell 14: Python - Hierarchical Clustering

# Linkage from JM-distance matrix

# Dendrogram visualization

# Identify potential groups (e.g., Rosaceae cluster)

## Cell 15: Markdown - Analysis 5: Decision & Export

## Cell 16: Python - Final Genus Selection

# Apply MIN_SAMPLES filter

# Optional: Apply separability grouping

# Export genus_selection_final.json

# Summary statistics
```

**Key Analyses:**

#### Analysis 1: Sample Count Validation

```python
def validate_sample_counts(berlin_df, leipzig_df, min_samples=500):
    """Check which genera meet minimum sample threshold in both cities."""
    berlin_counts = berlin_df["genus_latin"].value_counts()
    leipzig_counts = leipzig_df["genus_latin"].value_counts()

    # Genera present in both cities
    common_genera = set(berlin_counts.index) & set(leipzig_counts.index)

    # Filter by threshold
    viable_berlin = set(berlin_counts[berlin_counts >= min_samples].index)
    viable_leipzig = set(leipzig_counts[leipzig_counts >= min_samples].index)
    viable_both = viable_berlin & viable_leipzig & common_genera

    # Excluded genera (Phase 1 had them, Phase 2c lost them)
    excluded = common_genera - viable_both

    return {
        "viable_genera": sorted(list(viable_both)),
        "excluded_genera": sorted(list(excluded)),
        "berlin_counts": berlin_counts.to_dict(),
        "leipzig_counts": leipzig_counts.to_dict(),
        "n_viable": len(viable_both),
        "n_excluded": len(excluded)
    }
```

#### Analysis 2: Separability Matrix from JM-Distances

```python
def extract_jm_separability_matrix(temporal_metadata):
    """Aggregate JM-distances across months to create genus-genus similarity matrix."""
    viable_genera = temporal_metadata["viable_genera"]
    n_genera = len(viable_genera)

    # Initialize symmetric matrix
    jm_matrix = np.zeros((n_genera, n_genera))

    # Aggregate JM values across all months (mean)
    # NOTE: JM data is in temporal_selection.json per month
    # Need to extract genus pairs and average

    # For now: placeholder - needs detailed parsing of temporal_selection.json
    # Expected: jm_matrix[i,j] = mean JM-distance between genus_i and genus_j

    return jm_matrix, viable_genera

def identify_poorly_separable_pairs(jm_matrix, genera_names, threshold=1.0):
    """Find genus pairs with JM-distance < threshold (hard to separate)."""
    n = len(genera_names)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if jm_matrix[i, j] < threshold:
                pairs.append({
                    "genus_1": genera_names[i],
                    "genus_2": genera_names[j],
                    "jm_distance": float(jm_matrix[i, j])
                })
    return sorted(pairs, key=lambda x: x["jm_distance"])
```

#### Analysis 3: Hierarchical Clustering for Groups

```python
def create_genus_groups(jm_matrix, genera_names, distance_threshold=0.5):
    """Cluster genera based on spectral similarity (low JM = high similarity)."""
    # Convert JM to distance: distance = 2 - JM (since JM ∈ [0,2])
    distance_matrix = 2.0 - jm_matrix

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=genera_names, ax=ax)
    ax.axhline(y=distance_threshold, color='r', linestyle='--', label='Cut threshold')
    ax.set_xlabel("Genus")
    ax.set_ylabel("Distance (2 - JM)")
    ax.set_title("Genus Hierarchical Clustering by Spectral Similarity")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return linkage_matrix
```

**Output JSON Schema:**

```json
{
  "version": "1.0",
  "created": "2026-02-09T15:30:00+00:00",
  "notebook": "exp_10_genus_selection_validation.ipynb",
  "config": {
    "min_samples_per_genus": 500,
    "jm_threshold": 1.0,
    "filtering_stage": "post_phase2c"
  },
  "phase_1_genera": {
    "count": 30,
    "list": ["ACER", "AESCULUS", ...]
  },
  "phase_2c_analysis": {
    "berlin_train_total": 471815,
    "leipzig_finetune_total": 91080,
    "genera_in_data": 16,
    "sample_counts": {
      "berlin": {"ACER": 95234, "TILIA": 78123, ...},
      "leipzig": {"ACER": 18234, "TILIA": 15432, ...}
    }
  },
  "validation_results": {
    "viable_genera": ["ACER", "BETULA", ...],
    "excluded_genera": ["CORYLUS", "MALUS"],
    "n_viable": 14,
    "n_excluded": 2,
    "exclusion_reasons": {
      "CORYLUS": "Berlin: 456 samples (<500)",
      "MALUS": "Leipzig: 487 samples (<500)"
    }
  },
  "separability_analysis": {
    "jm_matrix_source": "temporal_selection.json aggregated",
    "poorly_separable_pairs": [
      {
        "genus_1": "PRUNUS",
        "genus_2": "PYRUS",
        "jm_distance": 0.87,
        "interpretation": "hard to separate"
      }
    ],
    "proposed_groups": {
      "rosaceae": {
        "genera": ["PRUNUS", "MALUS", "PYRUS"],
        "mean_intra_group_jm": 0.92,
        "reasoning": "Rosaceae family, similar phenology"
      }
    }
  },
  "decision": {
    "strategy_applied": "exclude_low_sample",
    "final_genera_count": 14,
    "final_genera_list": ["ACER", "AESCULUS", ...],
    "grouping_applied": false,
    "reasoning": "14 genera have ≥500 samples in both cities and JM > 1.0 pairwise. No grouping needed for statistical power."
  },
  "impact_assessment": {
    "berlin_samples_retained": 468234,
    "leipzig_samples_retained": 89432,
    "retention_rate_berlin": 0.992,
    "retention_rate_leipzig": 0.982,
    "kl_divergence_change": 0.023
  }
}
```

---

### Component 2: exp_10→exp_11 Renaming & Integration

**Files to Modify:**

#### 2.1 Rename Notebook

```bash
# File operation
mv notebooks/exploratory/exp_10_algorithm_comparison.ipynb \
   notebooks/exploratory/exp_11_algorithm_comparison.ipynb
```

#### 2.2 Modify exp_11 - Add Config Import (Cell 6, after data loading)

**ADD new cell:**

```python
# Cell: Load Genus Selection Configuration
config_path = BASE_PATH / "phase_3_experiments/metadata/genus_selection_final.json"

if config_path.exists():
    with open(config_path) as f:
        genus_config = json.load(f)
    FINAL_GENERA = genus_config["decision"]["final_genera_list"]
    print(f"✅ Loaded genus config: {len(FINAL_GENERA)} genera selected")
    print(f"   Excluded: {genus_config['validation_results']['excluded_genera']}")
else:
    # Fallback: use all genera in data
    FINAL_GENERA = None
    print("⚠️  No genus_selection_final.json found - using all genera in data")

# Filter datasets
if FINAL_GENERA is not None:
    original_count = len(berlin_train)
    berlin_train = berlin_train[berlin_train["genus_latin"].isin(FINAL_GENERA)]
    leipzig_finetune = leipzig_finetune[leipzig_finetune["genus_latin"].isin(FINAL_GENERA)]
    print(f"   Berlin: {original_count} → {len(berlin_train)} samples")
```

#### 2.3 Move Output Folders

```bash
# If exp_10 outputs already exist
mv outputs/phase_3_experiments/figures/exp_10_algorithm_comparison \
   outputs/phase_3_experiments/figures/exp_11_algorithm_comparison

mv outputs/phase_3_experiments/metadata/exp_10_*.json \
   outputs/phase_3_experiments/metadata/exp_11_*.json
```

---

### Component 3: Config System Integration

#### 3.1 Create Template Config

**File:** `configs/experiments/genus_selection.yaml`

```yaml
version: 1.0
description: |
  Genus selection configuration for Phase 3 experiments.
  Defines minimum sample thresholds and separability criteria.

thresholds:
  min_samples_per_genus: 500
  jm_distance_minimum: 1.0 # Pairs below this are "poorly separable"
  sample_safety_margin: 1.2 # 20% buffer over minimum

strategies:
  exclude_low_sample:
    description: "Remove genera with <500 samples in either city"
    priority: 1

  group_similar:
    description: "Group genera with JM < threshold into combined classes"
    priority: 2
    enabled: false

  hybrid:
    description: "Exclude very low (<400) + group similar (JM < 0.8)"
    priority: 3
    enabled: false

# Filled by exp_10 notebook execution
runtime_config:
  selected_genera: []
  excluded_genera: []
  genus_groups: {}
  execution_timestamp: null
```

#### 3.2 Runner Notebook Integration Pattern

**Add to:** `03b_berlin_optimization.ipynb`, `03c_transfer_evaluation.ipynb`, `03d_finetuning.ipynb`

**Cell to insert (after setup):**

```python
# Cell: Import Genus Selection Config
genus_config_path = BASE_PATH / "phase_3_experiments/metadata/genus_selection_final.json"

if genus_config_path.exists():
    with open(genus_config_path) as f:
        genus_config = json.load(f)
    SELECTED_GENERA = genus_config["decision"]["final_genera_list"]

    # Filter all datasets
    for split_name in ["berlin_train", "berlin_val", "berlin_test", "leipzig_finetune", "leipzig_test"]:
        if split_name in globals():
            df = globals()[split_name]
            original = len(df)
            df_filtered = df[df["genus_latin"].isin(SELECTED_GENERA)]
            globals()[split_name] = df_filtered
            print(f"{split_name}: {original} → {len(df_filtered)} samples")
else:
    print("⚠️  genus_selection_final.json not found - using all genera")
    SELECTED_GENERA = None
```

---

### Component 4: Documentation Updates

#### 4.1 Experiment Overview

**File:** `docs/documentation/03_Experiments/00_Experiment_Overview.md`

**Changes:**

1. **Experiment Table (line ~221):**

```markdown
| Notebook                    | Zweck                                   | Abhängigkeiten |
| --------------------------- | --------------------------------------- | -------------- |
| exp_10_genus_selection      | Genus-Auswahl validieren & finalisieren | exp_09, 03a    |
| exp_11_algorithm_comparison | 4 Algorithmen vergleichen               | exp_10         |
```

2. **Dependency Graph (line ~240):**

```markdown
exp_08 ──→ exp_08b ──→ exp_08c ──→ exp_09 ──→ 03a ──→ exp_10 ──→ exp_11 ──→ 03b ──→ 03c ──→ 03d
↓
genus_selection_final.json
↓
[used by 03b, 03c, 03d]
```

3. **Hyperparameter Tuning Section (line ~168):**

```markdown
1. **Coarse Grid Search** (exp_11): Grobe Hyperparameter-Bereiche, wenige Kombinationen
```

#### 4.2 Setup Fixierung

**File:** `docs/documentation/03_Experiments/01_Setup_Fixierung.md`

**ADD new section after exp_09 (before "Zusammenfassung"):**

```markdown
---

## Genus Selection Validation (exp_10)

**Ausführungsdatum:** [PENDING]
**Status:** [PENDING]
**Zweck:** Validierung dass alle Genera nach Phase 2c Filterung noch ≥500 Samples haben

### Problem

Phase 1 filtert Baumkataster auf Genera mit ≥500 Samples in beiden Städten → 30 viable genera.
Phase 2c wendet Proximity-Filter (5m) und Outlier-Removal an → reduziert Samples um ~20%.

**Risiko:** Einige Genera könnten unter 500-Sample-Threshold gefallen sein.

### Analysen

1. **Sample Count Validation:** Tatsächliche Genus-Counts nach Phase 2c
2. **Sample Sufficiency:** Vergleich mit Literatur (RF needs ~100 samples/class)
3. **Separability Matrix:** JM-Distanzen zwischen Genus-Paaren (aus temporal_selection.json)
4. **Genus Grouping:** Hierarchisches Clustering für potenzielle Gruppen
5. **Final Decision:** Exclude low-sample genera vs. group similar genera

### Entscheidung

**[WIRD NACH AUSFÜHRUNG AUSGEFÜLLT]**

**Gewählte Strategie:** [exclude_low_sample | group_similar | hybrid]

**Finale Genus-Liste:** [N genera]

- Ausgeschlossene Genera: [...]
- Gruppierte Genera: [...]

**Reasoning:** [...]

### Output

- **Config:** `outputs/phase_3_experiments/metadata/genus_selection_final.json`
- **Visualisierungen:**
  - `genus_sample_counts.png` - Bar chart mit 500-Sample-Threshold
  - `jm_separability_heatmap.png` - Genus-Genus Separabilität
  - `genus_dendrogram.png` - Hierarchisches Clustering
```

#### 4.3 Ergebnisse

**File:** `docs/documentation/03_Experiments/05_Ergebnisse.md`

**Changes:**

1. **Line 16 (Progress Update):**

```markdown
- **Runner Phase:** 🔄 In Arbeit (03a Setup complete, exp_10 Genus Selection complete, exp_11 Algorithm Comparison laufend)
```

2. **ADD new section after exp_09 (line ~362):**

```markdown
---

### Exp 10: Genus Selection Validation

**Ausführungsdatum:** [PENDING]
**Status:** [PENDING]
**Zweck:** Validierung der Genus-Auswahl nach Phase 2c Filterung und Finalisierung der Klassenliste

**Datenbasis:**
- **Berlin Train:** 471.815 Bäume (nach Proximity + Outlier Filter)
- **Leipzig Finetune:** 91.080 Bäume
- **Ursprüngliche Genera (Phase 1):** 30

---

#### Analyse 1: Sample Count Validation

**Methode:** Zählung verfügbarer Samples pro Genus in beiden Städten nach Phase 2c

**Ergebnis:** [PENDING]

**Implikation:** [PENDING]

---

#### Analyse 2: Separability Assessment

**Methode:** JM-Distanz-Matrix aus temporal_selection.json aggregiert

**Ergebnis:** [PENDING]

**Implikation:** [PENDING]

---

#### Finale Entscheidung

**Gewählte Strategie:** [PENDING]

**Finale Genus-Liste:** [N genera]

**Begründung:** [PENDING]

---

### Exp 11: Algorithm Comparison

**Status:** 🔄 In Arbeit
**Abhängigkeit:** exp_10 (verwendet finale Genus-Liste)

[Rest of exp_10 content moved here]
```

#### 4.4 Phase 2 Methodische Erweiterungen

**File:** `docs/documentation/02_Feature_Engineering/05_Methodische_Erweiterungen.md`

**ADD new section at end:**

```markdown
---

## 6. Genus-Auswahl-Validierung (Post-Phase 2c Filter-Kaskade)

**Problem:**
MIN_SAMPLES_PER_GENUS = 500 wird in Phase 1 (Datenverarbeitung) angewendet, resultiert in 30 viable genera. Phase 2c wendet jedoch weitere Filter an:

- Proximity-Filter (5m): Entfernt 20% der Samples
- Outlier-Removal: Entfernt weitere ~1-2% der Samples

**Folge:** Genus-Sample-Counts können unter 500-Threshold fallen.

### Empfohlener Workflow
```

Phase 1: filter_viable_genera()
↓ 30 genera mit ≥500 Samples
Phase 2a: Feature Extraction
↓ (keine Genus-Änderung)
Phase 2b: Data Quality (Outlier Detection)
↓ (Outlier markiert, nicht entfernt)
Phase 2c: Final Preparation (Proximity Filter + Outlier Removal)
↓ ~20% Sample-Reduktion
Phase 3 exp_10: Genus Selection Validation ← **NEU**
↓ Re-Validierung der 500-Sample-Threshold
↓ Finale Genus-Liste (12-30 Genera)
Phase 3 exp_11+: Algorithm Training
↓ Nutzt validierte Genus-Liste

```

### Implementierung

**Ort:** `notebooks/exploratory/exp_10_genus_selection_validation.ipynb`

**Analysen:**
1. Tatsächliche Sample-Counts nach Phase 2c
2. Statistische Sufficiency (Literatur: ~100 Samples/Klasse für RF)
3. Genus-Separabilität (JM-Distanz-Matrix)
4. Optionale Genus-Gruppierung

**Output:** `genus_selection_final.json` (genutzt von allen Phase 3 Runner Notebooks)

### Begründung

- **Statistische Power:** Random Forests benötigen ~100-200 Samples pro Klasse für robuste Klassifikation (Breiman, 2001)
- **Transfer-Robustheit:** Genus muss in beiden Städten ausreichend repräsentiert sein
- **Reproduzierbarkeit:** Explizite Genus-Liste in Config ermöglicht exaktes Reproduzieren
- **Methodische Transparenz:** Dokumentation warum bestimmte Genera ausgeschlossen wurden

### Alternative Strategien

**Strategie 1: Exclude Low-Sample** (empfohlen)
- Entferne Genera mit <500 Samples in mindestens einer Stadt
- Einfach, klar interpretierbar
- **Nachteil:** Verlust von Genera

**Strategie 2: Group Similar**
- Bilde Gruppen aus schlecht separierbaren Genera (JM < 1.0)
- Z.B. Rosaceae-Gruppe: PRUNUS + MALUS + PYRUS
- **Vorteil:** Mehr Samples pro Klasse
- **Nachteil:** Verlust von Granularität

**Strategie 3: Hybrid**
- Exclude sehr niedrige (<400 Samples)
- Group ähnliche (JM < 0.8)
- **Vorteil:** Balanciert Granularität und Power
- **Nachteil:** Komplexere Interpretation

### Referenzen

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Belgiu, M., & Drăguţ, L. (2016). Random forest in remote sensing: A review of applications and future directions. *ISPRS Journal of Photogrammetry and Remote Sensing*, 114, 24-31.
```

#### 4.5 Figure Documentation

**File:** `docs/presentation/FIGURE_DOCUMENTATION.md`

**Changes:**

1. **Line 553 (Folder Reference):**

```markdown
**Ordner:** `exp_10_genus_selection/`
**Ordner:** `exp_11_algorithm_comparison/`
```

2. **ADD new section:**

```markdown
---

## Phase 3a: Genus Selection Validation

**Ordner:** `exp_10_genus_selection/`

| Grafik                  | Datei                         | Beschreibung                                                                | Verwendung                       |
| ----------------------- | ----------------------------- | --------------------------------------------------------------------------- | -------------------------------- |
| Sample Count Validation | `genus_sample_counts.png`     | Bar chart: Samples pro Genus (Berlin + Leipzig), 500-Sample-Threshold-Linie | Methodenabschnitt: Genus-Auswahl |
| JM Separability Heatmap | `jm_separability_heatmap.png` | Heatmap: Genus-Genus JM-Distanzen (aggregiert über Monate)                  | Separabilitäts-Analyse           |
| Genus Dendrogram        | `genus_dendrogram.png`        | Hierarchisches Clustering basierend auf spektraler Ähnlichkeit              | Genus-Gruppierungs-Exploration   |

**Kontext:**
Validierung dass alle Genera nach Phase 2c Filterung noch statistisch ausreichend Samples haben (≥500). Identifikation von schlecht separierbaren Genus-Paaren für optionale Gruppierung.
```

---

### Component 5: Code Integration (if needed)

**Potential new utility function:**

**File:** `src/urban_tree_transfer/utils/genus_selection.py` (NEW)

```python
"""Genus selection utilities for Phase 3 experiments."""

from pathlib import Path
import json
from typing import Any

import pandas as pd


def load_genus_selection_config(
    base_path: Path | str = "outputs",
) -> dict[str, Any] | None:
    """Load genus_selection_final.json if exists.

    Args:
        base_path: Base path to outputs folder

    Returns:
        Config dict or None if file doesn't exist
    """
    config_path = Path(base_path) / "phase_3_experiments/metadata/genus_selection_final.json"

    if not config_path.exists():
        return None

    with open(config_path) as f:
        return json.load(f)


def filter_by_genus_config(
    df: pd.DataFrame,
    config: dict[str, Any] | None,
    genus_column: str = "genus_latin",
) -> pd.DataFrame:
    """Filter DataFrame to selected genera from config.

    Args:
        df: DataFrame with genus column
        config: Config dict from load_genus_selection_config()
        genus_column: Name of genus column

    Returns:
        Filtered DataFrame (or original if config is None)
    """
    if config is None:
        return df

    selected_genera = config["decision"]["final_genera_list"]
    return df[df[genus_column].isin(selected_genera)].copy()


def get_final_genera_list(base_path: Path | str = "outputs") -> list[str] | None:
    """Get final genus list from config.

    Args:
        base_path: Base path to outputs folder

    Returns:
        List of genus names or None if config doesn't exist
    """
    config = load_genus_selection_config(base_path)
    if config is None:
        return None
    return config["decision"]["final_genera_list"]
```

**Usage in Runner Notebooks:**

```python
from urban_tree_transfer.utils.genus_selection import load_genus_selection_config, filter_by_genus_config

genus_config = load_genus_selection_config(BASE_PATH)
berlin_train = filter_by_genus_config(berlin_train, genus_config)
```

---

## 📋 Implementation Checklist

### Pre-Implementation

- [ ] **Review PRD:** All stakeholders approve approach
- [ ] **Backup exp_10:** Copy current `exp_10_algorithm_comparison.ipynb` to `exp_10_algorithm_comparison_BACKUP.ipynb`
- [ ] **Check exp_10 status:** If exp_10 already completed, results need validation against new genus list

### Phase 0: Phase 2c Filter Integration (PREREQUISITE - 1 hour)

- [ ] Update `notebooks/runners/02c_final_preparation.ipynb`
- [ ] Add MIN_SAMPLES_PER_GENUS filter cell before split export
- [ ] Export `genus_filter_phase2c.json` with final genera + excluded genera
- [ ] Test Phase 2c: validate 12-20 final genera
- [ ] Update `phase_2_final_summary.json`: include genus filter stats

### Phase 1: exp_10 Notebook (1-2 hours)

- [ ] Create `notebooks/exploratory/exp_10_genus_selection_validation.ipynb`
- [ ] Implement Section 1: Review Phase 2c Filter Results (load `genus_filter_phase2c.json`)
- [ ] Implement Section 2: JM Separability Matrix
- [ ] Implement Section 3: Genus Grouping Exploration
- [ ] Implement Section 4: Final Recommendation & Export
- [ ] Test notebook in Colab
- [ ] Execute notebook → generate `genus_selection_final.json`
- [ ] Review: separability analysis + grouping recommendations

### Phase 2: exp_10→exp_11 Renaming (30 min)

- [ ] Rename `exp_10_algorithm_comparison.ipynb` → `exp_11_algorithm_comparison.ipynb`
- [ ] Add genus config import cell to exp_11
- [ ] Add genus filtering logic to exp_11
- [ ] Move `exp_10_algorithm_comparison/` outputs → `exp_11_algorithm_comparison/`
- [ ] Update exp_11 metadata exports (change "exp_10" → "exp_11" in JSON outputs)

### Phase 3: Config System (30 min)

- [ ] Create `configs/experiments/genus_selection.yaml` template
- [ ] Validate `genus_selection_final.json` schema matches documentation
- [ ] Add genus config import to `03b_berlin_optimization.ipynb`
- [ ] Add genus config import to `03c_transfer_evaluation.ipynb` (if exists)
- [ ] Add genus config import to `03d_finetuning.ipynb` (if exists)

### Phase 4: Documentation (1 hour)

- [ ] Update `00_Experiment_Overview.md` (Table + Dependency Graph)
- [ ] Update `01_Setup_Fixierung.md` (Add exp_10 section)
- [ ] Update `05_Ergebnisse.md` (Add exp_10 results, rename exp_10→exp_11)
- [ ] Update `02_Feature_Engineering/05_Methodische_Erweiterungen.md` (Add Section 6)
- [ ] Update `FIGURE_DOCUMENTATION.md` (Add exp_10 section, update exp_11)

### Phase 5: Phase 2c Integration (1 hour)

- [ ] Update `02c_final_preparation.ipynb`: Add genus filter cell before split export
- [ ] Add `genus_filter_phase2c.json` export
- [ ] Test Phase 2c filter: validate expected 12-20 final genera
- [ ] Update `phase_2_final_summary.json`: include genus filter stats

### Phase 6: Optional Code Utils (30 min)

- [ ] Create `src/urban_tree_transfer/utils/genus_selection.py`
- [ ] Implement `load_genus_selection_config()`
- [ ] Implement `filter_by_genus_config()`
- [ ] Add tests to `tests/utils/test_genus_selection.py`

### Post-Implementation

- [ ] **Execute exp_10:** Run notebook, validate results
- [ ] **Validate exp_11:** Check if results change with filtered genus list
- [ ] **Update CHANGELOG.md:** Document exp_10 addition and exp_10→exp_11 renumbering
- [ ] **Commit changes:** Single commit with message "feat: Add exp_10 genus selection validation, rename exp_10→exp_11"

---

## 🚨 Risks & Mitigation

### Risk 1: exp_10 Algorithm Comparison bereits abgeschlossen

**Wahrscheinlichkeit:** Hoch (Notebook läuft bereits)

**Impact:** Hoch (Ergebnisse müssen neu validiert werden)

**Mitigation:**

- exp_11 Ergebnisse mit alten exp_10 Ergebnissen vergleichen
- Falls Genus-Liste identisch (14-16 Genera): Keine Neuberechnung nötig
- Falls Genus-Liste reduziert (z.B. auf 12): Champion-Configs erneut evaluieren
- **Worst-Case:** Nur Champion-Evaluation wiederholen (~2-3h), nicht gesamte Coarse Grid Search

### Risk 2: Zu viele Genera fallen unter Threshold

**Wahrscheinlichkeit:** Mittel (Proximity-Filter entfernt ~20% Samples)

**Impact:** Mittel (Weniger Klassen → eventuell zu geringe Komplexität)

**Mitigation:**

- Falls <10 Genera übrig: Diskussion ob Proximity-Threshold zu streng (5m → 3m?)
- Alternative: Gruppierung ähnlicher Genera statt Ausschluss
- **Entscheidungsregel:** Minimum 8 Genera für wissenschaftliche Validität

### Risk 3: JM-Distanz-Daten nicht ausreichend

**Wahrscheinlichkeit:** Niedrig (temporal_selection.json enthält JM-Werte)

**Impact:** Mittel (Separabilitäts-Analyse nicht möglich)

**Mitigation:**

- Fallback: Nur Sample-Count-Validierung, keine Grouping-Analyse
- JM-Daten aus temporal_selection.json sind ausreichend für Heatmap
- Bei Bedarf: Literatur-basierte Genus-Familien (z.B. Rosaceae, Betulaceae)

### Risk 4: Breaking Changes in Runner Notebooks

**Wahrscheinlichkeit:** Niedrig (Backwards-kompatibel durch Fallback)

**Impact:** Hoch (03b/03c/03d können nicht ausgeführt werden)

**Mitigation:**

- Genus-Filtering hat Fallback: Falls `genus_selection_final.json` fehlt, nutze alle Genera
- Testen mit und ohne Config-Datei
- Clear error messages wenn Config-Format falsch

---

## 🎯 Success Metrics

**Quantitative:**

- [ ] **Final Genus Count:** 12-18 (valide Range)
- [ ] **Sample Retention:** >95% von Phase 2c Samples behalten
- [ ] **KL-Divergence Change:** <0.05 (class distribution bleibt stabil)
- [ ] **Notebook Execution Time:** <10 Minuten
- [ ] **All Tests Pass:** genus_selection.py utility functions

**Qualitative:**

- [ ] **Methodische Rechtfertigung:** Klare Begründung in Dokumentation warum N Genera gewählt
- [ ] **Reproduzierbarkeit:** Config-JSON ermöglicht exaktes Nachmachen
- [ ] **Integration:** Runner Notebooks nutzen Config ohne Breaking Changes
- [ ] **Visualizations:** Dendrogram und Heatmap sind publication-ready

---

## 📅 Timeline

**Total Estimated Time:** 4-6 hours implementation + 1-2 hours execution

| Phase | Tasks                     | Time   | Who        | Dependencies        |
| ----- | ------------------------- | ------ | ---------- | ------------------- |
| Pre   | PRD Review, Backup        | 30 min | Team       | -                   |
| 1     | exp_10 Notebook Creation  | 2-3h   | Developer  | Phase 2 complete    |
| 2     | exp_10→exp_11 Renaming    | 30 min | Developer  | Phase 1 complete    |
| 3     | Config System Integration | 30 min | Developer  | Phase 1 complete    |
| 4     | Documentation Updates     | 1h     | Developer  | Phase 1-3 complete  |
| 5     | Optional Code Utils       | 30 min | Developer  | -                   |
| Post  | Execution & Validation    | 1-2h   | Researcher | All phases complete |

**Critical Path:**

```
PRD → exp_10 Creation → Execute exp_10 → exp_11 Integration → Documentation
```

**Earliest Start:** ASAP (exp_10 algorithm comparison läuft parallel)

**Deadline:** Before 03b Berlin Optimization starts (CRITICAL)

---

## 🔗 Related Documents

- **PRD 003a_setup.md:** Setup-Entscheidungen (CHM, Outlier, Features)
- **PRD 003b_berlin.md:** Berlin Optimization (nutzt finale Genus-Liste)
- **PRD exp_07_cross_city_baseline.md:** Cross-City Baseline (ähnliche Analyse-Struktur)
- **docs/documentation/01_Data_Processing/01_Data_Processing_Methodik.md:** MIN_SAMPLES_PER_GENUS Definition
- **docs/documentation/02_Feature_Engineering/04_Feature_Engineering_Ergebnisse.md:** Phase 2 Ergebnisse (30 genera)
- **src/urban_tree_transfer/config/constants.py:** MIN_SAMPLES_PER_GENUS = 500

---

## Appendix A: JM-Distance Extraction Logic

**Problem:** `temporal_selection.json` enthält JM-Distanzen per month, aber nicht als Matrix.

**Solution:** Rekonstruiere Genus-Genus-Matrix aus monatlichen JM-Werten

```python
def extract_jm_matrix_from_temporal_metadata(metadata_path: Path) -> tuple[np.ndarray, list[str]]:
    """Extract genus-genus JM-distance matrix from temporal_selection.json.

    Returns:
        (jm_matrix, genus_names) where jm_matrix[i,j] is aggregated JM-distance
    """
    with open(metadata_path) as f:
        meta = json.load(f)

    viable_genera = meta["viable_genera"]
    n_genera = len(viable_genera)

    # NOTE: temporal_selection.json structure needs to be inspected
    # Expected: monthly_jm_statistics has genus-pair JM values per month

    # Placeholder: Assume we have monthly_pair_distances
    # Structure: {"month_1": {"ACER_TILIA": 1.23, "ACER_QUERCUS": 1.45, ...}, ...}

    jm_matrix = np.zeros((n_genera, n_genera))

    # Aggregate across months (mean JM-distance)
    # ...implementation depends on actual JSON structure...

    # For diagonal: JM(genus, genus) = 2.0 (perfect separability from self)
    np.fill_diagonal(jm_matrix, 2.0)

    return jm_matrix, viable_genera
```

**Action:** Inspect `temporal_selection.json` structure to implement extraction logic.

---

## Appendix B: Sample Code Snippets

### B.1 Visualization: Sample Count Bar Chart

```python
def plot_sample_counts(berlin_counts, leipzig_counts, min_threshold=500):
    """Bar chart showing sample counts per genus with threshold line."""
    genera = sorted(set(berlin_counts.index) | set(leipzig_counts.index))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(genera))
    width = 0.35

    berlin_vals = [berlin_counts.get(g, 0) for g in genera]
    leipzig_vals = [leipzig_counts.get(g, 0) for g in genera]

    ax.bar(x - width/2, berlin_vals, width, label='Berlin Train', alpha=0.8)
    ax.bar(x + width/2, leipzig_vals, width, label='Leipzig Finetune', alpha=0.8)
    ax.axhline(y=min_threshold, color='r', linestyle='--', linewidth=2,
               label=f'Min Threshold ({min_threshold})')

    ax.set_xlabel('Genus')
    ax.set_ylabel('Sample Count')
    ax.set_title('Genus Sample Counts after Phase 2c Filtering')
    ax.set_xticks(x)
    ax.set_xticklabels(genera, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig
```

### B.2 Decision Logic

```python
def make_genus_selection_decision(
    berlin_counts: pd.Series,
    leipzig_counts: pd.Series,
    jm_matrix: np.ndarray,
    genera_names: list[str],
    min_samples: int = 500,
    jm_threshold: float = 1.0,
    strategy: str = "exclude_low_sample"
) -> dict:
    """Make final genus selection decision based on strategy."""

    # Identify viable genera (≥min_samples in both cities)
    viable_berlin = set(berlin_counts[berlin_counts >= min_samples].index)
    viable_leipzig = set(leipzig_counts[leipzig_counts >= min_samples].index)
    viable_both = viable_berlin & viable_leipzig

    excluded = set(genera_names) - viable_both

    if strategy == "exclude_low_sample":
        final_genera = sorted(list(viable_both))
        groups = {}
        reasoning = f"{len(final_genera)} genera have ≥{min_samples} samples in both cities."

    elif strategy == "group_similar":
        # Identify poorly separable pairs
        poorly_sep = identify_poorly_separable_pairs(jm_matrix, genera_names, jm_threshold)
        # ... grouping logic ...
        final_genera = sorted(list(viable_both))  # Placeholder
        groups = {}  # Placeholder
        reasoning = "Grouping logic not yet implemented."

    elif strategy == "hybrid":
        # Combine exclusion + grouping
        final_genera = sorted(list(viable_both))  # Placeholder
        groups = {}  # Placeholder
        reasoning = "Hybrid strategy not yet implemented."

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {
        "strategy_applied": strategy,
        "final_genera_count": len(final_genera),
        "final_genera_list": final_genera,
        "excluded_genera": sorted(list(excluded)),
        "genus_groups": groups,
        "reasoning": reasoning
    }
```

---

## Questions & Open Items

1. **JM-Distance Structure:** Is temporal_selection.json structured with per-month genus-pair JM values? Need to verify actual JSON structure.
2. **Grouping Priority:** Should we implement grouping strategies in exp_10 or leave for future extension?
   - **Recommendation:** Implement "exclude_low_sample" only in exp_10, document grouping as "future work"

3. **exp_10 Algorithm Comparison Status:** If already completed, do we rerun with filtered genus list?
   - **Decision Rule:** Only rerun if final genus list changes by >2 genera

4. **Minimum Viable Genera:** What's the minimum acceptable number of genera for scientific validity?
   - **Recommendation:** 8 genera minimum (literature: multi-class needs at least 6-8 classes)

---

**End of PRD**
