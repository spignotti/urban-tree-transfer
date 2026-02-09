# PRD: exp_07 Cross-City Baseline Analysis

**PRD ID:** exp_07
**Status:** Pending Implementation
**Created:** 2026-02-09
**Dependencies:** Phase 2 outputs (berlin_*.parquet, leipzig_*.parquet)
**Critical Path:** No (optional, parallel to exp_08-10)

---

## 🎯 Goal

**Was soll gebaut werden:**
Ein exploratives Notebook zur deskriptiven Analyse der Unterschiede zwischen Berlin- und Leipzig-Datensätzen **vor** dem Training. Ziel ist Hypothesengenerierung für die Transfer-Evaluation (PRD 003c).

**Erfolgskriterium:**
Quantifizierung und Visualisierung der Domain-Shift-Größe zwischen den Städten anhand von:
- Klassenverteilungen
- Phänologischen Profilen
- Strukturellen Unterschieden (CHM)
- Feature-Distributionen (spektrale Signaturen)
- Statistischen Effektstärken (Cohen's d)

---

## 🧑‍💻 User & Use Case

**Zielgruppe:** Forschende, die Transfer-Learning-Performance interpretieren wollen

**Hauptanwendung:**
- Vorab-Verständnis der Domain-Unterschiede
- Hypothesengenerierung für Transfer-Gap-Erklärung
- Literaturvalidierung (z.B. "Nadelbäume haben stabilere spektrale Signaturen")

**Gelöstes Problem:**
Bisher ist unklar, **wie groß** der Domain Shift zwischen Berlin und Leipzig ist. Sind die Städte spektral sehr ähnlich (→ guter Transfer erwartet) oder sehr unterschiedlich (→ hoher Transfer-Gap erwartet)?

---

## ✅ Success Criteria

- [ ] **6 Analysen implementiert:** Class Distribution, Phenological Profiles, CHM Distribution, Feature Distribution, Cohen's d Heatmap, Correlation Structure
- [ ] **Visualisierungen erzeugt:** Alle Plots in `outputs/phase_3_experiments/figures/exp_07_baseline/` gespeichert
- [ ] **Effektstärken quantifiziert:** Cohen's d für top-5 Genera × top-20 Features berechnet
- [ ] **Spektrale Überlappung analysiert:** Feature-Distributions zeigen Overlap-Grad
- [ ] **Notebook läuft in Colab:** Installiert package von GitHub, lädt Daten von Google Drive
- [ ] **Keine Outputs für andere Notebooks:** Rein deskriptiv, keine JSON-Outputs (laut PRD)

---

## 🧩 Context & References

### Wichtige Dateien

- **PRD 003_phase3_complete.md:** Abschnitt 4.1 beschreibt exp_07 Details
- **docs/documentation/03_Experiments/00_Experiment_Overview.md:** Experimentelle Struktur
- **docs/literature/research_reports/02_Transfer_Learning_Framework.md:** Domain Shift Literatur

### Ähnliche Features im Code

- **exp_01_temporal_analysis.ipynb:** Phänologische Analyse (NDVI/EVI Plots)
- **exp_03_correlation_analysis.ipynb:** Korrelationsmatrizen
- **exp_02_chm_assessment.ipynb:** CHM Violin Plots
- **src/urban_tree_transfer/utils/visualization.py:** Standardisierte Plot-Funktionen

### Externe Referenzen

- Cohen's d: Effektstärke für Mittelwertunterschiede (small: 0.2, medium: 0.5, large: 0.8)
- Ridge plots: Überlappende Density Plots für Feature-Distributionen

---

## 🏗️ Technical Details

### Main Components

**Notebook:** `notebooks/exploratory/exp_07_cross_city_baseline.ipynb`

**6 Kern-Analysen:**

#### 1. Class Distribution Comparison
- **Input:** `berlin_train.parquet`, `leipzig_finetune.parquet`
- **Analyse:** `value_counts()` für `genus_latin`
- **Visualisierung:** Stacked Bar Chart (Genus × City)
- **Output:** `genus_distribution_comparison.png`

#### 2. Phenological Profiles (Top-5 Genera)
- **Input:** NDVI/EVI Spalten für top-5 häufigste Genera
- **Analyse:** Monatliche Zeitreihen gruppiert nach Genus + City
- **Visualisierung:** Line Plot (12 Monate, separate Linien für Berlin/Leipzig)
- **Output:** `phenological_profiles_top5.png`
- **Interpretation:** Zeigt saisonale Unterschiede (z.B. unterschiedliche Blattaustrieb-Zeitpunkte)

#### 3. CHM Distribution per Genus
- **Input:** CHM-Features (`CHM_mean`, `CHM_std`, `CHM_percentile_*`)
- **Analyse:** Violin Plots für `CHM_mean` pro Genus × City
- **Visualisierung:** Side-by-side Violin Plots
- **Output:** `chm_violin_per_genus.png`
- **Interpretation:** Strukturelle Unterschiede (Baumhöhe, Kronenvolumen)

#### 4. Feature Distribution Overlap
- **Input:** Top-20 wichtigste Features (aus exp_09 Feature Importance bekannt)
- **Analyse:** Kernel Density Estimation (KDE) für jedes Feature × City
- **Visualisierung:** Ridge Plots (überlappende KDE-Kurven)
- **Output:** `feature_distribution_overlap.png`
- **Interpretation:** Zeigt spektrale Signatur-Überlappung

#### 5. Statistical Differences (Cohen's d Heatmap)
- **Input:** Top-5 Genera × Top-20 Features
- **Analyse:** Cohen's d berechnen zwischen Berlin/Leipzig pro Genus-Feature-Kombination
- **Visualisierung:** Heatmap (Genus × Feature, Farbe = Effektstärke)
- **Output:** `cohens_d_heatmap.png`
- **Interpretation:** Quantifiziert Domain Shift (große Effekte = Transfer schwierig)

**Cohen's d Berechnung:**
```python
def cohens_d(berlin_values, leipzig_values):
    n1, n2 = len(berlin_values), len(leipzig_values)
    var1, var2 = np.var(berlin_values, ddof=1), np.var(leipzig_values, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(berlin_values) - np.mean(leipzig_values)) / pooled_std
```

#### 6. Correlation Structure Comparison
- **Input:** Alle Features (Berlin/Leipzig separat)
- **Analyse:** Pearson-Korrelation zwischen Features
- **Visualisierung:** Side-by-side Heatmaps (Berlin | Leipzig)
- **Output:** `correlation_structure_comparison.png`
- **Interpretation:** Zeigt, ob Feature-Interaktionen in beiden Städten ähnlich sind

---

### Dateien zu erstellen/bearbeiten

**Neu:**
- `notebooks/exploratory/exp_07_cross_city_baseline.ipynb`

**Erweitern (falls nötig):**
- `src/urban_tree_transfer/utils/visualization.py` — Ridge Plot Funktion hinzufügen
- `src/urban_tree_transfer/experiments/evaluation.py` — `compute_cohens_d()` Funktion

---

### Dependencies

**Python Packages:**
- `scipy.stats` — Cohen's d Berechnung
- `seaborn` — Ridge Plots, Violin Plots
- `matplotlib` — Standard Plotting

---

## 🧪 Validation

### Manuelle Tests

```bash
# Notebook in Colab ausführen:
# 1. Mount Google Drive
# 2. Install package: !pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q
# 3. Run all cells
# 4. Check: outputs/phase_3_experiments/figures/exp_07_baseline/ enthält 6 PNG-Dateien
```

### Erwartete Outputs

```
outputs/phase_3_experiments/figures/exp_07_baseline/
├── genus_distribution_comparison.png
├── phenological_profiles_top5.png
├── chm_violin_per_genus.png
├── feature_distribution_overlap.png
├── cohens_d_heatmap.png
└── correlation_structure_comparison.png
```

---

## 🚫 Anti-Patterns to Avoid

- ❌ **Nicht:** Modelle trainieren (rein deskriptiv!)
- ❌ **Nicht:** JSON-Outputs erzeugen (keine Entscheidungen, nur Visualisierung)
- ❌ **Nicht:** Leipzig-Daten für Setup-Entscheidungen nutzen (nur Vergleich, keine Ablation)
- ❌ **Nicht:** Statistische Tests durchführen (nur Effektstärken beschreiben)

---

## 📝 Implementation Notes

### Reihenfolge

1. **Daten laden:** `berlin_train.parquet`, `leipzig_finetune.parquet` (baseline variant)
2. **Analysen 1-3:** Class Distribution, Phenology, CHM (einfach)
3. **Analysen 4-6:** Feature Distributions, Cohen's d, Correlation (aufwendiger)

### Cohen's d Interpretation

| Effektstärke | Interpretation                  |
| ------------ | ------------------------------- |
| \|d\| < 0.2  | Negligible (Transfer sollte gut funktionieren) |
| 0.2-0.5      | Small (moderater Transfer-Gap erwartet) |
| 0.5-0.8      | Medium (deutlicher Transfer-Gap) |
| \|d\| > 0.8  | Large (hoher Transfer-Gap, Genus möglicherweise nicht transferierbar) |

### Hypothesen für Transfer-Evaluation (PRD 003c)

Basierend auf exp_07 Ergebnissen können wir hypothetisieren:
- **Hohe Feature-Overlap:** Guter Transfer erwartet
- **Große Cohen's d für Nadelbäume:** Widerspricht Literatur (Fassnacht 2016) → erfordert Diskussion
- **Unterschiedliche Korrelationsstruktur:** NN-Modelle könnten schlechter transferieren als ML

---

## 🔗 Weiterführende Analysen

Dieses Notebook **bereitet vor**:
- **PRD 003c (Transfer Evaluation):** Erklärt, warum manche Genera gut/schlecht transferieren
- **Error Analysis (03b):** Verknüpft Berlin-Fehler mit strukturellen Unterschieden

---

**Geschätzte Implementierungszeit:** 2-3 Stunden
**Priorität:** Optional (kann übersprungen werden, wenn Zeit knapp ist)
**Nutzen:** Hoch für Interpretation, niedrig für Performance
