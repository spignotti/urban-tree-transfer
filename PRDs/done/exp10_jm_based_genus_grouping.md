# PRD: JM-Distance Based Genus Grouping in exp_10

**Status:** 🟡 Ready for Implementation  
**Priority:** High  
**Target Notebook:** exp_10_genus_selection_validation.ipynb  
**Affected Files:** exp_10, 03a, setup_decisions.json  
**Estimated Effort:** 4-6 hours

---

## 1. Context & Problem Statement

### Current Situation (❌ Suboptimal)

**exp_10 verwendet aktuell Zentroid-Distanzen:**

```python
# Berechnet nur Abstand zwischen Genus-Mittelwerten
genus_centroids = combined.groupby('genus_latin')[features].mean()
distances = pdist(centroids_scaled, metric='euclidean')
```

**Probleme:**

- ✗ Ignoriert Intra-Klassen-Varianz (Streuung innerhalb der Genera)
- ✗ Keine Aussage über tatsächliche Klassifizierbarkeit
- ✗ Zwei Genera können weit auseinander liegen, aber massiv überlappen

**Beispiel:**

```
ACER:     ●●●●●●●●●●●●●●●  (hoch gestreut, σ=5)
QUERCUS:      ●●●●●●●●●●●●●●●  (hoch gestreut, σ=5)
→ Zentroid-Distanz: mittel
→ Realität: MASSIVE Überlappung → schlecht klassifizierbar!
```

### Proposed Solution (✅ Besser)

**Jeffries-Matusita (JM) Distance:**

- Berücksichtigt **Mittelwert UND Kovarianz-Struktur**
- Misst tatsächliche statistische Trennbarkeit
- Standard-Metrik in Remote Sensing für Feature Separability
- Werte: 0 (identisch) bis 2 (perfekt trennbar)

**Interpretation:**

- JM < 1.0: Schlecht trennbar → **Gruppieren**
- JM ≥ 1.8: Gut trennbar → Separate Klassen

---

## 2. Objectives & Success Criteria

### Primary Objectives

1. **Ersetze Distanz-Analyse durch JM-basierte Separability-Analyse**
2. **Verwende relativen Threshold** (z.B. "untere 20% gruppieren") statt festem Wert
3. **Erstelle aussagekräftige Gruppennamen** mit deutschen + lateinischen Namen
4. **Integriere fix_missing_genus_german()** für konsistente Metadaten
5. **Update 03a** um beide Name-Spalten zu mappen (genus_latin + genus_german)

### Success Criteria

- [ ] exp_10 berechnet vollständige JM-Matrix (pairweise für alle viable Genera)
- [ ] Relative Threshold-Strategie implementiert (kein hard-coded 0.5)
- [ ] JSON enthält genus_to_final_mapping für **beide** Spalten (latin + german)
- [ ] Visualisierung zeigt welche Genera gruppiert wurden
- [ ] 03a wendet Mapping auf beide Spalten an
- [ ] fix_missing_genus_german() in exp_10 UND 03a aufgerufen
- [ ] Alle Tests passed (KL-Divergence Validation)

---

## 3. Detailed Requirements

### 3.1 Data Loading Strategy

**Phase 1: Sample Count Validation (ALLE Splits)**

```python
# Lade ALLE Splits für umfassende Sample Count Validation
berlin_train, berlin_val, berlin_test = data_loading.load_berlin_splits(...)
leipzig_finetune, leipzig_test = data_loading.load_leipzig_splits(...)

# Prüfe Sample Counts über ALLE Splits
all_splits = [berlin_train, berlin_val, berlin_test, leipzig_finetune, leipzig_test]
combined_for_counts = pd.concat(all_splits, ignore_index=True)

# Validiere: ≥500 samples pro Genus in BEIDEN Städten
genus_counts_berlin = combined_for_counts[combined_for_counts['city']=='berlin'].groupby('genus_latin').size()
genus_counts_leipzig = combined_for_counts[combined_for_counts['city']=='leipzig'].groupby('genus_latin').size()

viable_genera = [g for g in all_genera
                 if genus_counts_berlin.get(g, 0) >= 500
                 and genus_counts_leipzig.get(g, 0) >= 500]
```

**Phase 2: JM-Analyse (NUR Berlin Train)**

```python
# Für Separability-Analyse: NUR Berlin Train verwenden
berlin_train_viable = berlin_train[berlin_train['genus_latin'].isin(viable_genera)].copy()

# Grund: Test-Daten dürfen Gruppierungsentscheidung nicht beeinflussen (Data Leakage)
# Leipzig nicht verwenden um Stadt-spezifische Bias zu vermeiden
```

**Rationale:**

- Sample Counts: Über alle Splits → realistische Verfügbarkeit
- JM-Analyse: Nur Train → keine Data Leakage, keine Stadt-Bias

---

### 3.2 JM-Distance Implementation

**Funktion 1: Bhattacharyya Distance**

```python
def bhattacharyya_distance(X_i: np.ndarray, X_j: np.ndarray) -> float:
    """
    Compute Bhattacharyya distance between two multivariate Gaussian distributions.

    Formula:
        B = 1/8 * (μ_i - μ_j)^T * Σ^-1 * (μ_i - μ_j)
            + 1/2 * ln(|Σ| / sqrt(|Σ_i| * |Σ_j|))

        where Σ = (Σ_i + Σ_j) / 2 (pooled covariance)

    Parameters
    ----------
    X_i, X_j : array-like, shape (n_samples, n_features)
        Samples from genus i and j

    Returns
    -------
    B : float
        Bhattacharyya distance
    """
    mu_i = np.mean(X_i, axis=0)
    mu_j = np.mean(X_j, axis=0)

    cov_i = np.cov(X_i, rowvar=False)
    cov_j = np.cov(X_j, rowvar=False)

    # Pooled covariance
    cov_pool = (cov_i + cov_j) / 2

    # Regularization für numerische Stabilität
    cov_pool += np.eye(cov_pool.shape[0]) * 1e-6

    # Term 1: Mahalanobis distance
    diff = mu_i - mu_j
    try:
        mahal = diff.T @ np.linalg.inv(cov_pool) @ diff
    except np.linalg.LinAlgError:
        # Fallback für singuläre Matrizen
        mahal = diff.T @ np.linalg.pinv(cov_pool) @ diff

    # Term 2: Determinant ratio
    det_pool = np.linalg.det(cov_pool)
    det_i = np.linalg.det(cov_i)
    det_j = np.linalg.det(cov_j)

    if det_pool <= 0 or det_i <= 0 or det_j <= 0:
        det_term = 0  # Fallback bei degeneraten Kovarianzen
    else:
        det_term = np.log(det_pool / np.sqrt(det_i * det_j))

    B = 0.125 * mahal + 0.5 * det_term
    return B
```

**Funktion 2: Jeffries-Matusita Distance**

```python
def jeffries_matusita_distance(X_i: np.ndarray, X_j: np.ndarray) -> float:
    """
    Compute JM distance (bounded [0, 2]).

    Formula:
        JM = 2 * (1 - exp(-B))

    Returns
    -------
    JM : float
        Jeffries-Matusita distance (0 = identical, 2 = perfectly separable)
    """
    B = bhattacharyya_distance(X_i, X_j)
    return 2 * (1 - np.exp(-B))
```

**Funktion 3: Pairwise JM Matrix**

```python
def compute_jm_matrix(df: pd.DataFrame,
                      feature_cols: list,
                      class_col: str = 'genus_latin') -> pd.DataFrame:
    """
    Compute pairwise JM distances for all classes.

    Returns
    -------
    jm_matrix : DataFrame
        Symmetric matrix (n_classes × n_classes) with JM distances
    """
    classes = sorted(df[class_col].unique())
    n_classes = len(classes)
    jm_matrix = np.zeros((n_classes, n_classes))

    for i, class_i in enumerate(classes):
        X_i = df[df[class_col] == class_i][feature_cols].values

        for j, class_j in enumerate(classes):
            if i >= j:
                continue  # Matrix ist symmetrisch

            X_j = df[df[class_col] == class_j][feature_cols].values
            jm_matrix[i, j] = jeffries_matusita_distance(X_i, X_j)
            jm_matrix[j, i] = jm_matrix[i, j]  # Symmetrie

    return pd.DataFrame(jm_matrix, index=classes, columns=classes)
```

---

### 3.3 Relative Threshold Strategy

**Problem:** Fester Threshold (z.B. JM < 1.0) kann unangemessen sein wenn:

- Alle Genera gut trennbar sind (JM > 1.5) → kein Gruppieren nötig
- Viele Genera schlecht trennbar (JM < 0.8) → zu viele Gruppen

**Lösung: Percentile-basierter Threshold**

```python
# Berechne JM-Matrix
jm_matrix = compute_jm_matrix(berlin_train_viable, feature_cols)

# Extrahiere Distanzen (ohne Diagonale = 0)
jm_values = jm_matrix.values[np.triu_indices_from(jm_matrix.values, k=1)]

# Option A: Untere 20% als "schlecht trennbar" definieren
PERCENTILE = 20
threshold = np.percentile(jm_values, PERCENTILE)

# Option B: Adaptive Strategie
if np.percentile(jm_values, 10) >= 1.5:
    # Fast alle gut trennbar → kein Gruppieren
    threshold = 0.0
    print("✅ Alle Genera gut trennbar (Q10 ≥ 1.5) - kein Gruppieren erforderlich")
elif np.percentile(jm_values, 50) < 1.0:
    # Median schlecht → toleranter Threshold
    threshold = np.percentile(jm_values, 30)
    print(f"⚠️ Viele Genera schlecht trennbar (Median < 1.0) - Threshold: {threshold:.2f}")
else:
    # Standard: Untere 20%
    threshold = np.percentile(jm_values, 20)
    print(f"ℹ️ Threshold (P20): {threshold:.2f}")

print(f"\nJM-Statistik:")
print(f"   Min:     {jm_values.min():.3f}")
print(f"   Q10:     {np.percentile(jm_values, 10):.3f}")
print(f"   Q25:     {np.percentile(jm_values, 25):.3f}")
print(f"   Median:  {np.median(jm_values):.3f}")
print(f"   Q75:     {np.percentile(jm_values, 75):.3f}")
print(f"   Max:     {jm_values.max():.3f}")
```

**Empfehlung:** Percentile 20 (untere 20% werden gruppiert) als Default, mit adaptiver Fallback-Logik.

---

### 3.4 Hierarchical Clustering

```python
# JM-Distanzen zu Clusterable Distance konvertieren
# Hohe JM → niedrige Distance (für Clustering)
clustering_distances = 2.0 - jm_matrix.values

# Condensed form für scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
condensed = squareform(clustering_distances)

# Ward Linkage
from scipy.cluster.hierarchy import linkage, fcluster
linkage_matrix = linkage(condensed, method='ward')

# Cut bei Threshold
distance_threshold = 2.0 - threshold  # Zurück-konvertieren
cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

# Erstelle Gruppen
genus_to_cluster = dict(zip(viable_genera, cluster_labels))
cluster_to_genera = {}
for genus, cluster_id in genus_to_cluster.items():
    if cluster_id not in cluster_to_genera:
        cluster_to_genera[cluster_id] = []
    cluster_to_genera[cluster_id].append(genus)

# Nur Clusters mit >1 Genus sind Gruppen
genus_groups = {}
for cluster_id, genera_list in cluster_to_genera.items():
    if len(genera_list) > 1:
        genera_list_sorted = sorted(genera_list)
        group_name = f"Gruppe {cluster_id}"
        genus_groups[group_name] = genera_list_sorted
```

---

### 3.5 German Name Mapping

**Strategie: Kombinierte Namen**

```python
# Fix fehlende deutsche Namen BEVOR Gruppierung
berlin_train_viable = data_loading.fix_missing_genus_german(berlin_train_viable)

# Erstelle German Name Mapping
genus_german_mapping = {}

# 1. Singleton Genera: Original-Namen behalten
for genus in viable_genera:
    original_german = berlin_train_viable[
        berlin_train_viable['genus_latin'] == genus
    ]['genus_german'].iloc[0]

    # Falls Singleton (nicht gruppiert)
    if genus_to_final[genus] == genus:
        genus_german_mapping[genus] = original_german

# 2. Gruppen: Kombinierte Namen
for group_name, genera_list in genus_groups.items():
    german_names = []
    for genus in genera_list:
        german = berlin_train_viable[
            berlin_train_viable['genus_latin'] == genus
        ]['genus_german'].iloc[0]
        german_names.append(german)

    # Format: "Ahorn / Linde / Esche"
    genus_german_mapping[group_name] = " / ".join(german_names)

print("\n✅ German Name Mapping:")
for final_class, german_name in sorted(genus_german_mapping.items()):
    if final_class.startswith("Gruppe"):
        original_genera = genus_groups[final_class]
        print(f"   {final_class:15s} = {german_name:40s} ({', '.join(original_genera)})")
    else:
        print(f"   {final_class:15s} = {german_name}")
```

**Beispiel Output:**

```
ACER             = Ahorn
TILIA            = Linde
Gruppe 1         = Prunus / Malus / Pyrus           (PRUNUS, MALUS, PYRUS)
Gruppe 2         = Eiche / Buche                    (QUERCUS, FAGUS)
```

---

### 3.6 Final Genus Mapping

```python
# Erstelle genus_to_final Mapping
genus_to_final = {}

# Singletons: genus -> genus
for cluster_id, genera_list in cluster_to_genera.items():
    if len(genera_list) == 1:
        genus = genera_list[0]
        genus_to_final[genus] = genus

# Gruppen: genus -> "Gruppe X"
for group_name, genera_list in genus_groups.items():
    for genus in genera_list:
        genus_to_final[genus] = group_name

final_genera_list = sorted(set(genus_to_final.values()))

print(f"\n✅ Final Genus Mapping:")
print(f"   Original viable genera: {len(viable_genera)}")
print(f"   Final classes:          {len(final_genera_list)}")
print(f"   Reduction:              {len(viable_genera) - len(final_genera_list)} genera merged")
```

---

### 3.7 Visualization Requirements

**Grafik 1: JM-Heatmap**

```python
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    jm_matrix,
    cmap='RdYlGn',  # Rot = schlecht trennbar, Grün = gut trennbar
    vmin=0, vmax=2,
    cbar_kws={'label': 'Jeffries-Matusita Distance'},
    square=True,
    ax=ax
)
ax.set_title('Genus-Genus Separability (JM Distance)', fontsize=14, fontweight='bold')
ax.set_xlabel('Genus', fontsize=12)
ax.set_ylabel('Genus', fontsize=12)

# Markiere Threshold
ax.text(0.02, 0.98, f'Threshold: {threshold:.2f} (P{PERCENTILE})',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

save_figure(fig, FIGURES_DIR / "jm_separability_heatmap.png")
```

**Grafik 2: Dendrogram mit Threshold**

```python
fig, ax = plt.subplots(figsize=(16, 8))

# Normalisiere Linkage für Visualisierung
max_dist = linkage_matrix[:, 2].max()
normalized_linkage = linkage_matrix.copy()
normalized_linkage[:, 2] = normalized_linkage[:, 2] / max_dist

from scipy.cluster.hierarchy import dendrogram
dendrogram(
    normalized_linkage,
    labels=viable_genera,
    ax=ax,
    leaf_font_size=9,
    leaf_rotation=90
)

# Threshold Line
normalized_threshold = (2.0 - threshold) / max_dist
ax.axhline(y=normalized_threshold, color='red', linestyle='--',
           linewidth=2, label=f'Cut Threshold (JM={threshold:.2f})')

ax.set_xlabel('Genus', fontsize=12)
ax.set_ylabel('Normalized Distance', fontsize=12)
ax.set_title('Hierarchical Clustering of Genera (JM-based)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

save_figure(fig, FIGURES_DIR / "jm_dendrogram.png")
```

**Grafik 3: Gruppierungs-Übersicht**

```python
if genus_groups:
    fig, ax = plt.subplots(figsize=(12, len(genus_groups) * 0.8 + 2))

    y_pos = np.arange(len(genus_groups))
    group_names = list(genus_groups.keys())
    group_sizes = [len(genus_groups[g]) for g in group_names]

    bars = ax.barh(y_pos, group_sizes, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_names)
    ax.set_xlabel('Number of Genera in Group', fontsize=12)
    ax.set_title('Genus Groups Formed by JM-based Clustering', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Annotationen mit Genus-Namen
    for i, (group_name, genera_list) in enumerate(genus_groups.items()):
        german_name = genus_german_mapping[group_name]
        ax.text(group_sizes[i] + 0.1, i,
                f"{', '.join(genera_list)}\n({german_name})",
                va='center', fontsize=8)

    plt.tight_layout()
    save_figure(fig, FIGURES_DIR / "genus_groups_overview.png")
else:
    print("ℹ️ No genus groups formed - all genera sufficiently separable")
```

---

### 3.8 JSON Export Structure

**In setup_decisions.json einfügen:**

```json
{
  "chm_strategy": { ... },
  "proximity_strategy": { ... },
  "outlier_strategy": { ... },
  "feature_set": { ... },
  "selected_features": [ ... ],

  "genus_selection": {
    "version": "2.0",
    "created": "2026-02-09T19:00:00Z",
    "notebook": "exp_10_genus_selection_validation.ipynb",
    "method": "jm_distance_based_separability",

    "config": {
      "min_samples_per_genus": 500,
      "sample_count_validation": "all_splits",
      "separability_analysis": "berlin_train_only",
      "clustering_method": "ward",
      "distance_metric": "jeffries_matusita",
      "threshold_strategy": "percentile_20",
      "threshold_value": 0.87,
      "kl_threshold": 0.15
    },

    "jm_statistics": {
      "min": 0.45,
      "q10": 0.72,
      "q25": 0.87,
      "median": 1.23,
      "q75": 1.65,
      "max": 1.98
    },

    "validation_results": {
      "viable_genera": ["ACER", "TILIA", "PRUNUS", ...],
      "excluded_genera": ["GENUS_X", "GENUS_Y"],
      "n_viable": 28,
      "n_excluded": 2,
      "exclusion_reasons": {
        "GENUS_X": "Berlin: 480, Leipzig: 520 (< 500 in Berlin)",
        "GENUS_Y": "Berlin: 550, Leipzig: 450 (< 500 in Leipzig)"
      }
    },

    "grouping_analysis": {
      "n_clusters": 25,
      "n_groups": 3,
      "genus_groups": {
        "Gruppe 1": ["MALUS", "PRUNUS", "PYRUS"],
        "Gruppe 2": ["ACER", "PLATANUS"],
        "Gruppe 3": ["QUERCUS", "FAGUS"]
      },
      "genus_to_final_mapping": {
        "ACER": "Gruppe 2",
        "MALUS": "Gruppe 1",
        "PRUNUS": "Gruppe 1",
        "PYRUS": "Gruppe 1",
        "TILIA": "TILIA",
        ...
      },
      "genus_german_mapping": {
        "Gruppe 1": "Apfel / Kirsche / Birne",
        "Gruppe 2": "Ahorn / Platane",
        "Gruppe 3": "Eiche / Buche",
        "TILIA": "Linde",
        ...
      }
    },

    "decision": {
      "strategy_applied": "jm_based_grouping_percentile_20",
      "final_genera_count": 25,
      "final_genera_list": ["TILIA", "FRAXINUS", "Gruppe 1", "Gruppe 2", "Gruppe 3", ...],
      "grouping_applied": true,
      "reasoning": "28 genera have ≥500 samples in both cities. JM-based separability analysis (P20 threshold=0.87) identified 3 groups of poorly separable genera. Final dataset has 25 classes."
    },

    "impact_assessment": {
      "samples_retained": 145832,
      "retention_rate": 0.987,
      "final_class_counts": {
        "berlin_train": { "TILIA": 3421, "Gruppe 1": 8543, ... },
        "leipzig_finetune": { "TILIA": 1234, "Gruppe 1": 3201, ... }
      }
    },

    "kl_divergence_validation": {
      "berlin_train_vs_val": 0.042,
      "berlin_train_vs_test": 0.038,
      "berlin_val_vs_test": 0.015,
      "leipzig_finetune_vs_test": 0.067,
      "all_passed": true
    }
  }
}
```

---

### 3.9 03a Integration

**Änderungen in 03a_setup_fixation.ipynb:**

```python
# Section 2: Apply Setup Decisions (UPDATED)

for city, split in city_splits:
    # ... (existing setup application)

    # Apply genus selection if available
    if genus_config is not None:
        viable_genera = genus_config['validation_results']['viable_genera']
        genus_to_final = genus_config['grouping_analysis']['genus_to_final_mapping']
        genus_german_mapping = genus_config['grouping_analysis']['genus_german_mapping']

        # Fix missing German names FIRST (before filtering)
        df = data_loading.fix_missing_genus_german(df)

        # Filter to viable genera
        df = df[df['genus_latin'].isin(viable_genera)].copy()

        # Apply genus mapping (BOTH columns!)
        df['genus_latin'] = df['genus_latin'].map(genus_to_final)
        df['genus_german'] = df['genus_latin'].map(genus_german_mapping)

        # Sanity check
        if df['genus_german'].isna().any():
            raise ValueError(f"Missing German names after mapping in {split_name}")

        genus_filtering_applied = True
    else:
        # Still fix German names even without genus filtering
        df = data_loading.fix_missing_genus_german(df)
        genus_filtering_applied = False

    # Save
    output_path = OUTPUT_DIR / f"{split_name}.parquet"
    df.to_parquet(output_path, index=False)

    # Log
    print(f"  {split_name}: {len(df):,} samples")
    if genus_filtering_applied:
        print(f"     Genus mapping applied: {len(viable_genera)} → {len(genus_config['decision']['final_genera_list'])} classes")
        print(f"     German names mapped: ✅")
```

---

## 4. Implementation Steps

### Phase 1: Core JM Implementation (exp_10)

1. **Add JM Functions to exp_10**
   - Implementiere `bhattacharyya_distance()`
   - Implementiere `jeffries_matusita_distance()`
   - Implementiere `compute_jm_matrix()`
   - Test mit 2-3 Genera (Unit-Test-artig)

2. **Replace Distance Analysis Section**
   - ALTE Section 4 löschen (Zentroid-Distanz)
   - NEUE Section 4: "Separability Analysis (JM Distance)"
   - Compute JM Matrix für `berlin_train_viable`
   - Visualisiere JM Heatmap

3. **Implement Relative Threshold**
   - Berechne Percentile-basierte Threshold
   - Adaptive Fallback-Logik
   - Log JM-Statistik (Min, Q10, Q25, Median, Q75, Max)

4. **Update Clustering Section**
   - Convert JM → Clusterable Distance (`2 - JM`)
   - Hierarchical Clustering mit Ward
   - Cut bei relativem Threshold
   - Erstelle Genus-Gruppen

5. **Add German Name Mapping**
   - Call `fix_missing_genus_german()` FRÜH im Notebook
   - Erstelle `genus_german_mapping` für alle final classes
   - Validate: keine NaN-Werte

6. **Update JSON Export**
   - Extend `setup_decisions.json` (nicht separate Datei!)
   - Include: `genus_to_final_mapping` + `genus_german_mapping`
   - Include: JM statistics für Dokumentation

### Phase 2: Visualization (exp_10)

7. **Create JM Heatmap**
   - Seaborn heatmap mit RdYlGn colormap
   - Annotate Threshold
   - Save: `jm_separability_heatmap.png`

8. **Update Dendrogram**
   - Use JM-based linkage matrix
   - Annotate JM threshold (nicht normalized distance)
   - Save: `jm_dendrogram.png`

9. **Create Grouping Overview**
   - Horizontal bar chart: genus groups + sizes
   - Annotate mit lateinischen + deutschen Namen
   - Save: `genus_groups_overview.png`

### Phase 3: 03a Integration

10. **Update 03a Data Processing Loop**
    - Load `genus_german_mapping` from setup_decisions.json
    - Apply mapping to BOTH `genus_latin` AND `genus_german`
    - Call `fix_missing_genus_german()` before filtering
    - Validate no NaN in genus_german

11. **Update 03a Logging**
    - Show: "Genus mapping: 28 → 25 classes"
    - Show: "German names mapped: ✅"

### Phase 4: Testing & Validation

12. **Run Full Pipeline**

    ```bash
    exp_08 → exp_09 → exp_10 (NEW) → 03a (UPDATED) → exp_11
    ```

13. **Validate Outputs**
    - [ ] `setup_decisions.json` enthält `genus_selection.grouping_analysis.genus_german_mapping`
    - [ ] Alle finale Datensätze haben konsistente `genus_latin` + `genus_german`
    - [ ] Keine NaN-Werte in beiden Spalten
    - [ ] KL-Divergence Validation passed
    - [ ] 3 Visualisierungen erstellt

14. **Update Documentation**
    - Update CHANGELOG.md
    - Update 06_Methodische_Erweiterungen.md (JM-Methode als implementiert markieren)

---

## 5. Technical Considerations

### 5.1 Numerical Stability

**Problem:** Kovarianzen können singulär sein (det = 0) bei:

- Wenig Samples (<20)
- Highly correlated features
- Near-constant features

**Solutions:**

```python
# 1. Regularization
cov_pool += np.eye(cov_pool.shape[0]) * 1e-6

# 2. Pseudo-Inverse fallback
try:
    inv_cov = np.linalg.inv(cov_pool)
except np.linalg.LinAlgError:
    inv_cov = np.linalg.pinv(cov_pool)

# 3. Determinant fallback
if det_pool <= 0:
    det_term = 0  # Skip term if degenerate
```

### 5.2 Memory Optimization

**JM Matrix Size:** Mit 30 Genera → 30×30 = 900 Werte → ~7 KB (kein Problem)

**Aber:** Feature Matrix (50k samples × 50 features × float32) = ~10 MB

- Use `df[feature_cols].values` für Numpy-Performance
- Free memory nach jeder JM-Berechnung

### 5.3 Runtime Estimation

- JM Matrix (30×30): ~30 Sekunden (435 pairwise comparisons)
- Hierarchical Clustering: <1 Sekunde
- Visualizations: ~5 Sekunden
- **Total: ~40 Sekunden zusätzlich**

---

## 6. Testing Strategy

### Unit Tests (Optional)

```python
def test_jm_identical_distributions():
    """JM(X, X) sollte ~0 sein."""
    X = np.random.randn(100, 10)
    jm = jeffries_matusita_distance(X, X)
    assert jm < 0.01

def test_jm_orthogonal_distributions():
    """JM zwischen orthogonalen Verteilungen sollte ~2 sein."""
    X_i = np.random.randn(100, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    X_j = np.random.randn(100, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    jm = jeffries_matusita_distance(X_i, X_j)
    assert jm > 1.8  # Sehr gut trennbar

def test_jm_matrix_symmetry():
    """JM-Matrix sollte symmetrisch sein."""
    df = pd.DataFrame({
        'genus': ['A']*50 + ['B']*50 + ['C']*50,
        'f1': np.random.randn(150),
        'f2': np.random.randn(150)
    })
    jm_matrix = compute_jm_matrix(df, ['f1', 'f2'], 'genus')
    assert np.allclose(jm_matrix, jm_matrix.T)
```

### Integration Tests

1. **Test mit 3 Genera:**
   - 2 ähnlich (ACER, PLATANUS) → niedrige JM → sollten gruppiert werden
   - 1 distinct (TILIA) → hohe JM → sollte singleton bleiben

2. **Test German Name Mapping:**
   - Nach Gruppierung: "Gruppe 1" sollte deutschen Namen haben
   - Singletons sollten original deutschen Namen behalten

3. **Test 03a Integration:**
   - Beide Spalten gemappt
   - Keine NaN-Werte
   - Sample Counts konsistent

---

## 7. Success Metrics

### Quantitative

- [ ] JM-Matrix vollständig berechnet (kein NaN)
- [ ] Threshold liegt zwischen 0.5 und 1.5 (realistisch)
- [ ] 2-5 Gruppen gebildet (nicht zu viele, nicht zu wenige)
- [ ] Alle finale Klassen haben ≥500 samples
- [ ] KL-Divergence < 0.15 für alle Split-Paare
- [ ] genus_german 100% gefüllt (kein NaN)

### Qualitative

- [ ] Gruppen biologisch plausibel (z.B. Rosaceae-Familie zusammen?)
- [ ] Deutsche Namen lesbar und informativ
- [ ] Visualisierungen klar interpretierbar
- [ ] JSON-Struktur vollständig dokumentiert

---

## 8. Rollback Strategy

**Falls JM-Methode Probleme macht:**

1. **Fallback auf Original-Methode** (Zentroid-Distanz)
   - Keep old code commented out
   - Can revert with git

2. **Alternative: Manual Grouping**
   - Falls JM-Matrix numerische Probleme hat
   - Definiere Gruppen manuell basierend auf biologischer Expertise

3. **Option: Skip Grouping**
   - Falls alle Genera gut trennbar (JM > 1.5)
   - Keine Gruppen bilden → 28 einzelne Klassen

---

## 9. Documentation Requirements

### In-Code Documentation

- [ ] Docstrings für alle neuen Funktionen (Google-Style)
- [ ] Kommentare in exp_10 für JM-Threshold-Logik
- [ ] Comments in 03a für beide Name-Spalten Mapping

### External Documentation

- [ ] Update CHANGELOG.md mit "JM-Distance Based Genus Grouping"
- [ ] Update 06_Methodische_Erweiterungen.md (JM-Methode als umgesetzt)
- [ ] Ergänze Notebook-Header mit JM-Methodik

### Metadata

- [ ] setup_decisions.json enthält vollständige Metadaten
- [ ] Execution Logs dokumentieren JM-Statistik
- [ ] Figures haben aussagekräftige Titel + Annotationen

---

## 10. Timeline & Milestones

| Milestone           | Tasks                          | Estimated Time | Blocker |
| ------------------- | ------------------------------ | -------------- | ------- |
| M1: JM Core         | Functions + Matrix Computation | 1h             | -       |
| M2: Clustering      | Threshold + Grouping Logic     | 1h             | M1      |
| M3: Naming          | German Mapping + fix_missing   | 1h             | M2      |
| M4: Visualization   | 3 Figures                      | 1h             | M3      |
| M5: 03a Integration | Beide Spalten mappen           | 1h             | M4      |
| M6: Testing         | Full Pipeline Run + Validation | 1h             | M5      |
| **TOTAL**           |                                | **6h**         |         |

**Puffer:** +2h für Debugging/Edge Cases = **8h Total**

---

## 11. Acceptance Criteria

### Must Have (Blocker für Merge)

- [x] JM-Distance korrekt implementiert (Tests passed)
- [x] Relativer Threshold funktioniert
- [x] genus_german_mapping vollständig
- [x] 03a wendet beide Mappings an
- [x] Keine NaN-Werte in finalen Datensätzen
- [x] KL-Divergence Validation passed
- [x] 3 Visualisierungen erstellt

### Should Have (Nice-to-Have)

- [ ] Unit Tests für JM-Funktionen
- [ ] Threshold-Sensitivity-Analyse (mehrere Percentiles testen)
- [ ] Biologische Validierung der Gruppen (Taxonomie-Check)

### Could Have (Future Work)

- [ ] Alternative Distance Metrics (Mahalanobis, Wasserstein)
- [ ] Automatische Threshold-Optimierung via Cross-Validation
- [ ] Interactive Dendrogram für Exploration

---

## 12. References

**Jeffries-Matusita Distance:**

- Swain & Davis (1978): "Remote Sensing: The Quantitative Approach"
- Richards & Jia (2006): "Remote Sensing Digital Image Analysis", Chapter 10
- Bruzzone & Prieto (2000): IEEE Trans. Geosci. Remote Sens., 38(6)

**Tree Species Classification:**

- Fassnacht et al. (2016): "Review of studies on tree species classification from remotely sensed data"
- Immitzer et al. (2019): "Tree Species Classification with Random Forest Using Very High Spatial Resolution Data"

---

_Erstellt: 2026-02-09_  
_Review Status: Ready for Implementation_  
_Owner: To Be Assigned_
