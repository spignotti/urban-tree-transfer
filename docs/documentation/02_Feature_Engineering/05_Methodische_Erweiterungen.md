# Methodische Erweiterungen: Feature Engineering

Dieses Dokument beschreibt methodische Erweiterungen, die während der Phase 2 Feature Engineering diskutiert, aber aus Zeitgründen oder Scope-Beschränkungen nicht implementiert wurden.

---

## 1. CHM × Pflanzjahr: Wachstumsrate als Feature

### Beschreibung

Statt die absolute Baumhöhe (CHM) oder deren genus-normalisierte Varianten zu verwenden, könnte ein biologisch fundierteres Feature berechnet werden: die **relative Wachstumsrate** basierend auf Höhe und Baumalter.

### Mögliche Features

| Feature                     | Berechnung                                | Was es kodiert                                       |
| --------------------------- | ----------------------------------------- | ---------------------------------------------------- |
| **growth_rate**             | `CHM_1m / (current_year - plant_year)`    | Durchschnittliche Wachstumsrate in m/Jahr            |
| **CHM_residual**            | `CHM_1m - expected_height(genus, age)`    | Abweichung von erwarteter Höhe für Gattung und Alter |
| **height_age_ratio_zscore** | Z-Score von `growth_rate` innerhalb Genus | Relative Wuchsdynamik im Vergleich zu Artgenossen    |

### Biologische Begründung

Die absolute Baumhöhe hängt von vielen stadtspezifischen Faktoren ab:

- **Pflanzjahr/Alter**: Ältere Bäume sind höher (trivial)
- **Standort**: Park vs. Straße, Bodenverdichtung, Versiegelungsgrad
- **Pflege**: Schnittregime unterscheiden sich zwischen Städten
- **Klima/Boden**: Lokale Wachstumsbedingungen

Die **Wachstumsrate** hingegen ist stärker gattungsspezifisch:

- Schnellwüchsige Gattungen (Populus, Salix): 0.5–1.0 m/Jahr
- Mittel (Tilia, Acer): 0.3–0.5 m/Jahr
- Langsam (Quercus, Fagus): 0.2–0.4 m/Jahr

Ein Wachstumsrate-Feature würde den Alters-Confound entfernen und ein biologisch sinnvolleres Signal liefern, das potenziell besser zwischen Städten transferiert.

### Warum nicht implementiert?

1. **Hohe NaN-Rate bei `plant_year`**: Nicht alle Bäume im Kataster haben ein Pflanzjahr. Fehlende Werte würden das Feature für einen signifikanten Anteil der Daten unbrauchbar machen.
2. **Nicht-lineare Wachstumskurven**: Bäume wachsen nicht linear. Junge Bäume wachsen schneller, alte langsamer. Eine einfache Division `Höhe / Alter` ist nur eine grobe Approximation. Gattungsspezifische Wachstumsmodelle (z.B. Chapman-Richards-Kurve) wären nötig, was erheblichen Zusatzaufwand bedeutet.
3. **Datenqualität**: `plant_year` stammt aus Katasterdaten und kann Fehler enthalten (Nachpflanzungen, falsche Einträge). CHM ist per LiDAR/Stereo-Photogrammetrie gemessen und deutlich zuverlässiger.
4. **Scope Phase 2**: Feature Engineering war auf vorhandene Datenquellen (Sentinel-2, CHM, Kataster-Metadaten) fokussiert, nicht auf die Ableitung komplexer biologischer Modelle.

### Potenzial für Folgearbeit

- **Analyse der `plant_year`-Verfügbarkeit** pro Stadt und Genus als erster Schritt
- Einfache Version: `CHM_1m / max(current_year - plant_year, 1)` für Bäume mit bekanntem Pflanzjahr
- Fortgeschritten: Genus-spezifische Wachstumskurven aus Literatur oder aus den eigenen Daten ableiten
- Höhe-Alter-Residuen könnten besonders transferierbar sein, weil sie stadtunabhängige biologische Variation kodieren

---

---

## 2. Temporale Selektion: Phänologische Phasen statt Jahreszeiten

### Beschreibung

Die aktuelle Monatsauswahl in **exp_01_temporal_analysis.ipynb** gruppiert Monate nach meteorologischen Jahreszeiten (Frühling, Sommer, Herbst, Winter). Für die Baumgattungs-Klassifikation wäre jedoch eine Gruppierung nach **phänologischen Phasen** biologisch sinnvoller und wissenschaftlich präziser.

### Aktuelle Implementierung (Jahreszeiten)

```python
# Beispiel aus exp_01:
seasons = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall": [9, 10, 11],
    "Winter": [12, 1, 2]
}
```

### Vorgeschlagene Implementierung (Phänologische Phasen)

```python
# Für mitteleuropäische Laubbäume (Berlin/Leipzig):
phenological_phases = {
    "Leaf-Out": [3, 4],          # Blattaustrieb, höchste Variabilität zwischen Arten
    "Full-Canopy": [5, 6, 7, 8], # Vollbelaubung, maximale Biomasse
    "Senescence": [9, 10],       # Laubfärbung/Abwurf, artspezifische Timing-Unterschiede
    "Dormancy": [11, 12, 1, 2]   # Keine Blätter (Laubbäume), konstantes Signal (Nadelbäume)
}
```

### Biologische Begründung

**Leaf-Out (März-April):**

- Höchste inter-genus Variabilität im Timing (z.B. BETULA früh, QUERCUS spät)
- Wichtig für Genus-Diskriminierung durch unterschiedliche Phänologie
- Literatur: Hemmerling et al. (2021) - "Early spring phenology maximiert Separabilität"

**Full-Canopy (Mai-August):**

- Maximale spektrale Unterschiede durch Blattchemie und -struktur
- Red-Edge-Indizes (NDVIre, CIre) am informativsten
- Literatur: Immitzer et al. (2019) - "Juni-August optimal für Red-Edge features"

**Senescence (September-Oktober):**

- Laubfärbung unterscheidet Genera (Anthocyan-Akkumulation artspezifisch)
- SWIR-Bänder (B11, B12) zeigen Wasserverlust
- Literatur: Fassnacht et al. (2016) - "Herbstfärbung ist genus-spezifisch"

**Dormancy (November-Februar):**

- Laubbäume: Kaum Signal (nur Stamm/Zweige)
- Nadelbäume: Konstantes Signal (immergrün) → klare Trennung Laub/Nadel möglich
- Typischerweise niedrige JM-Distance Werte

### Literatur-Referenzen

- **Hemmerling et al. (2021):** "Dense S2 time series" betonen Wichtigkeit phänologischer Schlüsselphasen für Baumarten-Klassifikation
- **Immitzer et al. (2019):** "Optimal Sentinel-2 features" identifizieren Juni-August als beste Monate für Red-Edge features
- **Fassnacht et al. (2016):** "Tree species classification review" nennt phänologische Gradienten als Herausforderung und Chance
- **Grabska et al. (2019):** "S2 time series for forest stands" zeigen, dass saisonale Komposite (nicht Monate) robuster sind

### Vorgeschlagene Änderungen in exp_01

**1. Visualisierung: JM-Distance nach Phänologischer Phase**

```python
# Statt Jah reszeiten-Boxplot:
# → Phänologische Phasen-Boxplot

phase_jm = []
for phase_name, months in phenological_phases.items():
    phase_jm.append({
        'phase': phase_name,
        'mean_jm': jm_distances[months].mean(),
        'std_jm': jm_distances[months].std()
    })

# Barplot: Phänologische Phase (X) vs. Mean JM-Distance (Y)
# Zeigt: Leaf-Out und Senescence haben höchste Discriminative Power
```

**2. Cross-City Consistency pro Phänologischer Phase**

```python
# Test: Sind die Phasen in beiden Städten konsistent?
# Spearman ρ für Leaf-Out, Full-Canopy, Senescence separat

for phase_name, months in phenological_phases.items():
    berlin_phase_jm = jm_berlin[months].mean()
    leipzig_phase_jm = jm_leipzig[months].mean()
    # Compare ranks
```

**3. JSON-Output: Phänologische Annotierung**

```json
// In temporal_selection.json ergänzen:
{
  "selected_months": [3, 4, 5, 6, 7, 8, 9, 10],
  "phenological_coverage": {
    "leaf_out": [3, 4],
    "full_canopy": [5, 6, 7, 8],
    "senescence": [9, 10],
    "dormancy": [] // bewusst ausgeschlossen
  },
  "phenological_rationale": "Selected months cover all active growth phases (Leaf-Out, Full-Canopy, Senescence) while excluding dormancy period with low discriminative power. Consistent with Hemmerling et al. (2021) emphasis on phenological key phases."
}
```

### Warum nicht implementiert?

1. **Scope Phase 2:** Fokus lag auf JM-Distance-basierter Monatsauswahl, nicht auf biologischer Interpretation
2. **Jahreszeiten waren ausreichend:** Für erste Feature-Selektion war jahreszeiten-basierte Visualisierung pragmatisch
3. **Literatur-Review fehlte:** Tiefere Einarbeitung in phänologische Studien (Hemmerling, Immitzer) erfolgte erst nach Phase 2

### Potenzial für Folgearbeit

- **Einfache Umsetzung:** Code-Änderung in exp_01 ist minimal (nur Gruppierungs-Dictionary ersetzen)
- **Höherer wissenschaftlicher Wert:** Phänologische Phasen sind biologisch fundiert, nicht willkürlich
- **Cross-City-Transfer:** Phänologische Phasen könnten zwischen Städten robuster sein als absolute Monate (z.B. Leaf-Out immer wichtig, auch wenn Timing leicht verschoben)
- **Paper-Argumentation:** Ermöglicht stärkere Diskussion der Ergebnisse mit Bezug zu phänologischer Ökologie

---

## 3. Deutsche Gattungsnamen in Visualisierungen

### Beschreibung

In mehreren Phase-2-Visualisierungen werden aktuell lateinische Gattungsnamen verwendet (z. B. `TILIA`, `ACER`). Für die Abschlussarbeit und konsistente Darstellung in Phase 3 sollen stattdessen **deutsche Gattungsnamen** (Spalte `genus_german`) angezeigt werden.

### Warum nicht implementiert?

1. **Scope Phase 2:** Fokus lag auf methodischer Validierung, nicht auf sprachlicher Konsistenz der Plots.
2. **Verteilung auf mehrere Notebooks:** Mapping und Plot-Labels müssten in mehreren Exploratory-Notebooks angepasst werden.

### Potenzial für Folgearbeit

- Einheitliches Label-Mapping pro Notebook (z. B. `label = genus_german`)
- Prüfen, ob alle Visualisierungen konsequent deutsche Labels verwenden

---

## 4. Nadel-/Laubbaum-Spalte im Datensatz

### Beschreibung

Für Phase-3-Analysen (z. B. Gruppenauswertung, F1 nach Baumgruppe) wird eine explizite Spalte benötigt, die Bäume als **Nadelbaum** oder **Laubbaum** klassifiziert (z. B. `is_conifer` oder `tree_group`).

### Warum nicht implementiert?

1. **Scope Phase 2:** Zusätzliche Metadaten-Spalten wurden nicht erweitert.
2. **Implementierung quer durch Pipeline:** Die Spalte sollte in den finalen Outputs konsistent verfügbar sein (Phase 2b/2c), inklusive Tests/Schema.

### Potenzial für Folgearbeit

- Spalte in Feature-Pipeline ergänzen (mapping über bestehende Genus-Listen)
- Schema/Validatoren aktualisieren

---

## 5. Performance-Optimierung der Notebooks

### Beschreibung

Die aktuellen Phase-2-Notebooks (Feature Extraction, Data Quality, Final Preparation) arbeiten mit **~1 Million Bäumen** und **großen Raster-Dateien** (CHM: ~2 GB/Stadt, Sentinel-2: ~500 MB/Monat). Die Pipeline funktioniert, zeigt jedoch bei wiederholter Ausführung Optimierungspotenzial in Bezug auf:

- **Arbeitsspeicher-Verbrauch:** Große Raster werden teilweise mehrfach in den Speicher geladen
- **I/O-Operationen:** Wiederholtes Lesen identischer Raster-Tiles/Regionen
- **Vectorized Operations:** Einige Schleifen könnten durch NumPy/Pandas-Vektorisierungen ersetzt werden
- **Chunking-Strategien:** Baumverarbeitung könnte in kleineren Batches erfolgen, um RAM-Peaks zu vermeiden

### Warum nicht implementiert?

1. **Funktionale Priorität:** Die Pipeline liefert korrekte Ergebnisse. Performance-Tuning ist eine Optimierung, kein Bugfix.
2. **Scope Phase 2:** Fokus lag auf methodischer Korrektheit (Qualitätsprüfungen, Outlier-Handling, Splits), nicht auf Performance.
3. **Ausführungsumgebung:** Google Colab bietet ausreichend RAM (12-25 GB), Pipeline läuft durch.

### Beobachtete Engpässe

- **Feature Extraction (02a):** Rasterio-Sample-Operationen über 1 Mio. Punkte
- **Proximity Berechnung (02c):** Paarweise Distanzberechnungen für Spatial Split
- **Memory-Peaks:** CHM + Sentinel-2 gleichzeitig im Speicher

### Potenzial für Folgearbeit

**Kurzfristige Optimierungen:**

- **Raster-Windowing:** Nur benötigte Regionen des CHM/Sentinel-2 laden (via `rasterio.windows`)
- **Batch-Processing:** Bäume in Chunks von 50k verarbeiten statt alle auf einmal
- **Caching:** Häufig verwendete Raster-Arrays im Memory halten (mit Garbage Collection)

**Mittelfristige Optimierungen:**

- **Dask-Integration:** Parallele Verarbeitung mit delayed evaluation
- **Spatial Indexing:** R-Tree für schnellere räumliche Abfragen
- **Parquet-Partitioning:** Stadt-spezifische Partitionen für schnelleres Filtern

**Benchmarking:**

- Profilierung mit `line_profiler` oder `memory_profiler`
- Laufzeit-Vergleich vor/nach Optimierungen
- RAM-Peak-Monitoring

### Priorität

**Mittel-Hoch:** Phase 2 läuft aktuell, aber für Phase 3 (Experimente mit vielen Trainingsläufen) wird Performance-Optimierung wichtiger. Eine Überarbeitung der Notebooks mit Fokus auf Memory-Effizienz und I/O-Reduktion ist für zukünftige Iterationen empfohlen.

---

## 6. Konsolidierung der Notebook-Konfigurationen

### Beschreibung

Die aktuellen Notebooks (Phase 1, Phase 2a/b/c, Exploratory) enthalten **redundante Konfigurationen** direkt im Code. Parameter wie Pfade, Schwellenwerte, Feature-Listen und Verarbeitungsoptionen werden in mehreren Notebooks wiederholt definiert, was zu Inkonsistenzen und Wartungsaufwand führt.

**Beispiele für duplizierte Konfigurationen:**

- Feature-Listen (Sentinel-2 Bänder, Vegetation Indices)
- Outlier-Schwellenwerte (z.B. `CHM_MAX_VALID`, `NDVI_MIN_VALID`)
- Pfad-Definitionen (Data-Directory, Output-Directory)
- Split-Parameter (Test-Size, Random-Seed, Spatial-Distance)
- Plot-Styles (Figsize, DPI, Font-Sizes)

### Aktueller Zustand

**Phase 1 (01_data_processing.ipynb):**

```python
# Hardcoded im Notebook
PROJECT_CRS = "EPSG:25833"
BOUNDARY_BUFFER = 500
NODATA_VALUE = -9999.0
```

**Phase 2a (02a_feature_extraction.ipynb):**

```python
# Wieder definiert
SPECTRAL_BANDS = ["B2", "B3", "B4", ...]
VEG_INDICES = ["NDVI", "EVI", "GNDVI", ...]
CHM_MAX = 50.0
```

**Exploratory Notebooks:**

```python
# Jeweils neu definiert
OUTLIER_THRESHOLDS = {"NDVI": (0.2, 0.95), "CHM_1m": (0, 50)}
```

### Problem

- **Inkonsistenzrisiko:** Änderungen müssen über mehrere Notebooks manuell synchronisiert werden
- **Wartungsaufwand:** Schwellenwerte/Parameter an mehreren Stellen anpassen
- **Fehleranfälligkeit:** Typos oder vergessene Updates führen zu falschen Ergebnissen
- **Reproduzierbarkeit:** Schwierig nachzuvollziehen, welche Konfiguration zu welchem Experiment gehört

### Warum nicht implementiert?

1. **Iterative Entwicklung:** Während der Exploration/Entwicklung ist schnelles Experimentieren wichtiger als perfekte Konfiguration
2. **Scope Phase 2:** Fokus auf Methodik-Implementierung, nicht auf Code-Infrastruktur
3. **Funktionale Anforderung:** Pipeline liefert Ergebnisse, Refactoring ist kein Blocker

### Potenzial für Folgearbeit

**Kurzfristige Verbesserungen:**

**1. Experiment-Configs erweitern:**

```yaml
# configs/experiments/phase2_feature_engineering.yaml
extraction:
  sentinel2:
    bands: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
    indices:
      [
        NDVI,
        EVI,
        GNDVI,
        NDre1,
        NDVIre,
        CIre,
        IRECI,
        RTVIcore,
        NDWI,
        MSI,
        NDII,
        kNDVI,
        VARI,
      ]
  chm:
    value_range: [0, 50]
    buffer_radius: 30

quality:
  outlier_thresholds:
    NDVI: [0.2, 0.95]
    EVI: [-0.5, 1.5]
    CHM_1m: [0, 50]
    CHM_30m_mean: [0, 40]

splits:
  test_size: 0.2
  spatial_distance_km: 5.0
  random_seed: 42

visualization:
  figsize: [12, 7]
  dpi: 300
  style: "seaborn-v0_8-whitegrid"
```

**2. Config-Loader in Notebooks:**

```python
# Statt Hardcoding
from urban_tree_transfer.config.loader import load_experiment_config

config = load_experiment_config("phase2_feature_engineering")
bands = config["extraction"]["sentinel2"]["bands"]
outliers = config["quality"]["outlier_thresholds"]
```

**3. Zentralisierte Plot-Konfiguration:**

```python
# src/urban_tree_transfer/config/visualization.py
PUBLICATION_STYLE = {
    "style": "seaborn-v0_8-whitegrid",
    "figsize": (12, 7),
    "dpi": 300,
    "font_size": 12,
    "title_size": 14
}
```

**Mittelfristige Verbesserungen:**

- **Config-Versionierung:** Experiment-Configs mit Git-Commit-Hash verknüpfen
- **Config-Validation:** Pydantic-Schemas für Type-Safety
- **Hydra-Integration:** Hierarchische Config-Komposition für Experiment-Varianten
- **Config-Tracking:** Mlflow/Sacred für Experiment-Parameter-Logging

### Vorteile einer Konsolidierung

✅ **Single Source of Truth:** Parameter nur einmal definieren  
✅ **Reproduzierbarkeit:** Klare Zuordnung Config → Experiment  
✅ **Wartbarkeit:** Eine Stelle für Schwellenwert-Anpassungen  
✅ **Testbarkeit:** Config-Loader kann Unit-getestet werden  
✅ **Dokumentation:** YAML ist self-documenting (mit Kommentaren)

### Priorität

**Mittel:** Aktuell funktioniert die Pipeline, aber vor Phase 3 (Experimente mit vielen Varianten) sollte die Konfiguration konsolidiert werden. Für systematische Hyperparameter-Experimente ist ein sauberes Config-System essenziell.

---

---

## 7. Genus-Auswahl-Validierung (Post-Phase 2c Filter-Kaskade)

### Beschreibung

**Problem:**
MIN_SAMPLES_PER_GENUS = 500 wird in Phase 1 (Data Processing) angewendet, resultiert in 30 viable genera. Phase 2c wendet jedoch weitere Filter an:

- **Proximity-Filter (5m):** Entfernt ~20% der Samples (mixed-genus proximity)
- **Outlier-Removal:** Entfernt weitere ~1-2% der Samples (high/medium outliers)

**Folge:** Genus-Sample-Counts können unter 500-Threshold fallen, ohne dass dies in der Data Processing Pipeline validiert wird.

### Aktuelle Lösung (Interims-Ansatz)

**Status:** Der Sample-Count-Filter wird **nachträglich in Phase 3 exp_10** angewendet.

**Begründung:** Phase 2 wurde bereits vollständig ausgeführt. Eine Neuberechnung der Feature Engineering Pipeline (~20h Rechenzeit) ist nicht praktikabel. Das exploratory notebook exp_10 validiert die Genus-Auswahl nachträglich und stellt sicher, dass alle Experimente mit konsistenten, validierten Genera arbeiten.

### Empfohlene Implementierung (Future Work - Pipeline-Überarbeitung)

**Wann:** Bei einer zukünftigen Überarbeitung der Feature Engineering Pipeline

**Ort:** Am Ende von `02c_final_preparation.ipynb`, nach Proximity-Filter und vor Export der finalen Splits

**Workflow:**

```
Phase 1: filter_viable_genera()
  ↓ 30 genera mit ≥500 Samples (beide Städte)
Phase 2a: Feature Extraction
  ↓ (keine Genus-Änderung)
Phase 2b: Data Quality (Outlier Detection)
  ↓ (Outlier markiert, nicht entfernt)
Phase 2c: Final Preparation
  ├─ Proximity Filter (5m) → ~20% Sample-Reduktion
  ├─ Outlier Removal → ~1-2% Sample-Reduktion
  └─ Re-apply MIN_SAMPLES_PER_GENUS Filter ← **FUTURE: Hier einfügen**
      ↓ Finale Genus-Liste (12-30 Genera)
      ↓ Export gefilterte Datensätze
Phase 3: Training/Evaluation
  ↓ Nutzt nur validierte Genera (keine nachträgliche Filterung nötig)
```

**Vorteil dieser Integration:**

- ✅ Saubere Separation of Concerns (Data Processing vs. Experiments)
- ✅ Alle Phase 3 Notebooks starten mit validierten Genera
- ✅ exp_10 kann sich auf Separabilität/Gruppierung fokussieren (nicht Sample-Counting)

### Code-Implementierung (für zukünftige Pipeline-Revision)

**Hinzufügen am Ende von Notebook 02c (vor Split-Export):**

```python
# Cell: Re-apply Genus Sample Threshold (Post-Filter Validation)
from urban_tree_transfer.config import MIN_SAMPLES_PER_GENUS

print(f"\n{'='*60}")
print("GENUS SAMPLE VALIDATION (Post-Proximity & Outlier Filter)")
print(f"{'='*60}\n")

# Count samples per genus after all filters
berlin_genus_counts = berlin_filtered["genus_latin"].value_counts()
leipzig_genus_counts = leipzig_filtered["genus_latin"].value_counts()

# Identify viable genera (≥MIN_SAMPLES in BOTH cities)
viable_berlin = set(berlin_genus_counts[berlin_genus_counts >= MIN_SAMPLES_PER_GENUS].index)
viable_leipzig = set(leipzig_genus_counts[leipzig_genus_counts >= MIN_SAMPLES_PER_GENUS].index)
final_viable_genera = sorted(list(viable_berlin & viable_leipzig))

# Identify excluded genera (had ≥500 in Phase 1, now <500)
phase1_genera = set(berlin_filtered["genus_latin"].unique())  # All genera that passed Phase 1
excluded_genera = sorted(list(phase1_genera - set(final_viable_genera)))

print(f"Phase 1 viable genera: {len(phase1_genera)}")
print(f"Phase 2c viable genera: {len(final_viable_genera)}")
print(f"Excluded due to sample reduction: {len(excluded_genera)}")

if excluded_genera:
    print(f"\nExcluded genera: {excluded_genera}")
    for genus in excluded_genera:
        berlin_count = berlin_genus_counts.get(genus, 0)
        leipzig_count = leipzig_genus_counts.get(genus, 0)
        print(f"  {genus}: Berlin={berlin_count}, Leipzig={leipzig_count}")

# Filter datasets to final viable genera
berlin_filtered = berlin_filtered[berlin_filtered["genus_latin"].isin(final_viable_genera)].copy()
leipzig_filtered = leipzig_filtered[leipzig_filtered["genus_latin"].isin(final_viable_genera)].copy()

print(f"\nFinal dataset sizes:")
print(f"  Berlin: {len(berlin_filtered):,} samples")
print(f"  Leipzig: {len(leipzig_filtered):,} samples")
print(f"  Final genera: {len(final_viable_genera)}")

# Export genus filter metadata
genus_filter_summary = {
    "phase1_genera_count": len(phase1_genera),
    "phase2c_genera_count": len(final_viable_genera),
    "excluded_genera": excluded_genera,
    "final_viable_genera": final_viable_genera,
    "min_samples_threshold": MIN_SAMPLES_PER_GENUS,
    "sample_counts": {
        "berlin": {g: int(berlin_genus_counts.get(g, 0)) for g in final_viable_genera},
        "leipzig": {g: int(leipzig_genus_counts.get(g, 0)) for g in final_viable_genera}
    }
}

genus_filter_path = METADATA_DIR / "genus_filter_phase2c.json"
with open(genus_filter_path, "w") as f:
    json.dump(genus_filter_summary, f, indent=2)
print(f"\nExported: {genus_filter_path}")
```

**Output Metadata:** `outputs/phase_2_splits/metadata/genus_filter_phase2c.json`

### Migration-Plan (bei Pipeline-Überarbeitung)

**Schritt 1:** Filter in Phase 2c integrieren (siehe Code oben)

**Schritt 2:** exp_10 vereinfachen

- Entfernen: Sample-Count-Validierung (wird in Phase 2c gemacht)
- Behalten: Separabilitäts-Analyse, Genus-Gruppierung
- Umbenennen: "Genus Selection Review" statt "Genus Selection Validation"

**Schritt 3:** Phase 2 Pipeline neu ausführen

- Outputs mit neuen, finalen Genus-Listen generieren
- Alle Phase 3 Notebooks nutzen dann automatisch gefilterte Datensätze

### Warum nicht bereits implementiert?

**Zeitpunkt der Entdeckung:** Das Problem wurde erst während der Phase 3 Planung erkannt, als die Diskrepanz zwischen 30 (Phase 1) und 16 (aktuell beobachtet) Genera auffiel.

**Scope Phase 2:** Der ursprüngliche PRD 002 fokussierte auf Feature-Extraktion und Datensatz-Splits, nicht auf Re-Validierung der Genus-Auswahl nach Filterung.

**Rechenzeit:** Phase 2 wurde bereits vollständig durchgeführt (~20h). Eine Neuberechnung nur für diesen Filter ist ökonomisch nicht vertretbar im aktuellen Projektzeitplan.

### Priorität

**Interims-Lösung (JETZT):** exp_10 führt Sample-Count-Filter durch ✅

**Langfristig (FUTURE WORK):** Bei Pipeline-Revision Filter in Phase 2c verschieben 📋

---

## 6. Genus-Auswahl-Validierung (Post-Setup-Decisions Filter-Kaskade)

### Problem

`MIN_SAMPLES_PER_GENUS = 500` wird in Phase 1 angewendet → 30 viable genera.
Phase 3 Setup-Decisions (CHM, Proximity, Outlier, Feature-Selektion) reduzieren Datensatz weiter.

**Folge:** Genus-Sample-Counts können unter 500-Threshold fallen.

### Workflow

```
Phase 1: filter_viable_genera()
↓ 30 genera mit ≥500 Samples
Phase 2a-2c: Feature Extraction, Quality Control, Splits
↓ (keine Genus-Änderung)
Phase 3 exp_08-09: Setup Decisions
↓ CHM/Proximity/Outlier/Feature-Strategie definiert
Phase 3 exp_10: Genus Selection Validation ← **NEU**
↓ Sample-Count + Separability-basierte Genus-Selektion
↓ Gruppierung schlecht separierbarer Genera
↓ Finale Genus-Liste (12-20 Klassen erwartet)
Phase 3 exp_11+: Algorithm Training
↓ Nutzt finale, gruppierte Genus-Liste
```

### Implementierung

**Ort:** `notebooks/exploratory/exp_10_genus_selection_validation.ipynb`

**Analysen:**

1. Setup-Decisions laden und anwenden (CHM, Proximity, Outlier, Features)
2. Sample-Counts auf finalem Datensatz validieren (≥500 Threshold)
3. Genus-Separabilität mittels **JM-Distance** (Jeffries-Matusita) berechnen - sample-level pairwise distance, nur Berlin Train
4. Hierarchisches Clustering (Ward-Linkage) auf JM-Matrix für Genus-Gruppierung
5. Adaptive Percentile-Threshold (z.B. 20th) zur Identifikation schlecht separierbarer Genera
6. Schlecht separierbare Genera gruppieren (z.B. JM < Threshold)

**Output:** `setup_decisions.json` (erweitert um `genus_selection` Section, genutzt von exp_11, 03a, 03b, 03c, 03d)

### Strategie: Exclude & Group

**Stufe 1:** Exclude Low-Sample Genera

- Entferne Genera mit <500 Samples in mindestens einer Stadt
- Begründung: Statistische Power (RF benötigt ~100-200 Samples, wir nutzen 500 für Transfer-Robustheit)

**Stufe 2:** Group Similar Genera

- **Methode:** Jeffries-Matusita Distance (JM-Distance) zur Messung der spektralen Separabilität zwischen Genera
- **JM-Formel:** JM = 2(1 - e^(-B)), wobei B = Bhattacharyya Distance
- **Threshold:** Percentile-basiert (z.B. 20th percentile der pairwise JM-Distribution)
- Gruppiere Genera-Paare mit JM < Threshold
- Z.B. Rosaceae-Gruppe: PRUNUS + MALUS + PYRUS (wenn JM(PRUNUS, MALUS) < Threshold)
- **Vorteil:** Objektive, probabilistische Separabilitätsmessung; mehr Samples pro Klasse
- **Nachteil:** Verlust von Granularität

**Begründung für JM-Distance:**

- Berücksichtigt vollständige Verteilungsform (nicht nur Centroids)
- Standard in Remote Sensing für Class Separability (Richards & Jia 2006)
- Werte: JM=0 (identisch), JM=2 (perfekt separierbar)

### Methodische Validität: Genus-Filtering nach Spatial Splits

**✅ Methodisch unproblematisch**, weil:

1. Block-Grenzen (1200m) sind geografisch fix (nicht genus-abhängig)
2. Train/Val/Test Block-Zuordnungen bleiben unverändert
3. Räumliche Autokorrelations-Prevention bleibt intakt
4. KL-Divergence nach Filtering bestätigt Stratifizierung (Threshold: <0.15)

**Validation:** exp_10 berechnet KL-Divergence zwischen Splits nach Genus-Filtering als Sanity Check.

### Future Work

**Vergleich: Gruppiert vs. Ungrouped**

Aktuelle Analyse nutzt gruppierte Genera für optimistische Trennbarkeit.
Zukünftige Studien sollten untersuchen:

- Performance-Unterschied: Gruppiert vs. Einzeln
- Trade-off: Granularität vs. Klassifikations-Accuracy
- Genus-spezifische Insights die durch Gruppierung verloren gehen

### Referenzen

- Breiman, L. (2001). Random Forests. _Machine Learning_, 45(1), 5-32.
- Belgiu, M., & Drăguţ, L. (2016). Random forest in remote sensing. _ISPRS Journal_, 114, 24-31.

---

## Zusammenfassung

| Erweiterung                                 | Status              | Priorität für Folgearbeit                      |
| ------------------------------------------- | ------------------- | ---------------------------------------------- |
| CHM × Pflanzjahr (Wachstumsrate)            | Nicht implementiert | Mittel (abhängig von plant_year Verfügbarkeit) |
| Temporale Selektion: Phänologische Phasen   | Nicht implementiert | Hoch (einfach umsetzbar, hoher wiss. Wert)     |
| Deutsche Gattungsnamen in Plots             | Nicht implementiert | Mittel (Konsistenz/Lesbarkeit)                 |
| Nadel-/Laubbaum-Spalte im Datensatz         | Nicht implementiert | Hoch (benötigt in Phase-3-Analysen)            |
| Performance-Optimierung der Notebooks       | Nicht implementiert | Mittel-Hoch (wichtig für Phase 3)              |
| Konsolidierung der Notebook-Konfigurationen | Nicht implementiert | Mittel (wichtig für systematische Experimente) |
| **Genus-Filter nach Phase 2c**              | **Future Work**     | **Mittel (bei Pipeline-Revision)**             |

---

## PRD 002d Status (Methodological Improvements)

PRD 002d enthält 7 Verbesserungen:

- **Umgesetzt (1–5):** Cross-City JM Consistency, Post-Split Spatial Independence, Genus-spezifische CHM-Normalisierung, Biological Context Analysis, Geometrische Klarheit im Proximity Filter
- **Offen (6–7):** Deutsche Gattungsnamen in Visualisierungen, Nadel-/Laubbaum-Spalte im Datensatz

---

_Letzte Aktualisierung: 2026-02-06_
