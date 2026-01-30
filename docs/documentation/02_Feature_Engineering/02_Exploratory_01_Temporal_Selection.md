# Exploratory 01: Temporal Feature Selection

**Notebook:** `notebooks/exploratory/exp_01_temporal_analysis.ipynb`
**Zweck:** Monats-Selektion via Jeffries-Matusita Distance
**Status:** ✅ Implementiert

---

## Ziel

Identifikation der zeitlich diskriminativsten Monate für Genus-Klassifikation. Reduktion von 12 auf 6-10 Monate durch Analyse der monatlichen Separabilität mittels JM-Distance.

---

## Methodik

### JM Distance (Jeffries-Matusita)

**Formel (Standard, Range 0-2):**

```
JM = 2 × (1 - exp(-B))

wobei B = (1/8) × (μ₁ - μ₂)² / ((σ₁ + σ₂)/2)
        + (1/2) × ln(((σ₁ + σ₂)/2) / √(σ₁ × σ₂))
```

**Interpretation:**

- JM = 0: Vollständige Überlappung
- JM ≈ 1: Akzeptable Diskriminierung
- JM = 2: Perfekte Separabilität

**Berechnung:**

1. Für jedes Feature (23 S2-Features) pro Monat (12 Monate)
2. Für alle Genus-Paare (z.B. ACER vs TILIA, ACER vs QUERCUS, ...)
3. Mittelung über alle 23 Features pro Monat → Monatlicher JM-Wert
4. Mittelung über beide Städte (Berlin + Leipzig)

**Numerische Stabilität:**

- Epsilon-Handling: `σ = max(σ, 1e-6)` verhindert Division durch Null
- Clipping: JM ∈ [0, 2]
- NaN-Filterung vor Berechnung

---

## Monats-Selektion

### Kriterien

**Top-N Approach mit saisonaler Balance:**

1. **Ranking:** Monate nach mittlerem JM sortieren (höher = besser)
2. **Top-N:** 8 beste Monate auswählen
3. **Seasonal Balance:** Mindestabdeckung sicherstellen:
   - Frühling (März-Mai): ≥1 Monat
   - Sommer (Juni-Aug): ≥2 Monate
   - Herbst (Sep-Nov): ≥1 Monat
   - Winter (Dez-Feb): Optional (typisch niedrige JM-Werte)

**Rationale:**

- **Phenologische Abdeckung:** Laub-Phasen (Austrieb, Vollbelaubung, Laubfärbung) differenzieren Genera
- **Datenreduktion:** 6-10 statt 12 Monate reduziert Feature-Dimensionen (~33% weniger)
- **Robustheit:** Vermeidung redundanter Winter-Monate (niedrige Aktivität)

### Performance-Optimierung

**Sampling:** 10.000 Bäume pro Genus (statt komplettem Datensatz)

- **Begründung:** JM-Schätzung stabil ab ~5.000 Samples, 10k bietet Puffer
- **Trade-off:** ~80% schnellere Berechnung bei <5% JM-Varianz

---

## Visualisierungen

### Plot 1: JM Distance Line Chart

**X-Achse:** Monate (1-12)
**Y-Achse:** Mean JM Distance (gemittelt über Genus-Paare)
**Linien:** Berlin (blau), Leipzig (orange)
**Error Bars:** Standardabweichung über Genus-Paare

**Interpretation:** Peaks zeigen Monate mit hoher Separabilität (typisch: Mai-September)

### Plot 2: JM Distance Heatmap (pro Stadt)

**Zeilen:** Top 15 Genus-Paare (häufigste Genera kombiniert)
**Spalten:** Monate (1-12)
**Farbe:** JM Distance (Viridis Colormap)
**Annotationen:** JM-Werte in Zellen

**Interpretation:** Hellere Bereiche zeigen genus-spezifische phenologische Unterschiede

---

## Output

### JSON: `temporal_selection.json`

**Schema:**

```json
{
  "analysis_date": "2026-01-30T...",
  "total_trees_analyzed": 1045234,
  "viable_genera": ["ACER", "TILIA", "QUERCUS", ...],
  "genus_pairs_analyzed": 45,
  "monthly_jm_statistics": {
    "1": {"mean": 0.65, "std": 0.12, "min": 0.42, "max": 0.89},
    ...
  },
  "selection_method": "top_n_with_seasonal_balance",
  "selection_threshold": 0.80,
  "selected_months": [3, 4, 5, 6, 7, 8, 9, 10],
  "rejected_months": [1, 2, 11, 12],
  "rationale": "Selected 8 months with highest mean JM..."
}
```

**Verwendung:** Phase 2b (Quality Control) lädt `selected_months` und filtert temporal Features

### Plots (3 Dateien)

```
outputs/phase_2/figures/exp_01_temporal/
├── jm_distance_by_month.png        # Line chart (Berlin vs Leipzig)
├── jm_heatmap_berlin.png           # Heatmap Berlin
└── jm_heatmap_leipzig.png          # Heatmap Leipzig
```

**DPI:** 300 (Publication-ready)

---

## Validierung

**In-Notebook Checks:**

- ✅ JM-Werte in Range [0, 2]
- ✅ Keine NaN in Ergebnissen
- ✅ Mindestens 500 Samples pro Genus
- ✅ 6-10 Monate selektiert (plausible Range)
- ✅ Cross-city Schema-Konsistenz

**Post-Execution:**

- JSON-Schema vollständig
- 3 PNG-Plots gespeichert
- Execution Log generiert

---

## Nächste Schritte

**Manual Sync (nach Colab-Ausführung):**

1. Download `temporal_selection.json` von Drive
2. Commit zu Git: `outputs/phase_2/metadata/temporal_selection.json`
3. Push zu GitHub

**Verwendung in Phase 2b:**

Die JSON-Datei wird geladen und die Liste `selected_months` extrahiert. Beim Laden der Feature-Daten werden dann nur noch Spalten mit Monatssuffixen aus den selektierten Monaten behalten. Nicht-temporale Features (Metadaten, CHM) bleiben vollständig erhalten.

---

**Dokumentations-Status:** ✅ Exploratory 01 (Temporal Selection) dokumentiert
