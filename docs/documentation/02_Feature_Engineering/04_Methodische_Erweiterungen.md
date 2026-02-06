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

## Zusammenfassung

| Erweiterung                               | Status              | Priorität für Folgearbeit                      |
| ----------------------------------------- | ------------------- | ---------------------------------------------- |
| CHM × Pflanzjahr (Wachstumsrate)          | Nicht implementiert | Mittel (abhängig von plant_year Verfügbarkeit) |
| Temporale Selektion: Phänologische Phasen | Nicht implementiert | Hoch (einfach umsetzbar, hoher wiss. Wert)     |
| Deutsche Gattungsnamen in Plots           | Nicht implementiert | Mittel (Konsistenz/Lesbarkeit)                 |
| Nadel-/Laubbaum-Spalte im Datensatz       | Nicht implementiert | Hoch (benötigt in Phase-3-Analysen)            |

---

## PRD 002d Status (Methodological Improvements)

PRD 002d enthält 7 Verbesserungen:

- **Umgesetzt (1–5):** Cross-City JM Consistency, Post-Split Spatial Independence, Genus-spezifische CHM-Normalisierung, Biological Context Analysis, Geometrische Klarheit im Proximity Filter
- **Offen (6–7):** Deutsche Gattungsnamen in Visualisierungen, Nadel-/Laubbaum-Spalte im Datensatz

---

_Letzte Aktualisierung: 2026-02-06_
