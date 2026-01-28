# Methodische Grundlagen

## Wissenschaftliche Prinzipien

### Reproduzierbarkeit

Alle Experimente sind vollständig reproduzierbar durch:

- **Fixierter Random Seed**: `42` für alle stochastischen Prozesse
- **Versionierte Konfigurationen**: YAML-Dateien für alle Experimente
- **Dokumentierte Entscheidungen**: Jede methodische Wahl wird begründet

### Spatial Awareness

Geografische Daten erfordern besondere methodische Sorgfalt:

**Spatial Leakage vermeiden:**

- Keine zufälligen Train/Test-Splits bei räumlichen Daten
- Spatial Block Cross-Validation
- Strikte räumliche Trennung zwischen Training und Evaluation

**Urban-spezifische Herausforderungen:**

- Sub-Pixel-Problem: Baumkronen (5-15m) bei 10m Pixelauflösung
- Mixed Pixels: Überlagerung von Baum, Straße, Gebäude
- Ungenauigkeiten Im Baukataster: Veraltete/nicht aktualisierte Datensätze, Messungenaugkeiten

### Occam's Razor

Bei vergleichbarer Performance wird die einfachere Lösung bevorzugt:

- Weniger Features > Mehr Features (bei ähnlicher Accuracy)
- Interpretierbare Modelle > Black-Box-Modelle
- Threshold: Δ < 2-3% rechtfertigt keine Komplexitätssteigerung

## Datengrundlage und Designentscheidungen

### Städteauswahl

Die Wahl von Berlin und Leipzig als Untersuchungsstädte basiert auf mehreren methodischen Überlegungen:

**Berlin (Source Domain):**

- **Datenverfügbarkeit**: Hervorragende Datenqualität und -vollständigkeit
- **Stichprobengröße**: ~800.000 Bäume im Kataster ermöglichen robustes Training
- **Diversität**: Hohe Gattungsvielfalt für repräsentative Modellentwicklung
- **Infrastruktur**: Open-Data-Portal mit gut dokumentierten WFS-Diensten

**Leipzig (Target Domain):**

- **Geografische Ähnlichkeit**: ~190 km Entfernung, kontinentale Klimazone
- **Stadtstruktur**: Vergleichbare urbane Morphologie (Gründerzeit, Plattenbauten, Grünanlagen)
- **Datenverfügbarkeit**: Gut gepflegtes Baumkataster mit konsistentem Schema
- **Unabhängigkeit**: Räumlich getrennt, keine Autokorrelation mit Berlin

Diese Kombination ermöglicht eine saubere Trennung zwischen Training und Transfer-Evaluation bei gleichzeitig kontrollierten klimatischen und strukturellen Rahmenbedingungen.

### Jahresauswahl und temporale Konsistenz

**Referenzjahr: 2021**

Alle Sentinel-2 Daten werden aus dem Jahr 2021 verwendet. Diese Entscheidung basiert auf folgenden Überlegungen:

**Höhenmodelle (CHM):**

- Berlin DOM/DGM: 2021 ✅
- Leipzig DOM/DGM: 2022 (1 Jahr Differenz)

**Entscheidung für Sentinel-2 2021:**

- Priorisierung der **Source Domain** (Berlin): Maximale temporale Konsistenz zwischen CHM und Spektraldaten für robustes Training
- Leipzig: 1-Jahres-Differenz akzeptabel, da:
  - Gefällte Bäume 2021-2022 werden durch Outlier-Filter erkannt (fehlende CHM-Signatur)
  - CHM primär zur Positionskorrektur genutzt, nicht als kritisches Feature
  - Sentinel-Outlier-Filter fängt Inkonsistenzen ab

**Klimatische Besonderheiten 2021:**

- **Vegetationsperiode**: Normal verlaufend ohne extreme Dürre (im Gegensatz zu 2018/2019)
- **Niederschlag**: Leicht überdurchschnittlich, gute Vegetationsentwicklung
- **Temperaturen**: Moderate Sommermonate, keine Hitzewellen
- **Interpretation**: 2021 repräsentiert typische mitteleuropäische Wachstumsbedingungen, Modelle sollten gut auf andere "normale" Jahre transferierbar sein

### Koordinatenreferenzsystem

**Projekt-CRS: EPSG:25833 (ETRS89 / UTM Zone 33N)**

Gründe für diese Wahl:

- **Native CRS beider Städte**: Berlin und Leipzig nutzen beide EPSG:25833 für offizielle Geodaten
- **Metrisches System**: UTM ermöglicht direkte Distanzmessungen in Metern (wichtig für Buffer, Spatial Splits)
- **Geringe Verzerrung**: Beide Städte liegen zentral in UTM Zone 33N, minimale projektionsbedingte Verzerrungen
- **Kompatibilität**: Sentinel-2 wird von GEE in EPSG:25833 exportiert
- **Vermeidung von Reprojektionsfehlern**: Minimiert Transformationen und damit potenzielle Genauigkeitsverluste

Alternative EPSG:4326 (WGS84) wurde verworfen, da geografische Koordinaten keine direkte Distanzberechnung ermöglichen.

## Experimentelles Design

### Phasen-Abhängigkeiten

```
Vorverarbeitung (Data Processing)
         │
         ▼
Feature Engineering
         │
         ├── Extraktion
         ├── Qualitätskontrolle
         ├── Feature-Selektion
         └── Spatial Splits
         │
         ▼
Phase 1: Berlin-Optimierung
         │
         │  Fixiert: Algorithmus, Hyperparameter, Features
         │
         ▼
Phase 2: Transfer-Evaluation
         │
         │  Fixiert: Berlin-Modell
         │
         ▼
Phase 3: Fine-Tuning-Analyse
```

Jede Phase baut auf den fixierten Ergebnissen der vorherigen auf. Rückwärts-Iterationen sind methodisch nicht vorgesehen.

### Ablationsprinzip

Pro Experiment wird **eine Variable** variiert, alle anderen bleiben konstant:

- Ermöglicht kausale Attribution von Performance-Unterschieden
- Vermeidet Konfundierung von Effekten
- Erleichtert Interpretation

### Metriken

**Primärmetrik:** Macro-F1

- Berücksichtigt alle Klassen gleichwertig
- Robust gegenüber Klassenimbalance
- Standard in Multi-Class-Klassifikation

**Sekundärmetriken:**

- Weighted-F1 (gewichtet nach Klassenhäufigkeit)
- Per-Genus-F1 (für detaillierte Analyse)
- Train-Val-Gap (Overfitting-Indikator)

## Transfer Learning Framework

### Begriffsdefinitionen

| Begriff                | Definition                                     |
| ---------------------- | ---------------------------------------------- |
| **Source Domain**      | Berlin (Trainingsstadt)                        |
| **Target Domain**      | Leipzig (Transferstadt)                        |
| **Zero-Shot Transfer** | Direkter Modell-Transfer ohne lokale Anpassung |
| **Fine-Tuning**        | Nachtraining mit lokalen Target-Daten          |

### Transfer-Hypothesen

**H1 (Baseline):** Ein auf Berlin trainiertes Modell zeigt signifikanten Performance-Verlust bei Anwendung auf Leipzig.

**H2 (Feature-Stabilität):** Bestimmte Features (z.B. Red-Edge-Indizes) sind robuster gegenüber Domain-Shift als andere.

**H3 (Genus-Variabilität):** Transfer-Robustheit variiert zwischen Baumgattungen; häufige/distinkte Gattungen transferieren besser.

**H4 (Fine-Tuning-Effizienz):** Bereits kleine Mengen lokaler Daten (~10-25%) ermöglichen signifikante Performance-Recovery.

## Datenqualität

### Filterkriterien für Bäume

Nicht alle Bäume im Kataster sind für die Klassifikation geeignet:

| Kriterium                      | Begründung                                          |
| ------------------------------ | --------------------------------------------------- |
| Pflanzjahr ≤ Referenzjahr      | Baum muss zum Aufnahmezeitpunkt existiert haben     |
| Genus bekannt                  | Supervised Learning erfordert Labels                |
| Ausreichende Samples pro Genus | Mindestens N Bäume pro Klasse für robustes Training |
| Räumliche Qualität             | Keine offensichtlichen Koordinatenfehler            |

### Gattungs-Isolations-Filter

Bei 10m Pixelauflösung können benachbarte Bäume unterschiedlicher Gattungen zu gemischten spektralen Signaturen führen. Der Gattungs-Isolations-Filter adressiert dieses Problem:

- **Problem**: Bäume mit <20m Abstand zu Bäumen anderer Gattungen haben potentiell kontaminierte Spektralsignaturen
- **Lösung**: Filterung von Bäumen, die weniger als 20m von einem Baum einer anderen Gattung entfernt sind
- **Trade-off**: Reduzierte Stichprobengröße, aber reinere spektrale Signaturen pro Gattung

## Computational Constraints

Die verfügbaren Rechenressourcen beeinflussen maßgeblich das experimentelle Design:

### Ausführungsumgebung

**Google Colab** dient als primäre Ausführungsumgebung:

- **GPU-Zugriff**: T4/A100 GPUs für Deep-Learning-Modelle
- **RAM**: 12-25 GB (je nach Runtime)
- **Laufzeitbeschränkung**: ~12h kontinuierliche Ausführung
- **Storage**: Google Drive als persistenter Datenspeicher

### Methodische Implikationen

Diese Constraints beeinflussen die Projektgestaltung:

| Constraint          | Implikation                                                                        |
| ------------------- | ---------------------------------------------------------------------------------- |
| Laufzeitlimit       | Modular gestaltete Pipelines (Data Processing → Feature Engineering → Experiments) |
| RAM-Limit           | Vektorisierte Operationen, inkrementelles Laden großer Datensätze                  |
| Instabiler Speicher | Checkpoints nach jeder Phase, keine In-Memory-Caching von Rohdaten                 |
| GPU-Verfügbarkeit   | DL-Modelle optional, ML-Modelle als robuste Baseline                               |

### Reproduzierbarkeit trotz Cloud-Instabilität

- Alle Zwischenergebnisse werden persistent gespeichert (Google Drive)
- Pipelines sind unterbrechbar und fortsetzbar (Checkpoints)
- Code ist lokal ausführbar (ohne Colab), aber langsamer

## Limitationen (a priori)

### Bekannte Einschränkungen

1. **Zeitliche Limitation**: Daten aus einem einzigen Jahr (single-year snapshot)
2. **Räumliche Limitation**: Nur zwei Städte, beide in Deutschland
3. **Taxonomische Limitation**: Genus-Level, nicht Species-Level
4. **Methodische Limitation**: ML und DL auf tabularen Features, keine 2D-CNNs auf Satellitenbildern direkt

### Umgang mit Unsicherheiten

- Unsicherheiten werden dokumentiert, nicht versteckt
- Konfidenzintervalle wo möglich
- Sensitivitätsanalysen für kritische Parameter

## Dokumentationsstandards

### Struktur pro Experiment

1. **Forschungsfrage**: Was wird getestet?
2. **Methodik**: Wie wird getestet? (kompakt, Tabellen bevorzugt)
3. **Ergebnisse**: Was kam heraus? (Zahlen, Visualisierungen)
4. **Entscheidung**: Was wurde gewählt und warum?
5. **Limitationen**: Was sind die Einschränkungen?

### Anti-Patterns

- Keine langen Einleitungen
- Keine Erklärung von ML-Grundlagen
- Keine Redundanz zwischen Dokumenten
- Keine Spekulation über nicht durchgeführte Experimente
