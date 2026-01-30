# Projektübersicht: Urban Tree Transfer

## Forschungskontext

Die automatisierte Erfassung urbaner Baumbestände gewinnt im Kontext von Klimaanpassung und Stadtplanung zunehmend an Bedeutung. Während Baumkataster in vielen deutschen Städten existieren, sind diese oft unvollständig, veraltet oder methodisch heterogen. Satellitengestützte Fernerkundung bietet das Potenzial, Baumgattungen flächendeckend und kosteneffizient zu klassifizieren.

Eine zentrale Herausforderung besteht jedoch in der **Übertragbarkeit** solcher Modelle: Ein Klassifikator, der für eine Stadt trainiert wurde, zeigt typischerweise erhebliche Leistungseinbußen bei der Anwendung auf andere Städte. Dies limitiert die praktische Anwendbarkeit erheblich.

## Forschungsfrage

**Kernfrage:** Wie gut lassen sich Machine-Learning-Modelle zur Baumgattungs-Klassifikation von einer Stadt auf eine andere übertragen, und wie viel lokale Trainingsdaten sind erforderlich, um die Transferverluste zu kompensieren?

**Teilfragen:**

1. Welche Modellkonfiguration erreicht die beste Performance bei Single-City-Training?
2. Wie stark degradiert die Performance bei Zero-Shot-Transfer auf eine neue Stadt?
3. Wie verhält sich die Performance-Recovery in Abhängigkeit von der Menge lokaler Fine-Tuning-Daten?

## Studiendesign

### Städteauswahl

| Stadt       | Rolle         | Begründung                                                                                           |
| ----------- | ------------- | ---------------------------------------------------------------------------------------------------- |
| **Berlin**  | Training      | Größtes deutsches Baumkataster (~800k Bäume), hohe Datenqualität, diverse Baumgattungen              |
| **Leipzig** | Transfer-Ziel | Ähnliche klimatische Bedingungen (kontinental), vergleichbare Stadtstruktur, unabhängiges Testgebiet |

Die Wahl von Berlin und Leipzig ermöglicht eine klare methodische Trennung:

- **Keine klimatischen Confounds**: Beide Städte liegen in der kontinentalen Klimazone
- **Vergleichbare Phänologie**: Ähnliche Vegetationsperioden und saisonale Muster
- **Räumliche Unabhängigkeit**: ~190 km Entfernung, keine räumliche Autokorrelation

### Experimentstruktur

Die Experimente (Phase 3 des Projekts) folgen einer sequentiellen Struktur, wobei jedes Experiment auf den Ergebnissen des vorherigen aufbaut:

```
Experiment 1: Berlin-Optimierung
    │
    │   Ziel: Optimales Modell für Single-City-Klassifikation
    │   Output: Best Model + Hyperparameter + Feature-Importance
    │
    ▼
Experiment 2: Transfer-Evaluation
    │
    │   Ziel: Quantifizierung der Transfer-Performance (Berlin → Leipzig)
    │   Output: Transfer-Metriken + Genus-spezifische Analyse
    │
    ▼
Experiment 3: Fine-Tuning-Analyse
    │
    │   Ziel: Dateneffizienz bei lokalem Fine-Tuning
    │   Output: Fine-Tuning-Kurve + Empfehlungen
    │
    ▼
Ergebnisse & Interpretation
```

## Datengrundlage

### Primärdaten

| Datentyp     | Quelle                              | Auflösung  | Zeitraum           |
| ------------ | ----------------------------------- | ---------- | ------------------ |
| Baumkataster | Open Data Portale (Berlin, Leipzig) | Punktdaten | Aktuell            |
| Sentinel-2   | Google Earth Engine                 | 10m / 20m  | Vegetationsperiode |
| Höhenmodelle | Landesvermessung (DOM/DGM)          | 1m         | Stadtspezifisch    |

### Feature-Raum

- **Spektrale Features**: 10 Sentinel-2 Bänder
- **Vegetationsindizes**: NDVI, EVI, Red-Edge-Indizes, Wasserindizes
- **Temporale Dimension**: Monatliche Komposite
- **Strukturelle Features**: Canopy Height Model

## Erwartete Ergebnisse

1. **Optimierte Single-City-Baseline**: Performance-Obergrenze für Berlin
2. **Transfer-Charakterisierung**: Quantifizierung des Performance-Drops und Identifikation transfer-robuster/sensitiver Gattungen
3. **Praktische Empfehlungen**: Mindestdatenmenge für effektives Fine-Tuning

## Projektrahmen

- **Bearbeitungszeitraum**: Januar-März 2026
- **Bearbeitung**: Einzelperson
- **Ressourcen**: Google Colab (GPU), Google Drive (Speicher)

## Abgrenzung

**Im Scope:**

- Baumgattungs-Klassifikation (Genus-Level)
- Transfer zwischen zwei deutschen Städten
- ML-Methoden auf tabularen Features (Random Forest, XGBoost)
- DL-Methoden auf tabularen Features (1D-CNN, TabNet)

**Außerhalb des Scope:**

- Artbestimmung (Species-Level)
- 2D-CNNs direkt auf Satellitenbildern
- Mehr als zwei Städte
- Echtzeit-Anwendung
