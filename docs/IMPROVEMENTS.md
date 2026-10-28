# Improvements

> Migriert aus Notion am 2026-10-28 (Seite `b746f8661f244215937e4b276c44b231`, zuletzt bearbeitet 2026-03-30). Text unverändert übernommen.

Umsetzbare Verbesserungen für eine spätere Publikationsrunde, gegliedert nach Phase und Pipeline-Schritt.

> ✅ **Alle \[Rework\]-Items wurden in Phase C (C1–C13) umgesetzt und in die Methodik-Dokumentation überführt.** Dieses Dokument enthält nur noch **\[Paper\]**-Items.

---

## Phase 1: Data Processing

### Datenakquise & Vorverarbeitung

- **\[Paper\] SCL-Maskierung erweitern:** Aktuell nur Klasse 4+5 (Vegetation, Bare Soil). Evaluation von Klasse 6 (Water) und 11 (Snow/Ice) für Wintermonate, wo Klasse 4+5 allein zu wenig valide Pixel liefern könnte.
- **\[Paper\] CHM-Artefakt-Quantifizierung:** Werte \< −2m werden auf 0 gesetzt. Häufigkeit und Pixel-Anteil dieser Artefakte quantifizieren.

### Feature-Definition

- **\[Paper\] Vegetationsindex-Begründung:** Literaturbasierte Referenzierung der 13 Indizes steht aus (Immitzer et al. 2019, Grabska et al. 2019, Hemmerling et al. 2021).

### Genus-Filterung

- **\[Paper\] MIN\_SAMPLES\_PER\_GENUS Sensitivität:** Aktuell 500 — Sensitivitätsanalyse mit 200/500/1000 wäre aufschlussreich.
- **\[Paper\] Genus-Auswahl-Validierung nach Filter-Kaskade:** Phase-2c-Filter (Proximity, Outlier) reduzieren Sample-Counts unter den 500er-Threshold. Post-Filter-Validierung mit JM-Distance-basiertem Clustering für Genus-Gruppierung (schlecht separierbare Genera zusammenfassen). Interims-Lösung: exp\_10 in Phase 3.

## Phase 2: Feature Engineering

### Temporal Selection (Exp 01)

- **\[Paper\] JM-Distance Normalverteilungsannahme validieren:** Standard-JM setzt univariate Normalverteilung voraus. Shapiro-Wilk-Test pro Feature/Monat zur Absicherung. Multivariate Variante (JM mit Kovarianzmatrizen) als Alternative.
- **\[Paper\] Selektions-Threshold Sensitivität:** Threshold 0.80 ist empirisch gewählt — keine Literaturgrundlage. Sensitivitätsanalyse mit verschiedenen Thresholds (0.6 / 0.8 / 1.0) und deren Auswirkung auf die finale Klassifikation.
- **\[Paper\] Phänologische Phasen statt Jahreszeiten:** Monatsauswahl nach biologisch fundierten Phasen (Leaf-Out, Full-Canopy, Senescence, Dormancy) statt meteorologischen Jahreszeiten gruppieren. Einfach umsetzbar, wissenschaftlich überzeugender und potenziell robuster im Cross-City-Transfer.

### CHM Assessment (Exp 02)

- **\[Paper\] Bootstrap-CI Iterationen erhöhen:** Aktuell 1000 Iterationen für Cohen's d CIs — Standard ist 5000–10000 für stabile 95%-CIs.
- **\[Paper\] Rare Genera Fallback Sensitivität:** Threshold \<10 Samples für city-level Fallback ist willkürlich. Sensitivitätsanalyse mit 5/10/20.

### Correlation Analysis (Exp 03)

- **\[Paper\] Monatliche Korrelationsanalyse statt Temporal Stacking:** Aktuell werden alle Monate gestackt (80k Werte). Das mittelt über saisonale Schwankungen. Ein Feature-Paar könnte im Sommer hochkorreliert, im Winter aber informativ sein. Monatliche Korrelation mit Aggregation (z.B. min(\|r\|)) wäre differenzierter.

### Outlier Detection (Exp 04)

- **\[Paper\] Gewichteter Consensus:** Aktuell ungewichtet — jede Methode zählt gleich. Die Methoden haben aber unterschiedliche Reliabilität je Anomalie-Typ. Gewichteter Consensus (z.B. basierend auf Method-Specificity oder False-Positive-Rate) wäre methodisch sauberer.
- **\[Paper\] Soft-Weighting statt Removal:** Ablation plant nur binäre Removal-Strategien. Alternative: Sample-Gewichtung im Training (niedrigeres Gewicht für High-Severity), was gradueller ist als hartes Entfernen.

### Spatial Autocorrelation (Exp 05)

- **\[Paper\] Feature-Repräsentativität erweitern:** Aktuell nur 5 von 187 Features getestet (NDVI\_07, CHM\_1m, B8\_07, B4\_07, NDre1\_07). Features mit anderem Autokorrelationsverhalten (z.B. Winter-Bänder, CHM-Derivate) könnten längere Decay Distances zeigen. Stichprobenartig 2–3 weitere Feature-Typen testen.
- **\[Paper\] Saisonale Variation:** Nur Juli getestet. Wintermonate könnten andere räumliche Autokorrelationsmuster zeigen (Laub- vs. Nadelbäume). Zumindest ein Winter-Monat als Vergleich.

### Mixed-Genus Proximity (Exp 06)

- **\[Paper\] Soft-Weighting statt binärem Filter:** Aktuell harter Threshold (\< 20m = entfernen). Alternative: Gewichtetes Training, bei dem Bäume nahe anderer Gattungen ein niedrigeres Sample-Gewicht bekommen. Gradueller und verliert weniger Daten.
- **\[Paper\] Baumgröße berücksichtigen:** Ein großer QUERCUS hat einen deutlich größeren Kronen-Footprint als ein kleiner MALUS. Proximity-Threshold könnte genus- oder größenabhängig skaliert werden.

### Feature Extraction (Runner 02a)

- **\[Paper\] Multi-Pixel Sampling statt Point-Sampling:** Aktuell wird nur der exakte Punkt gesampelt. Ein 3×3 Median-Kernel wäre robuster gegen Sub-Pixel-Verschiebungen und Rauschen, allerdings auf Kosten räumlicher Auflösung.

### Data Quality (Runner 02b)

- **\[Paper\] Edge-NaN-Tolerance differenzierter:** Aktuell max 1 konsekutiver Edge-NaN für alle Features gleich. Features mit stärkerem saisonalen Gradient (z.B. NDVI) könnten von strengerem Edge-Threshold profitieren als stabilere Features (z.B. SWIR).

### Final Preparation (Runner 02c)

- **\[Paper\] Mahalanobis Kovarianzmatrix-Stabilität prüfen:** Bei \~187 Features nach Redundancy-Removal und kleinen Genera kann Σ⁻¹ numerisch instabil sein. Condition Number loggen, ggf. Shrinkage-Estimator.

### Feature-Design (Übergreifend)

- **\[Paper\] CHM × Pflanzjahr Wachstumsrate:** Growth-Rate-Feature (`CHM / Alter`) als biologisch fundiertere Alternative zur absoluten Baumhöhe. Entfernt Alters-Confound, potenziell besser transferierbar. Limitiert durch hohe NaN-Rate bei `plant_year` und nicht-lineare Wachstumskurven.

### Fehlende Daten aus Pipeline-Rerun (24.03.2026)

Folgende Datenpunkte wurden beim Pipeline-Rerun nicht exportiert oder nie berechnet. Bei einem Paper-Rework nachträglich erzeugen.

- **\[Paper\] CHM City-Level Welch-Teststatistik/p-Wert:** Berechnet in `exp_02`, aber nicht in `chm_assessment.json` serialisiert. Export-Logik erweitern.
- **\[Paper\] CHM Per-Genus Welch-F/p-Werte:** Nie berechnet. Welch-ANOVA pro Genus (Berlin vs. Leipzig CHM-Verteilungen) implementieren und exportieren.
- **\[Paper\] Bootstrap-CIs Algorithm Comparison:** Nie berechnet. Fold-Scores nicht gespeichert, daher Bootstrap auf Basis der CV-Folds nicht möglich. Fold-Scores serialisieren, dann Bootstrap-CIs berechnen.
- **\[Paper\] RF Feature Importances:** Kein RF-Fit gespeichert. RF nach HP-Tuning erneut fitten und Importances exportieren.
- **\[Paper\] CNN-1D Per-Genus Berlin:** Nie berechnet. Per-Genus Evaluation für CNN-1D auf Berlin Test implementieren.
- **\[Paper\] CNN-1D Per-Genus Transfer:** Berechnet, aber nicht serialisiert. Export in `transfer_metrics.json` ergänzen.
- **\[Paper\] Bootstrap-CIs Transfer:** Berechnet, aber nicht serialisiert. Export in `transfer_metrics.json` ergänzen.
- **\[Paper\] Bootstrap-CIs konsistent für alle Experimente:** CIs nur für Berlin Test-Set (XGBoost, 1D-CNN) und XGBoost Fine-Tuning vorhanden. Für Zero-Shot Transfer und 1D-CNN Fine-Tuning fehlen CIs komplett. Bootstrap-CIs (1000 Resamples, 95%) für alle ausgewerteten Test-Set-Metriken berechnen und serialisieren.
- **\[Paper\] CNN-1D From-Scratch Baseline:** Nie berechnet. From-Scratch Training für CNN-1D auf Leipzig implementieren.
- **\[Paper\] XGBoost Confusion Matrix Leipzig Fine-Tuned 100%:** In `outputs/phase_3_experiments/metadata/finetuning_curve.json` nur metadata, powerlaw\_fit, results exportiert. Confusion Matrix fehlt. Export der 100%-Fine-Tuned XGBoost Confusion Matrix ergänzen (raw + normalized CSV/JSON für `reports/data/`). Blockiert Figure A4.
- **\[Paper\] CNN-1D Confusion Matrix Berlin (exp\_10):** Nicht exportiert. CNN-1D Berlin Confusion Matrix in Phase-3-Outputs ergänzen.
- **\[Paper\] Correlation-Grafik: Indizes fehlen:** Aktuelle Korrelationsmatrix enthält nur Sentinel-2-Bänder, keine Vegetationsindizes. Index-Features in die Korrelationsvisualisierung aufnehmen.
- **\[Paper\] CNN-1D Per-Genus Fine-Tuning:** Nie berechnet. Per-Genus Evaluation über FT-Fraktionen für CNN-1D implementieren.
- **\[Paper\] Absolute Genus-Counts pro Stadt:** Vorhanden in Pipeline-Daten, aber nicht in Report-JSONs extrahiert. Export in `pipeline_summary.json` ergänzen.

## Phase 3: Experiments

### Setup-Fixierung (Exp 08–10, Phase 3a)

- **\[Paper\] Adaptive Block-Size pro Stadtgebiet:** Global 1200m Block-Size ist konservativ. Innenstadt (hohe Baumdichte, kurze Autokorrelation) vs. Suburbs (niedrige Dichte) könnten von differenzierten Block-Sizes profitieren.
- **\[Paper\] Sensitivity Analysis für Ablation-Schwellenwerte:** F1 > 0.02, Gap > 5pp etc. sind empirisch gewählt. Systematische Schwellenwert-Sensitivitätsanalyse fehlt.
- **\[Paper\] Hughes-Effekt explizit dokumentieren:** Wenn F1(all features) \< F1(top-k) → klassischer Hughes-Effekt. Diesen Zusammenhang im Bericht explizit benennen und einordnen.

### Berlin-Optimierung (Exp 11, Phase 3b)

- **\[Paper\] Multi-Seed Evaluation:** Aktuell Single Random Seed. Bootstrap-CI ist ein Kompromiss — echte Multi-Seed-Evaluation (5–10 Seeds) wäre robuster für Varianzschätzung.
- **\[Paper\] Vollständiges HP-Tuning:** Mehr Trials, voller Datensatz statt 100k-Subset, 3-Fold CV statt 1-Fold Holdout. Aktueller Speed-Run-Kompromiss für den Bericht begründen.
- **\[Paper\] Post-Training Fehleranalyse erweitern:** Genus-spezifische Analyse nach Pflanzjahr, Straßen- vs. Anlagenbaum, Kronenüberlappung etc. teilweise nicht durchgeführt. Würde Confusion-Matrix-Patterns erklären.

### Transfer-Evaluation (Phase 3c)

- **\[Paper\] Feature Stability Analysis vertiefen:** Spearman ρ = 0.45 zeigt moderate Stabilität der Feature Importances zwischen Berlin und Leipzig. Tiefere Analyse möglich: welche Feature-Gruppen sind stabil vs. instabil? Spektral vs. strukturell?
- **\[Paper\] 2D-Robustheitskategorisierung:** Genus-Performance zweidimensional einordnen (Berlin-F1 × Transfer-Δ) statt eindimensional. Ermöglicht Unterscheidung zwischen „generell schwierig" und „transfer-sensitiv".

### Finetuning (Phase 3d)

- **\[Paper\] From-Scratch Baselines bei allen Fraktionen:** Aktuell nur bei 100%. Vollständige Kurve From-Scratch vs. Finetuned über alle Fraktionen (10%–100%) würde Transfer-Vorteil sauber quantifizieren.
- **\[Paper\] Multiple Fine-Tuning-Strategien:** Nur Standard-Finetuning (alle Layer). Freeze-Layer, Discriminative Learning Rates, Gradual Unfreezing als Alternativen.
- **\[Paper\] Intelligent Sample Selection:** Statt zufälligem Subset aktive/informative Samples wählen (vgl. Tong et al. 2019). Könnte bei kleinen Fraktionen (10–25%) erheblichen Unterschied machen.
- **\[Paper\] Power-Law Extrapolation validieren:** Skalierungskurve basiert auf nur 4 Datenpunkten. Mehr Zwischenpunkte (15%, 30%, 50%, 75%) für robustere Extrapolation.

### Übergreifend Experiments

- **\[Paper\] Domain-Adaptation-Methoden (Domain Shift):** Der unbehandelte Domain Shift ist das zentrale Transferproblem. Die aktuelle Arbeit etabliert eine **Transferability-Baseline ohne explizite Domain-Adaptation**. Transfer Efficiency 67.5% (XGBoost) quantifiziert den Leistungsverlust. Nächster Schritt: DANN, Feature Alignment, MMD-Regularisierung. Im Bericht als Baseline-Argument framen: *„Diese Arbeit misst den Transfer-Verlust — die Reduktion ist Future Work."*
- **\[Paper\] Deep-Learning-Optimierung:** CNN-1D zeigt leichte Vorteile in Berlin (F1 0.602 vs. 0.571), aber deutliche Schwächen im Transfer (0.367 vs. 0.386) und Finetuning (Plateau bei \~0.44). **Empfehlung für Bericht:** CNN-1D als Kontrastpunkt stehen lassen. Story: *„XGBoost ist das stärkere Modell für tabulare Fernerkundungsdaten in diesem Setting."* Für später: LSTM, TabTransformer, InceptionTime.
- **\[Paper\] Ensemble ML+NN:** XGBoost + CNN-1D Ensemble (Stacking, Blending) — komplementäre Stärken nutzen.
- **\[Paper\] Alternative Transfer-Szenarien:** Multi-Source Transfer (Berlin + Leipzig → dritte Stadt), bidirektionaler Transfer (Leipzig → Berlin).
- **\[Paper\] Winter-Monate für Nadel-/Laubbaum-Diskriminierung:** Monate 12–3 explizit einbeziehen. Laubbäume kahl (niedrige NDVI/VARI), Nadelbäume immergrün (hohe NDVI) — biologisch einfachste Trennung, in aktueller Pipeline nicht genutzt. Könnte Group 1-Performance verbessern und Nadelbaum-Kollaps im Transfer (-61.7%) abmildern. Erfordert robustere Cloud-Maskierung und Schnee-Handling.
- **\[Paper\] Self-Supervised Pre-Training auf unlabeled Zeitreihen:** Tan et al. (2025) zeigten +9.9pp F1 durch Masked Prediction vor supervisiertem Training. Reduziert Abhängigkeit von gelabelten Daten — relevant für Transfer-Szenarien mit limitierten Target-Labels.
- **\[Paper\] GeoFM-Embeddings als alternative Feature-Repräsentation:** Earth Observation Foundation Models (AlphaEarth, TESSERA, Clay, Prithvi-EO-2.0) erzeugen dichte Vektor-Repräsentationen aus Sentinel-2-Zeitreihen, die handgefertigte Spektral- und Phänologie-Features ersetzen. Evaluierungen zeigen +3-5 F1-Punkte gegenüber klassischen S1+S2-Composites bei Vegetationsklassifikation, extreme Label-Effizienz (5% Trainingsdaten reichen für asymptotische Accuracy) und vollständige Internalisierung ökologischer Kovariaten. Drei Designentscheidungen: (1) Pixel-Level Embeddings (TESSERA, AlphaEarth) statt Patch-Level (Clay, Prithvi), da 10m-Auflösung für urbane Einzelbäume kritisch ist, (2) nonlinearer Klassifikationskopf (MLP/1D-CNN) statt linearem Probing oder RF, da die Repräsentationen auf nichtlinearen Mannigfaltigkeiten liegen, (3) Sensor Fusion mit hochauflösenden Priors (Sub-Meter Optik, LiDAR-CHM) gegen das 10m-Mischpixel-Problem. Direkt anschlussfähig an den bestehenden 1D-CNN-Workflow: TESSERA-Embeddings als Input statt handgefertigter 147-Feature-Vektoren testen. Siehe Earth Observation Foundation Models (Notion-Archiv).
- **\[Paper\] Data Augmentation für unterrepräsentierte Genera:** Temporal Jitter, spektrale Shifts, Mixup für Genera mit Komplett-Kollaps (CORYLUS, PYRUS) oder starkem Transfer-Drop. Könnte Few-Shot-Finetuning bei kleinen Fraktionen signifikant verbessern.
- **\[Paper\] Source-City-Selection Algorithmus:** Automatische Auswahl der optimalen Source-Stadt basierend auf Wasserstein Distance zwischen Genus-Histogrammen, klimatischer Ähnlichkeit und geografischer Distanz. Skalierungsvoraussetzung für Multi-City-Modell-Hubs.
- **\[Paper\] Regionale Modell-Hubs:** Ein gut trainiertes Modell lässt sich auf **mehrere Städte im selben Klimaraum** anwenden und lokal kalibrieren. Statt pro Stadt ein eigenes Modell → regionales Hub-Modell + lokale Feinanpassung. Voraussetzung: Source-City-Selection und validierte Transfer-Schwellenwerte.
- **\[Paper\] Sample-Size-Schwellenwerte für Transfer quantifizieren:** \~20k Samples reichen für Source-Niveau-Performance, \~80k für messbaren Finetuning-Vorteil gegenüber From-Scratch. Systematisch validieren (Power-Analyse, Lernkurven) und als **Planungsgrundlage für neue Städte** dokumentieren.
- **\[Paper\] Explainability-Analyse (SHAP/LIME):** Modell-Interpretierbarkeit fehlt komplett. SHAP-Values für XGBoost und Grad-CAM/LIME für CNN-1D würden Feature-Beiträge pro Genus erklären und Transfer-Unterschiede zwischen Städten interpretierbar machen.

## Aus Kritikpunkte-Review (C1–C13)

*Fehlende Items, identifiziert beim systematischen Abgleich der 13 Kritikpunkte gegen die bestehende Improvements-Liste.*

- **\[Paper\] Kontrollierte Feature-Ablation ML vs. DL:** XGBoost auf 144 Features und/oder CNN-1D auf Top-50 Features testen, um den Feature-Dimensionalitäts-Confound zwischen den Paradigmen zu isolieren. Aktuell sind Performance-Unterschiede nicht rein algorithmisch, sondern teilweise durch Feature-Unterschiede erklärbar.
- **\[Paper\] Setup-Algorithmus-Interaktion:** Setup-Entscheidungen (CHM, Proximity, Outlier, Features) mit XGBoost und/oder CNN-1D statt nur RF validieren. Optimales Setup für RF ist nicht zwingend optimal für andere Algorithmen.
- **\[Paper\] Snap-to-Peak Ground-Truth-Validierung:** Stichprobenartig korrigierte Baumpositionen gegen hochauflösende Orthophotos oder Feldbegehung validieren. Aktuell nur Adjustment-Rates (95.5%/94.7%) berichtet, keine Accuracy.
- **\[Paper\] Proximity F1 pro Threshold:** Systematisch F1-Scores für verschiedene Proximity-Thresholds (5m, 10m, 15m, 20m) evaluieren statt nur binärer Ablation (baseline vs. 5m). Ergänzt das bestehende Soft-Weighting-Item.
- **\[Paper\] Bi-weekly/dekadal Komposite:** Als Alternative zu monatlichen Medianen testen. Feinere temporale Auflösung könnte phänologische Übergänge (Blattaustrieb, Seneszenz) besser erfassen, auf Kosten höherer Cloud-Kontamination.
- **\[Paper\] Scaler-Refit beim Fine-Tuning:** Leipzig-spezifischen oder gemischten Scaler (Berlin+Leipzig) als Alternative zum Berlin-Scaler testen. Erwartete Effektrichtung: konservativer Bias durch Berlin-Scaler, der Leipzig-Features komprimiert.

---

## Engineering & Workflow

*Keine methodischen Verbesserungen, sondern infrastrukturelle Optimierungen für effizienteres Arbeiten. Relevant beim Repo-Neuaufsetzen.*

- **Performance-Optimierung der Notebooks:** Raster-Windowing (nur benötigte Regionen laden), Batch-Processing (50k Bäume statt alle), Caching häufig genutzter Arrays. Mittelfristig: Dask-Integration, Spatial Indexing (R-Tree), Parquet-Partitioning.
- **Config-Konsolidierung:** Redundante Parameter (Pfade, Schwellenwerte, Feature-Listen) aus Notebooks in zentrale YAML-Configs auslagern. Single Source of Truth für Experiment-Parameter. Mittelfristig: Pydantic-Validation, Hydra-Komposition.
- **Hybrid Local/Colab Workflow:** Für kleinere Änderungen lokale Ausführung statt Colab-Roundtrip. Google Drive direkt mit Repo verbinden. Colab nur für große Datenmengen und Modelltraining.
