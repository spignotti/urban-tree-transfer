# 03_Transfer_Evaluation

# Phase 3.4: Transfer-Evaluation

## Einleitung

Die Transfer-Evaluation ist das **Herzstück der Forschungsfrage**: Wie gut generalisieren Berlin-trainierte Modelle auf Leipziger Bäume ohne jegliche Anpassung?

Praktisch hochrelevant: Eine Stadt mit Baumkataster und ML-Ressourcen entwickelt Modelle, die andere Städte "out-of-the-box" nutzen könnten.

---

## Forschungsfragen

1. **Transfer-Gap:** Wie viel Performance verlieren Berlin-optimierte Modelle bei Zero-Shot Transfer nach Leipzig?
2. **ML vs. NN:** Transferieren ML-Modelle oder Neuronale Netze besser?
3. **Per-Genus Robustheit:** Welche Baumgattungen transferieren gut, welche schlecht?

---

## Methodik

### Zero-Shot Transfer

Modell auf Leipzig angewendet, ohne jemals Leipziger Daten gesehen zu haben. Training auf Berlin Train + Val, Evaluation auf Leipzig Test.

**Begründung:** Realistische Simulation (Stadt ohne Trainingsdaten), klare Baseline für Fine-Tuning, eindeutig messbarer Transfer-Gap.

**Kritisch — Gleicher Scaler:** Berlin-Scaler verwenden, nur transform auf Leipzig. Neuen Scaler fitten wäre implizite Adaptation an Zieldomäne → kein echtes Zero-Shot.

### Beide Champions testen

ML- und NN-Champion → ermöglicht Vergleich: "Transferieren NNs besser als ML?" Wenn ML besser transferiert → Empfehlung für ressourcenarme Städte; wenn NN → Deep Learning lohnt sich auch für Transfer.

---

## Transfer-Metriken

| Metrik | Berechnung | Interpretation |
| --- | --- | --- |
| **Absolute Drop** | F1\_Berlin − F1\_Leipzig | Direkte Performance-Differenz in Prozentpunkten |
| **Relative Drop** | (F1\_Berlin − F1\_Leipzig) / F1\_Berlin × 100 | Prozentualer Verlust bezogen auf Ausgangsperformance |

Beide zusammen: Absolute Drop zeigt Verschlechterung, Relative Drop normalisiert für Ausgangsperformance.

### Per-Genus Robustheit

| Kategorie | Relativer Drop | Interpretation |
| --- | --- | --- |
| **Robust** | \< 5% | Transferiert exzellent (innerhalb typischer Trainingsvarianz) |
| **Medium** | 5–15% | Transferiert akzeptabel |
| **Poor** | > 15% | Transfer-Probleme (spürbarer Qualitätsverlust) |

### Statistische Absicherung

**Bootstrap CI:** 1000 Resamples, 95% CI für alle Metriken.

**Transfer-Gap Signifikanz:** Mann-Whitney U Test auf Bootstrap-Verteilungen (H0: Berlin F1 = Leipzig F1, α = 0.05).

---

## A-Priori Hypothesen

**KRITISCH:** Formuliert **vor Ausführung** von 03c, um post-hoc Bias zu vermeiden. Die vier Hypothesen bilden eine logische Kette: *Transfer ist heterogen (H1) → Immergrün-Effekt (H2) → Spektrale Trennbarkeit als Prädiktor (H3) → Fine-Tuning schließt Gap genus-spezifisch (H4).*

---

### H1 (Genus-Heterogenität)

**Hypothese:** Der Transfer-Performance-Verlust variiert signifikant zwischen Genera — der Domain Shift wirkt nicht gleichmäßig.

**Begründung:** Verschiedene Genera haben unterschiedliche spektrale Signaturen, phänologische Profile und morphologische Eigenschaften. Genera mit stadtspezifischeren Merkmalen zeigen stärkeren Transfer-Verlust.

**Test:** Kruskal-Wallis H-Test auf per-sample Prediction Correctness (0/1), gruppiert nach Genus. Per-sample statt per-Genus ΔF1 → Hunderte Beobachtungen pro Genus für validen Test. **α = 0.05.**

---

### H2 (Koniferen-Robustheit)

**Hypothese:** Immergrüne Genera (Koniferen) zeigen geringere F1-Verluste beim Transfer als Laubgehölze.

**Begründung:** Fassnacht et al. (2016): Nadelbäume haben ganzjährig stabilere Spektralsignaturen. Phänologische Varianz zwischen Städten geringer bei Immergrünen.

**Test:** Mann-Whitney U zwischen Koniferen-Gaps und Laubbaum-Gaps (via `is_conifer`). **α = 0.05.**

---

### H3 (Spektrale Separabilität)

**Hypothese:** Genera mit höherer spektraler Trennbarkeit (JM-Distance) behalten beim Transfer eine höhere Klassifikationsgenauigkeit.

**Begründung:** Genera mit spektral distinkterer Source Domain haben robustere Entscheidungsgrenzen. JM-Distance (aus exp\_01) als Proxy für inter-class Separabilität. Verbindet Feature Engineering (Phase 2) direkt mit Transfer-Ergebnis (Phase 3).

**Test:** Spearman Rangkorrelation (mittlere JM-Distance pro Genus vs. Leipzig-F1). **α = 0.05.**

---

### H4 (Fine-Tuning-Effizienz)

**Hypothese:** Genera mit größerem Zero-Shot-Transfer-Verlust benötigen überproportional mehr Fine-Tuning-Daten zur Performance-Recovery.

**Begründung:** Genera mit starkem Domain Shift haben komplexere Anpassungsbedarfe. Praktische Implikation: Gezieltes genus-spezifisches Datensammeln statt flächendeckender Erhebung.

**Test:** Spearman Rangkorrelation (Zero-Shot ΔF1 vs. Fine-Tuning-Fraktion bei 90% Recovery). **α = 0.05.**

---

**Protokoll je Hypothese:** Test-Statistik, p-Wert, Effektstärke (Cohen's d / ρ / η²), Ergebnis (Bestätigt / Verworfen / Inconclusive), Interpretation in Relation zur Literatur.

---

## Feature Stability Analysis

**Zweck:** Verstehen, welche Features city-specific vs. transferable sind.

**Methodik:**

1. Leipzig-From-Scratch Training mit identischen HP wie Berlin, eigenem Scaler
2. Spearman Rank Correlation der Feature Importances (Berlin vs. Leipzig)
3. Feature-Typ-Analyse: Spectral, Red-Edge, Moisture, CHM

**Interpretation:** ρ > 0.7 hohe Stabilität, 0.5–0.7 moderat, \< 0.5 niedrig (city-specific dominiert).

**Literatur-Erwartung:** Red-Edge stabil (Immitzer 2019), CHM instabil/city-specific (Tuia 2016).

---

## Best Transfer Model Selektion

- ML relativer Drop \< NN − 5% → ML
- NN relativer Drop \< ML − 5% → NN
- Sonst: Bessere absolute Leipzig-Performance gewinnt

**Schwellenwert 5%:** Kleinere Unterschiede praktisch irrelevant, könnten durch Zufall entstehen.

---

## Praktische Implikationen

| Transfer-Drop | Empfehlung |
| --- | --- |
| \< 10% | Zero-Shot nutzbar, Fine-Tuning optional |
| 10–20% | Zero-Shot möglich, Fine-Tuning empfohlen |
| 20–30% | Zero-Shot als Startpunkt, Fine-Tuning notwendig |
| > 30% | Transfer-Learning fragwürdig, From-Scratch erwägen |
