# 02_Exploratory_06_Mixed_Genus_Proximity

## Ziel

Bestimmung des optimalen Mindestabstands zwischen Bäumen verschiedener Gattungen für spektrale Reinheit in Sentinel-2 10m-Pixeln. Bäume zu nahe an anderen Genera → Pixel-Mixing kontaminiert Spektralsignaturen.

**Status:** ⏳ In Implementierung

---

## Methodik

### Problem: Sentinel-2 Pixel Mixing

Bei 10m-Pixeln können mehrere Bäume im selben Pixel liegen → Signal = gewichtete Mischung aller Objekte.

**Kontaminationszonen:**

| Distanz | Pixel-Überlappung | Kontamination |
| --- | --- | --- |
| \< 10m | Vollständig | HIGH |
| 10–15m | Partiell | MEDIUM |
| 15–20m | Kantenkontakt | LOW |
| > 20m | Keine | NONE |

### Nearest Different-Genus Distance

Pro Baum: Minimale Distanz zum nächsten Baum einer **anderen** Gattung (euklidisch, pro Stadt).

### Threshold Sensitivity Analysis

Getestete Thresholds: \[5m, 10m, 15m, 20m, 30m\]

**Entscheidungskriterien:**

1. **Retention Rate > 85%** (Datenverlust-Limit)
2. **Genus-Uniformität:** max. Abweichung \< 10% vom Mittel (kein Genus-Bias)
3. **≥ 2× Pixel Size = 20m** (1-Pixel-Puffer garantiert)

Gewählt wird der kleinste Threshold, der alle drei Kriterien erfüllt. Erfüllt keiner alle Kriterien, fällt die Wahl auf den Threshold mit der höchsten mittleren Retention. Im finalen Lauf traf dieser Fallback zu: Ab 20m bleiben nur noch \~26–30% der Bäume erhalten, daher **5m**.

### Dual Dataset Strategy

Phase 2c erstellt **zwei** Datensatz-Varianten:

- **Baseline:** Alle Bäume (keine Proximity-Filterung) — Real-world Scenario
- **Filtered:** Bäume \< 5m zur nächsten anderen Gattung entfernt — spektral reiner

Beide durchlaufen identische Pipeline → Phase 3 Ablationsstudie: Hilft der Filter?

---

## Schlüssel-Parameter

| Parameter | Wert | Begründung |
| --- | --- | --- |
| Empfohlener Threshold | 5m | Ergebnis exp\_06 (`proximity_filter.json`): Fallback auf höchste Retention, da kein Threshold alle Kriterien erfüllt |
| Min. Retention Rate | > 85% | Ausreichend Samples für Splits |
| Genus-Uniformität | \< 10% Abweichung | Keine Klassen-Imbalance durch Filter |

**Literatur:** Powell et al. (2007, RSE) — Mixed Pixels in urbanen Gebieten häufig. Kempeneers et al. (2013, IEEE JSTARS) — Sub-Pixel-Klassifikation.

### Limitationen

- Proximity ≠ Kontamination (PSF, Atmospheric Scattering können über Pixel-Grenzen streuen)
- Binäre Entscheidung statt kontinuierlicher Gewichtung
- Nur Genus-Level, ignoriert Baumgröße und saisonale Phänologie
