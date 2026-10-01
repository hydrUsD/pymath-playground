# Elektrotechnik-Formelsammlung (Python + matplotlib)

| Datei | Inhalt |
|---|---|
| `et_formeln.py` | ~210 Formeln & Einheitenumrechnungen in 14 Kategorien, SI-Basiseinheiten, NumPy-fähig (Skalare & Arrays) |
| `et_plots.py` | 14 matplotlib-Plots (RC/RLC, Bode, Resonanz, Drehstrom, Kompensation, Kloss, Diode, …) |
| `test_et_formeln.py` | 53 Tests: Dimensionsprüfung, Referenzwerte, Identitäten, numerische Gegenrechnung |
| `plots_vorschau/` | Beispielausgabe aller Plots |

Benötigt nur `numpy` und `matplotlib`.

```bash
python et_formeln.py                 # Formelkatalog + Dimensionspruefung
python et_plots.py --liste           # Plots auflisten
python et_plots.py bode drehstrom    # Plots anzeigen
python et_plots.py --alle --speichern plots
python test_et_formeln.py            # Verifikation
```

```python
import et_formeln as ef
ef.resonanzfrequenz(1e-3, 1e-6)                 # 5032.92 Hz
ef.kompensation_kapazitaet(10e3, 0.7, 0.95, 50, 230)
ef.formatiere_si(ef.kapazitaet_platten(4, 1e-2, 1e-3), "F")   # '354.2 pF'
ef.katalog("Filter")                            # Formeln einer Kategorie
```

Konventionen: alle Größen in SI (Winkel in rad, Drehzahl in 1/s); Umrechnungen (rpm, °C, dB, AWG …)
stehen in der Kategorie `Einheiten`. Variadische Funktionen: `widerstand_parallel(*R)`, `kapazitaet_reihe(*C)` u. a.
