# Elektrotechnik-Formelsammlung (Python + matplotlib)

| Datei | Inhalt |
|---|---|
| `et_formeln.py` | ~600 Formeln & Einheitenumrechnungen in 24 Kategorien, SI-Basiseinheiten, NumPy-fähig (Skalare & Arrays), dazu Digitaltechnik- und Mathe-Helfer |
| `ek_abdeckung.py` | Abgleich mit der [Formelsammlung von Elektronik Kompendium](https://www.elektronik-kompendium.de/sites/formeln/): jede der 343 Themenseiten → Funktionen, inkl. Hinweisen zu fehlerhaften EK-Formeln |
| `et_plots.py` | 14 matplotlib-Plots (RC/RLC, Bode, Resonanz, Drehstrom, Kompensation, Kloss, Diode, …) |
| `test_et_formeln.py` | 53 Tests: Dimensionsprüfung, Referenzwerte, Identitäten, numerische Gegenrechnung |
| `test_ek_abdeckung.py` | 38 Tests: EK-Abdeckung vollständig + Referenzwerte der Ergänzungen |
| `plots_vorschau/` | Beispielausgabe aller Plots |

Benötigt nur `numpy` und `matplotlib`.

```bash
python et_formeln.py                 # Formelkatalog + Dimensionspruefung
python et_plots.py --liste           # Plots auflisten
python et_plots.py bode drehstrom    # Plots anzeigen
python et_plots.py --alle --speichern plots
python test_et_formeln.py            # Verifikation
python test_ek_abdeckung.py          # Verifikation der EK-Ergaenzungen
python ek_abdeckung.py               # Abdeckungsbericht EK-Seite -> Funktionen
```

```python
import et_formeln as ef
ef.resonanzfrequenz(1e-3, 1e-6)                 # 5032.92 Hz
ef.kompensation_kapazitaet(10e3, 0.7, 0.95, 50, 230)
ef.formatiere_si(ef.kapazitaet_platten(4, 1e-2, 1e-3), "F")   # '354.2 pF'
ef.katalog("Filter")                            # Formeln einer Kategorie
ef.ntc_widerstand(323.15, 10e3, 3950, 298.15)   # NTC 10k/B3950 bei 50 °C: 3588 Ohm
ef.umstellen(ef.resonanzfrequenz, "C", 1000.0, L=10e-3)   # Formel nach C umstellen: 2.533e-6 F
ef.dezimal_zu_hex(419)                          # '1A3'
```

Konventionen: alle Größen in SI (Winkel in rad, Drehzahl in 1/s); Umrechnungen (rpm, °C, dB, AWG …)
stehen in der Kategorie `Einheiten`. Variadische Funktionen: `widerstand_parallel(*R)`, `kapazitaet_reihe(*C)` u. a.
