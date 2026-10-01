# f) Brückenschaltung mit zwei NTCs

Gegeben: NTC mit $R_N = R(25\,°C) = 10\,k\Omega$, $\beta = 3950\,K$, $U_S = 5\,V$, Messbereich $-10 \dots 50\,°C$.

$$R_T = R_N \cdot \mathrm{e}^{\beta\left(\frac{1}{T}-\frac{1}{T_N}\right)}$$

## 1. Aufbau der Schaltung

Belegung der Brücke:

- **R₁ = NTC** und **R₄ = NTC**. Die beiden NTCs liegen diagonal zueinander.
- **R₂ = R₃ = R_f = 10 kΩ**

Die Brücke besteht aus zwei Spannungsteilern, die nebeneinander an $U_S$ hängen. Gemessen wird die Differenz ihrer Mittelpunkte A und B:

$$U_A = U_S\cdot\frac{R_f}{R_T+R_f}\qquad U_B = U_S\cdot\frac{R_T}{R_f+R_T}$$

$$U_{Br} = U_A - U_B = U_S\cdot\frac{R_f - R_T}{R_f + R_T}$$

**Warum zwei NTCs?** Wird es wärmer, sinkt $R_T$. Dadurch **steigt** $U_A$, und gleichzeitig **fällt** $U_B$. Beide Effekte addieren sich, die Empfindlichkeit verdoppelt sich. Bei 25 °C gilt $R_T = R_f$, die Brücke ist dann abgeglichen und $U_{Br} = 0\,V$.

**Kompakte Form:** Setzt man $R_T$ ein und wählt $R_f = R_N$, lässt sich das umformen zu

$$U_{Br}(T) = U_S\cdot\tanh\!\Big(\tfrac{\beta}{2}\big(\tfrac{1}{T_N}-\tfrac{1}{T}\big)\Big)$$

$\tanh$ verläuft um 0 herum fast wie eine Gerade. Deshalb lässt sich die Brücke deutlich besser linearisieren als der einfache Spannungsteiler aus b).

## 2. Werte im Messbereich

| ϑ | $R_T$ | $U_{Br}$ |
|---|---|---|
| −10 °C | 58,246 kΩ | −3,535 V |
| 0 °C | 33,621 kΩ | −2,708 V |
| 10 °C | 20,175 kΩ | −1,686 V |
| 20 °C | 12,535 kΩ | −0,563 V |
| 25 °C | 10,000 kΩ | 0 V |
| 30 °C | 8,037 kΩ | +0,544 V |
| 40 °C | 5,301 kΩ | +1,535 V |
| 50 °C | 3,588 kΩ | +2,359 V |

## 3. Lineare Gleichung $U(\vartheta) = m\cdot\vartheta + b$

### Variante A: Gerade durch die Endpunkte des Messbereichs (−10 °C und 50 °C)

$$m = \frac{U(50) - U(-10)}{50-(-10)} = \frac{2{,}359 - (-3{,}535)}{60\,\text{K}} = 0{,}0982\,\tfrac{\text{V}}{\text{K}}$$

$$b = U(-10) - m\cdot(-10) = -3{,}535 + 0{,}982 = -2{,}552\,\text{V}$$

$$\boxed{U_{Br}(\vartheta) \approx 0{,}0982\,\tfrac{\text{V}}{°\text{C}}\cdot\vartheta - 2{,}552\,\text{V}}$$

Abweichung zur echten Kurve:

| ϑ | −10 °C | 0 °C | 10 °C | 25 °C | 40 °C | 50 °C |
|---|---|---|---|---|---|---|
| $U_{Br} - U_{lin}$ | 0 V | −0,155 V | −0,116 V | +0,097 V | +0,158 V | 0 V |

Die größte Abweichung liegt bei etwa **±0,16 V** (→ Material für d).

### Variante B: Tangente im Arbeitspunkt 25 °C (über die Ableitung, gehört zu h4)

$$m = \frac{U_S\cdot\beta}{2\,T_N^2} = \frac{5\cdot3950}{2\cdot298{,}15^2} = 0{,}111\,\tfrac{\text{V}}{\text{K}} \quad\Rightarrow\quad U_{Br}\approx 0{,}111\,\tfrac{\text{V}}{\text{K}}\cdot(\vartheta - 25\,°C)$$

Diese Gerade ist bei 25 °C exakt. An den Rändern des Messbereichs weicht sie stärker ab als Variante A. Daraus folgt die Antwort auf e): Die geringste Abweichung über den gesamten Verlauf bekommt man nur mit einer passend gewählten Geraden, zum Beispiel per linearer Regression über alle Messpunkte (h5).

## 4. Korrekturen zu Teil b)

1. **50 °C = 323,15 K**, nicht 333,15 K. Mit 333,15 K wurden 60 °C gerechnet. Richtig ist $R_{50°C} = 3{,}588\,k\Omega$.
2. **$U_{out}(50\,°C) = 8{,}009\,V$ ist unmöglich**, weil der Wert größer als $U_S = 5\,V$ ist. Richtig ist $5\,V \cdot \frac{10}{13{,}588} = 3{,}680\,V$. Außerdem steht in dieser Zeile das Label „−10 °C“ statt „50 °C“.
3. In den R-Formeln fehlt im Exponenten das **β**. Die Zahlenwerte für −10, 0 und 25 °C sind aber mit β gerechnet und stimmen.
4. Die Einheit ist **K**, nicht °K.
5. Das c)-Ergebnis mit $\ln(\dots)$ ist keine lineare Gleichung. Gesucht ist eine Gerade $m\cdot\vartheta + b$, angepasst an die Kurve aus b), genau wie in Abschnitt 3.
