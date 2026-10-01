> [!info] **Temperaturabhängiger Widerstand (NTC)**
> Spannungsteiler mit *NTC ($R(T=25°C) = 10k\ohm,\space B = 3950$),* $U_S = 5V,\space \texttt{Messbereich} = -10 \dots 50°C$
> $$\LARGE R_T = R_M \cdot \mathrm{e}^{\beta(\frac{1}{T}-\frac{1}{T_N})}$$
> 
> > [!question]+ **a)**
> > Wähle *Festwiderstand* $R_f$, sodass bei $25°C: U_{out}\approx2,5V$
> > 
> > > [!todo]- **Lösung**
> > > $$\large\displaylines{
> > > \begin{aligned}
> > > \frac{U_{out}}{U_S}&=\frac{R_i}{R + R_i}&\vert&\cdot (R+R_i)\cdot U_0\\
> > > U_{out}\cdot R+U_{out}\cdot R_i&=U_0\cdot R_i&\vert&-U_{out}\cdot R_i\\
> > > U_{out}\cdot R&=R_i(U_0-U_{out})&\vert&\div(U_0-U_{out}) \\
> > > \\\hline\\
> > > \end{aligned}\\
> > > \begin{equation}
> > > \begin{split}
> > > R_i&=\frac{U_{out}\cdot R}{U_0-U_{out}}\\
> > > &=\frac{2,5V\cdot10000\ohm}{5V-2,5V}\\
> > > &=\underline{\underline{10000\ohm =10k\ohm}}
> > > \end{split}
> > > \end{equation}
> > > }$$
> 
> > [!question]+ **b)**
> > Berechne $U_{out}$ bei $T = -10°C,\space 0°C,\space 25°C,\space 50°C$
> > 
> > > [!todo]- **Lösung**
> > > $$\LARGE\begin{equation}
> > > \begin{split}
> > > \frac{U_{out}(T)}{U_0}&=\frac{R_i}{R_T + R_i}&\vert&\cdot U_0\\
> > > U_{out}(T)&=U_0\cdot\frac{R_i}{R_T + R}
> > > \end{split}
> > > \end{equation}
> > > $$
> > > ---
> > > $$\large\begin{equation}
> > > \begin{split}
> > > R_{-10°C} &= 10k\ohm \cdot \mathrm{e}^{\frac{1}{263,15°K}-\frac{1}{298,15°K}} &= \underline{\underline{58,246K\ohm}} &= R_{-10°C}\\
> > > R_{0°C} &= 10k\ohm \cdot \mathrm{e}^{\frac{1}{273,15°K}-\frac{1}{298,15°K}} &= \underline{\underline{33,621K\ohm}} &= R_{0°C}\\
> > > R_{25°C} &= 10k\ohm \cdot \mathrm{e}^{\frac{1}{298,15°K}-\frac{1}{298,15°K}} &= \underline{\underline{10K\ohm}} &= R_{25°C}\\
> > > R_{50°C} &= 10k\ohm \cdot \mathrm{e}^{\frac{1}{333,15°K}-\frac{1}{298,15°K}} &= \underline{\underline{2,486K\ohm}} &= R_{50°C}\\
> > > \end{split}
> > > \end{equation}
> > > $$
> > > ---
> > > $$\large\begin{equation}
> > > \begin{split}
> > > U_{out}(-10°C) &= U_0 \cdot \frac{10k\ohm}{58,246k\ohm + 10k\ohm} &= \underline{\underline{1,465V}} &= U_{out}(-10°C)\\
> > > U_{out}(0°C) &= U_0 \cdot \frac{10k\ohm}{33,621k\ohm + 10k\ohm} &= \underline{\underline{2,292V}} &= U_{out}(0°C)\\
> > > U_{out}(25°C) &= U_0 \cdot \frac{10k\ohm}{10k\ohm + 10k\ohm} &= \underline{\underline{2,5V}} &= U_{out}(25°C)\\
> > > U_{out}(-10°C) &= U_0 \cdot \frac{10k\ohm}{2,486k\ohm + 10k\ohm} &= \underline{\underline{8,009V}} &= U_{out}(50°C)\\
> > > \end{split}
> > > \end{equation}$$
> 
> > [!question]+ **c)**
> > Gebe eine lineare Gleichung an, die man wählen könnte um diese Kurve abbilden zu können
> > 
> > > [!todo]- **Lösung**
> > > $$
> > > \large\boxed{\ln(U_{out}(T))=\ln\Bigg(U_0\cdot \frac{R_i}{R_i + R_N * \mathrm{e}^\beta(\frac{1}{T}-\frac{1}{T_N})}\Bigg)}
> > > $$
> 
> > [!question]+ **d)**
> > Vergleiche die Abweichungen an verschiedenen Temperaturen mit den anderen Azubis
> > 
> > > [!todo]- **Lösung**
> > > *”Hier könnte ihre Lösung stehen.”*
> 
> > [!question]+ **e)**
> > Diskutiert, wie man die gerigste Abweichung über den gesamten Verlauf bekommen könnte
> > 
> > > [!todo]- **Lösung**
> > > *”Hier könnte ihre Lösung stehen.”*
>
> > [!question]+ **f)**
> > Brückenschaltung mit zwei NTC's. Berechne $U_{ref}$ für den *Messbereich* $-10\dots50°C$
> > 
> > > [!to-do]- **Lösung**
> > > $$\tiny\boxed{U_{Br}(T)=U_S \cdot \frac{R_f}{R_N \cdot \mathrm{e}^{\beta(\frac{1}{T}-\frac{1}{T_N})} + R_f} - U_S \cdot \frac{R_N \cdot \mathrm{e}^{\beta(\frac{1}{T}-\frac{1}{T_N})}}{R_f + R_N \cdot \mathrm{e}^{\beta(\frac{1}{T}-\frac{1}{T_N})}} }$$
> > > ---
> > > $$\large\begin{equation}
> > > \begin{split}
> > > U_{Br}(T)&=U_S \cdot \frac{R_f}{R_N \cdot \mathrm{e}^{\beta(\frac{1}{T}-\frac{1}{T_N})} + R_f} - U_S \cdot \frac{R_N \cdot \mathrm{e}^{\beta(\frac{1}{T}-\frac{1}{T_N})}}{R_f + R_N \cdot \mathrm{e}^{\beta(\frac{1}{T}-\frac{1}{T_N})}} \\
> > > &= 0,733V-4,267V = \underline{\underline{-3,534V}}\\
> > > \end{split}
> > > \end{equation}
> > > $$
> 
> 
> > [!question]+ **g)**
> > Plotte die Ergebnisse von b) und f) in einen Plot und vergleiche die Ergebnisse.
> > 
> > > [!todo]- **Lösung**
> > > *g): Lösung*
> > >
>
> > [!question]+ **h)**
> > Plotte eine Grafik (in `matplotlib`) für die NTC Spannungen.
> > 1. Die Punkte
> > 2. Die Kurve, ausgerechnet über $R_N \cdot \mathrm{e}^{B(\frac{1}{T} - \frac{1}{T_N}}$
> > 3. Die lineare Approximation von uns mit $-10...50°C$.
> > 4. Die lineare Approximation im Arbeitspunkt z.B. $20°C$ über die Ableitung.
> > 5. Die lineare Regression über die Messpunkte.
> > 
> > > [!todo]- **Lösung**
> > > *h): Lösung*
> > >
