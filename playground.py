import et_formeln as ef
ef.resonanzfrequenz(1e-3, 1e-6)                 # 5032.92 Hz
ef.kompensation_kapazitaet(10e3, 0.7, 0.95, 50, 230)
ef.formatiere_si(ef.kapazitaet_platten(4, 1e-2, 1e-3), "F")   # '354.2 pF'
ef.katalog("Filter")                            # Formeln einer Kategorie
import matplotlib.pyplot as plt
import numpy as np


T_c = np.linspace(-25, 75, 20)
T_k = np.add(T_c, ef.T_NULL_C)

fig, ax = plt.subplots()
ax.plot(T_c, T_k)
plt.show()