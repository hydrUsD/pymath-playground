import et_formeln as ef
ef.resonanzfrequenz(1e-3, 1e-6)                 # 5032.92 Hz
ef.kompensation_kapazitaet(10e3, 0.7, 0.95, 50, 230)
ef.formatiere_si(ef.kapazitaet_platten(4, 1e-2, 1e-3), "F")   # '354.2 pF'
ef.katalog("Filter")                            # Formeln einer Kategorie
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3, floatmode="fixed")

T_c = np.linspace(-25, 75, 20)
T_k = np.add(T_c, ef.T_NULL_C)
T_N = 298.15
R_N = 10000
B = 3950
U_0 = 10

def dT(_t):
    return (( 1 / _t )- ( 1 / T_N ))

def R_T(_t):
    return ( R_N * np.e ** ( 3950 * dT(_t) ) )

def U_T(_t):
    return ( U_0 * ( R_N / ( R_T(T_k) + R_N )))

fig, ax = plt.subplots()
ax.plot(T_c, U_T(T_k))
plt.show()