import et_formeln as ef
ef.resonanzfrequenz(1e-3, 1e-6)                 # 5032.92 Hz
ef.kompensation_kapazitaet(10e3, 0.7, 0.95, 50, 230)
ef.formatiere_si(ef.kapazitaet_platten(4, 1e-2, 1e-3), "F")   # '354.2 pF'
ef.katalog("Filter")                            # Formeln einer Kategorie
ef.ntc_widerstand(323.15, 10e3, 3950, 298.15)   # NTC 10k/B3950 bei 50 °C: 3588 Ohm
ef.umstellen(ef.resonanzfrequenz, "C", 1000.0, L=10e-3)   # Formel nach C umstellen: 2.533e-6 F
ef.dezimal_zu_hex(419)                          # '1A3'
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

np.set_printoptions(precision=3, floatmode="fixed")

T_c = np.linspace(-10, 50, 61)   # Messbereich in 1-K-Schritten
T_k = np.add(T_c, ef.T_NULL_C)
T_N = 298.15
R_N = np.float64(10000)
R_f = R_N
B = 3950
U_S = 5
U_0 = U_S
MESSBEREICH = np.array([-10, 50])
Br = "Br"
dT = np.subtract(np.divide(1, T_k),  1/ T_N)
print("dT: ", dT)
R_T = R_N * np.exp(B * dT)
print("R_T: ", R_T)
U_A = ef.spannungsteiler(U_S, R_T, R_f)   # Spannung ueber R_f (R2-Argument)
print("U_A: ", U_A)
U_B = ef.spannungsteiler(U_S, R_f, R_T)   # Spannung ueber dem NTC
print("U_B: ", U_B)
U_BR = np.subtract(U_A, U_B)
print("U_BR: ", U_BR)

def dt(_t):
    _T_k = ef.celsius_zu_kelvin(_t)
    _result = np.subtract(np.divide(1, _T_k),( 1 / T_N ))
    return _result

def r_t(_t):
    _result = np.multiply(R_N, np.exp(np.multiply(3950, dt(_t))))
    return np.round(_result, 3)

def u_a(_t):
    _result = np.multiply(U_S, np.divide(R_f, np.add(r_t(_t), R_f)))
    return np.round(_result, 3)



def u_b(_t):
    _result = np.multiply(U_S, np.divide(r_t(_t), np.add(R_f, r_t(_t))))
    return np.round(_result, 3)



def u_br(_t):
    _T_k = ef.celsius_zu_kelvin(_t)
    _result = np.multiply(U_S, np.tanh(np.multiply((B / 2), np.subtract((1 / T_N), np.divide(1, _T_k)))))
    return np.round(_result, 3)

print("U_BR (tanh): ", u_br(T_c))



def lin_var_a(_t):
    _m = np.divide(np.subtract(U_BR[-1], U_BR[0]), np.subtract(ef.celsius_zu_kelvin(MESSBEREICH[1]), ef.celsius_zu_kelvin(MESSBEREICH[0])))
    _b = np.subtract(U_BR[0], np.multiply(_m, MESSBEREICH[0]))
    _result = np.add(np.multiply(_m, _t), _b)
    return np.round(_result, 3)

U_Br_a = lin_var_a(T_c)
print("U_Br_a: ", U_Br_a)

def lin_var_b(_t):
    _m = ( U_S * B ) / ( 2 * ( T_N ** 2 ))
    _T_N_C = ef.kelvin_zu_celsius(T_N)
    _result = np.multiply(_m, np.subtract(_t, _T_N_C))
    return np.round(_result, 3)

U_Br_b = lin_var_b(T_c)
print("U_Br_b: ", U_Br_b)



fig, ax = plt.subplots()
ax.grid(True)
ax.plot(T_c, np.asarray(U_BR), linewidth=2, label=r"Exponentiale Gl.: $U_{Br}(T)=U_a - U_b$")#, label=r"$U_{Br}(T) = U_S \cdot \tanh\left(\frac{\beta}{2}\left(\frac{1}{T_N} - \frac{1}{T}\right)\right)$")
ax.plot(T_c, np.asarray(U_Br_a), linewidth=2, label=r"Lineare Gl. (A): $U(\vartheta)=m\cdot\vartheta+b$")#, label=r"$U(\vartheta) = m\cdot\vartheta + b = U_{Br}(\vartheta) \approx 0,0982\,\frac{V}{°C}\cdot\vartheta - 2,552\,V$")
ax.plot(T_c, np.asarray(U_Br_b), linewidth=2, label=r"Lineare Gl. (B): $U_{Br}\approx 0,111\frac{V}{K}\cdot(\vartheta-25°C)$")#, label=r"$U(\vartheta) = m\cdot(\vartheta - T_N) = U_{Br}\approx 0,111\,\frac{V}{K}\cdot(\vartheta - 25\,°C)$")
ax.set(xlim=(-10, 50), ylim=(-5, 5), xlabel="Temperature (°C)", ylabel="U (V)", title="NTC Spannungsabfall")
ax.legend(fontsize=11)
plt.pause(1)
fig.canvas.draw_idle()
fig.canvas.flush_events()
plt.show()