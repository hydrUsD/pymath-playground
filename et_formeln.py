#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
et_formeln.py - Elektrotechnik-Formelsammlung mit Einheitenumrechnung
=====================================================================

Alle Formeln arbeiten in **SI-Basiseinheiten** (V, A, Ohm, F, H, s, Hz, m, kg, K, rad)
und akzeptieren Skalare *und* NumPy-Arrays (fuer matplotlib-Plots).

Aufbau
------
* Konstanten (CODATA 2018)
* Registry + Dimensionspruefung  (jede Formel traegt ihre Einheiten als Metadaten)
* Kategorien: Gleichstrom | Bauelemente | Felder | Wechselstrom | Drehstrom |
              Transformator | Schwingkreise | Schaltvorgaenge | Filter |
              Verstaerker | Halbleiter | Leitungen | Maschinen | Einheiten

Qualitaetssicherung
-------------------
1. ``pruefe_alle_dimensionen()``  - Dimensionshomogenitaet (Skalierungstest je SI-Basiseinheit)
2. ``test_et_formeln.py``         - Referenzwerte, physikalische Identitaeten, numerische Gegenrechnung

Aufruf ``python et_formeln.py`` gibt den Formelkatalog aus.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Konstanten (CODATA 2018) - Dimensionen siehe KONST_DIM
# ---------------------------------------------------------------------------
PI = math.pi
C0 = 299_792_458.0            # m/s      Lichtgeschwindigkeit (exakt)
MU0 = 1.25663706212e-6        # H/m      magnetische Feldkonstante
EPS0 = 8.8541878128e-12       # F/m      elektrische Feldkonstante
E_ELEM = 1.602176634e-19      # C        Elementarladung (exakt)
K_B = 1.380649e-23            # J/K      Boltzmann-Konstante (exakt)
H_PLANCK = 6.62607015e-34     # J*s      Planck-Konstante (exakt)
T_NULL_C = 273.15             # K        0 Grad Celsius (Umrechnung, keine Dimension)

KONST_DIM: Dict[str, str] = {
    "C0": "m/s",
    "MU0": "H/m",
    "EPS0": "F/m",
    "E_ELEM": "C",
    "K_B": "J/K",
    "H_PLANCK": "J*s",
}

# ---------------------------------------------------------------------------
# Dimensionsrechnung  (kg, m, s, A, K)
# ---------------------------------------------------------------------------
_BASIS = ("kg", "m", "s", "A", "K")
_EINHEITEN: Dict[str, Tuple[int, ...]] = {
    "1": (0, 0, 0, 0, 0), "rad": (0, 0, 0, 0, 0),
    "kg": (1, 0, 0, 0, 0), "m": (0, 1, 0, 0, 0), "s": (0, 0, 1, 0, 0),
    "A": (0, 0, 0, 1, 0), "K": (0, 0, 0, 0, 1),
    "Hz": (0, 0, -1, 0, 0), "N": (1, 1, -2, 0, 0), "J": (1, 2, -2, 0, 0),
    "W": (1, 2, -3, 0, 0), "VA": (1, 2, -3, 0, 0), "var": (1, 2, -3, 0, 0),
    "C": (0, 0, 1, 1, 0), "V": (1, 2, -3, -1, 0), "F": (-1, -2, 4, 2, 0),
    "ohm": (1, 2, -3, -2, 0), "S": (-1, -2, 3, 2, 0), "Wb": (1, 2, -2, -1, 0),
    "T": (1, 0, -2, -1, 0), "H": (1, 2, -2, -2, 0),
}


def dim_vektor(ausdruck: str) -> np.ndarray:
    """Einheitenausdruck ('V/A', 'kg*m^2', 'J/kg/K', 'm^-3') -> Exponentenvektor (kg,m,s,A,K)."""
    tokens = re.findall(r"[A-Za-z]+|\d+|\^|\*|/|\(|\)|-", ausdruck)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def nehmen():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def atom():
        t = nehmen()
        if t == "(":
            v = expr()
            assert nehmen() == ")", f"Klammer fehlt in '{ausdruck}'"
        elif t in _EINHEITEN:
            v = np.array(_EINHEITEN[t], dtype=float)
        else:
            raise ValueError(f"Unbekannte Einheit '{t}' in '{ausdruck}'")
        if peek() == "^":
            nehmen()
            vz = -1 if peek() == "-" else 1
            if vz == -1:
                nehmen()
            v = v * vz * int(nehmen())
        return v

    def expr():
        v = atom()
        while peek() in ("*", "/"):
            op = nehmen()
            w = atom()
            v = v + w if op == "*" else v - w
        return v

    ergebnis = expr()
    if pos != len(tokens):
        raise ValueError(f"Nicht vollstaendig geparst: '{ausdruck}'")
    return ergebnis


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REGISTRY: Dict[str, Callable] = {}


def formel(name: str, gleichung: str, ein: Dict[str, str], aus, kat: str,
           bsp: Dict[str, object], dimcheck: bool = True):
    """Decorator: registriert Formel samt Einheiten-Metadaten und Rechenbeispiel."""
    def deco(fn):
        fn.meta = dict(name=name, gleichung=gleichung, ein=ein, aus=aus, kat=kat,
                       bsp=bsp, dimcheck=dimcheck)
        REGISTRY[fn.__name__] = fn
        return fn
    return deco


def _skal(einheit: str, lam: np.ndarray) -> float:
    return float(np.prod(lam ** dim_vektor(einheit)))


def pruefe_dimension(fn: Callable, lam: Iterable[float]) -> Tuple[bool, str]:
    """Dimensionshomogenitaet: Skaliert man die SI-Basiseinheiten (kg,m,s,A,K) um lam,
    muss sich das Ergebnis exakt um lam^(Dimension des Ergebnisses) aendern."""
    lam = np.asarray(list(lam), dtype=float)
    m = fn.meta
    g = globals()
    basis_args = {k: np.asarray(v) for k, v in m["bsp"].items()}
    ref = fn(**basis_args)
    orig = {k: g[k] for k in KONST_DIM}
    try:
        for k, d in KONST_DIM.items():
            g[k] = orig[k] * _skal(d, lam)
        args = {k: v * _skal(m["ein"][k], lam) for k, v in basis_args.items()}
        res = fn(**args)
    finally:
        for k, v in orig.items():
            g[k] = v
    aus = m["aus"] if isinstance(m["aus"], (tuple, list)) else (m["aus"],)
    ref = ref if isinstance(ref, tuple) else (ref,)
    res = res if isinstance(res, tuple) else (res,)
    for i, (r0, r1, a) in enumerate(zip(ref, res, aus)):
        soll = np.asarray(r0) * _skal(a, lam)
        if not np.allclose(r1, soll, rtol=1e-9, atol=0):
            return False, f"{fn.__name__}: Ausgabe {i} ('{a}') inkonsistent: {r1} != {soll}"
    return True, "ok"


def pruefe_alle_dimensionen(verbose: bool = False) -> bool:
    """Prueft alle registrierten Formeln (je Basiseinheit einzeln + alle gleichzeitig)."""
    faktoren = [1.7, 2.3, 3.1, 1.3, 2.9]
    lams = [np.array(faktoren)] + [np.array([f if i == j else 1.0 for i, f in enumerate(faktoren)])
                                   for j in range(5)]
    alles_ok = True
    for name, fn in REGISTRY.items():
        if not fn.meta["dimcheck"]:
            continue
        for lam in lams:
            ok, msg = pruefe_dimension(fn, lam)
            if not ok:
                alles_ok = False
                print("FEHLER:", msg)
                break
        else:
            if verbose:
                print(f"  ok  {name}")
    return alles_ok


# ===========================================================================
# 1. GLEICHSTROM UND NETZWERKE
# ===========================================================================
K1 = "Gleichstrom"


@formel("Ohmsches Gesetz (Spannung)", "U = R * I", {"R": "ohm", "I": "A"}, "V", K1, dict(R=100.0, I=0.5))
def spannung_ohm(R, I):
    return R * I


@formel("Ohmsches Gesetz (Strom)", "I = U / R", {"U": "V", "R": "ohm"}, "A", K1, dict(U=12.0, R=6.0))
def strom_ohm(U, R):
    return U / R


@formel("Ohmsches Gesetz (Widerstand)", "R = U / I", {"U": "V", "I": "A"}, "ohm", K1, dict(U=12.0, I=2.0))
def widerstand_ohm(U, I):
    return U / I


@formel("Leitwert", "G = 1 / R", {"R": "ohm"}, "S", K1, dict(R=50.0))
def leitwert(R):
    return 1.0 / R


@formel("Elektrische Leistung", "P = U * I", {"U": "V", "I": "A"}, "W", K1, dict(U=230.0, I=10.0))
def leistung_ui(U, I):
    return U * I


@formel("Leistung (Strom, Widerstand)", "P = I^2 * R", {"I": "A", "R": "ohm"}, "W", K1, dict(I=2.0, R=10.0))
def leistung_i2r(I, R):
    return I ** 2 * R


@formel("Leistung (Spannung, Widerstand)", "P = U^2 / R", {"U": "V", "R": "ohm"}, "W", K1, dict(U=12.0, R=6.0))
def leistung_u2r(U, R):
    return U ** 2 / R


@formel("Elektrische Arbeit / Energie", "W = P * t", {"P": "W", "t": "s"}, "J", K1, dict(P=1000.0, t=3600.0))
def energie(P, t):
    return P * t


@formel("Elektrische Ladung", "Q = I * t", {"I": "A", "t": "s"}, "C", K1, dict(I=2.0, t=1800.0))
def ladung(I, t):
    return I * t


@formel("Stromdichte", "J = I / A", {"I": "A", "A": "m^2"}, "A/m^2", K1, dict(I=16.0, A=2.5e-6))
def stromdichte(I, A):
    return I / A


@formel("Widerstand eines Leiters", "R = rho * l / A", {"rho": "ohm*m", "l": "m", "A": "m^2"}, "ohm", K1,
        dict(rho=1.72e-8, l=100.0, A=1.5e-6))
def widerstand_leiter(rho, l, A):
    return rho * l / A


@formel("Widerstand bei Temperatur (linear)", "R = R0 * (1 + alpha * dT)",
        {"R0": "ohm", "alpha": "1/K", "dT": "K"}, "ohm", K1, dict(R0=1.0, alpha=0.00393, dT=100.0))
def widerstand_temperatur(R0, alpha, dT):
    return R0 * (1.0 + alpha * dT)


@formel("Spannungsfall Gleichstrom (Hin+Rueck)", "dU = 2 * rho * l * I / A",
        {"I": "A", "rho": "ohm*m", "l": "m", "A": "m^2"}, "V", K1, dict(I=16.0, rho=1.72e-8, l=30.0, A=2.5e-6))
def spannungsfall_gleichstrom(I, rho, l, A):
    return 2.0 * rho * l * I / A


def widerstand_reihe(*R):
    """Reihenschaltung: R = R1 + R2 + ... (variadisch)"""
    return sum(np.asarray(r, dtype=float) for r in R)


def widerstand_parallel(*R):
    """Parallelschaltung: 1/R = 1/R1 + 1/R2 + ... (variadisch)"""
    return 1.0 / sum(1.0 / np.asarray(r, dtype=float) for r in R)


@formel("Parallelschaltung (2 Widerstaende)", "R = R1*R2 / (R1+R2)", {"R1": "ohm", "R2": "ohm"}, "ohm", K1,
        dict(R1=100.0, R2=300.0))
def widerstand_parallel2(R1, R2):
    return R1 * R2 / (R1 + R2)


@formel("Spannungsteiler (unbelastet)", "U2 = U * R2 / (R1 + R2)", {"U": "V", "R1": "ohm", "R2": "ohm"}, "V", K1,
        dict(U=12.0, R1=1000.0, R2=2000.0))
def spannungsteiler(U, R1, R2):
    return U * R2 / (R1 + R2)


@formel("Stromteiler (Strom durch R1)", "I1 = I * R2 / (R1 + R2)", {"I": "A", "R1": "ohm", "R2": "ohm"}, "A", K1,
        dict(I=3.0, R1=1.0, R2=2.0))
def stromteiler(I, R1, R2):
    return I * R2 / (R1 + R2)


@formel("Klemmenspannung realer Quelle", "U = Uq - Ri * I", {"Uq": "V", "Ri": "ohm", "I": "A"}, "V", K1,
        dict(Uq=12.6, Ri=0.05, I=20.0))
def klemmenspannung(Uq, Ri, I):
    return Uq - Ri * I


@formel("Max. Leistung bei Anpassung (Ra = Ri)", "Pmax = Uq^2 / (4 Ri)", {"Uq": "V", "Ri": "ohm"}, "W", K1,
        dict(Uq=10.0, Ri=5.0))
def leistung_max_anpassung(Uq, Ri):
    return Uq ** 2 / (4.0 * Ri)


@formel("Wirkungsgrad der Quelle", "eta = Ra / (Ri + Ra)", {"Ri": "ohm", "Ra": "ohm"}, "1", K1,
        dict(Ri=5.0, Ra=15.0))
def wirkungsgrad_quelle(Ri, Ra):
    return Ra / (Ri + Ra)


@formel("Stern -> Dreieck", "R12 = R1+R2+R1*R2/R3 (zyklisch)", {"R1": "ohm", "R2": "ohm", "R3": "ohm"},
        ("ohm", "ohm", "ohm"), K1, dict(R1=10.0, R2=20.0, R3=30.0))
def stern_zu_dreieck(R1, R2, R3):
    return (R1 + R2 + R1 * R2 / R3,
            R2 + R3 + R2 * R3 / R1,
            R3 + R1 + R3 * R1 / R2)


@formel("Dreieck -> Stern", "R1 = R12*R31/(R12+R23+R31) (zyklisch)",
        {"R12": "ohm", "R23": "ohm", "R31": "ohm"}, ("ohm", "ohm", "ohm"), K1,
        dict(R12=30.0, R23=40.0, R31=50.0))
def dreieck_zu_stern(R12, R23, R31):
    s = R12 + R23 + R31
    return R12 * R31 / s, R12 * R23 / s, R23 * R31 / s


@formel("Wheatstone-Bruecke (abgeglichen)", "Rx = R3 * R1 / R2", {"R1": "ohm", "R2": "ohm", "R3": "ohm"}, "ohm", K1,
        dict(R1=100.0, R2=200.0, R3=50.0))
def wheatstone_rx(R1, R2, R3):
    """Bruecke: linker Zweig R1 (oben)/R2 (unten), rechter Zweig Rx (oben)/R3 (unten).
    Abgleich (Diagonalspannung 0): R1/R2 = Rx/R3."""
    return R3 * R1 / R2


@formel("Erwaermung durch Verlustleistung", "dT = P * t / (m * c)",
        {"P": "W", "t": "s", "m": "kg", "c": "J/kg/K"}, "K", K1, dict(P=2000.0, t=60.0, m=1.0, c=4182.0))
def erwaermung_dT(P, t, m, c):
    return P * t / (m * c)


@formel("Knotenpotentialverfahren", "G * phi = I  ->  phi", {"G": "S", "I": "A"}, "V", K1,
        dict(G=[[3.0, -1.0], [-1.0, 2.0]], I=[1.0, 2.0]))
def knotenpotentiale(G, I):
    """Loest G*phi = I (Leitwertmatrix in S, eingepraegte Knotenstroeme in A)."""
    return np.linalg.solve(np.asarray(G, dtype=float), np.asarray(I, dtype=float))


@formel("Laufzeit Batterie/Akku", "t = Q / I", {"Q": "C", "I": "A"}, "s", K1, dict(Q=7200.0, I=0.5))
def batterie_laufzeit(Q, I):
    """Q in Coulomb (1 Ah = 3600 C, siehe ah_zu_coulomb)."""
    return Q / I


# ===========================================================================
# 2. BAUELEMENTE (C, L)
# ===========================================================================
K2 = "Bauelemente"


@formel("Plattenkondensator", "C = eps0*eps_r*A/d", {"eps_r": "1", "A": "m^2", "d": "m"}, "F", K2,
        dict(eps_r=1.0, A=1e-4, d=1e-3))
def kapazitaet_platten(eps_r, A, d):
    return EPS0 * eps_r * A / d


@formel("Koaxialkondensator / -kabel", "C = 2 pi eps0 eps_r l / ln(ra/ri)",
        {"eps_r": "1", "l": "m", "r_i": "m", "r_a": "m"}, "F", K2, dict(eps_r=2.25, l=1.0, r_i=0.45e-3, r_a=1.5e-3))
def kapazitaet_koax(eps_r, l, r_i, r_a):
    return 2.0 * PI * EPS0 * eps_r * l / np.log(r_a / r_i)


@formel("Kugelkondensator (freie Kugel)", "C = 4 pi eps0 eps_r r", {"eps_r": "1", "r": "m"}, "F", K2,
        dict(eps_r=1.0, r=0.1))
def kapazitaet_kugel(eps_r, r):
    return 4.0 * PI * EPS0 * eps_r * r


@formel("Induktivitaet Zylinderspule (lang)", "L = mu0*mu_r*N^2*A/l",
        {"mu_r": "1", "N": "1", "A": "m^2", "l": "m"}, "H", K2, dict(mu_r=1.0, N=100.0, A=1e-4, l=0.1))
def induktivitaet_spule(mu_r, N, A, l):
    return MU0 * mu_r * N ** 2 * A / l


@formel("Induktivitaet Ringkernspule", "L = mu0*mu_r*N^2*A/(2 pi r_m)",
        {"mu_r": "1", "N": "1", "A": "m^2", "r_m": "m"}, "H", K2, dict(mu_r=2000.0, N=200.0, A=1e-4, r_m=0.02))
def induktivitaet_ringkern(mu_r, N, A, r_m):
    return MU0 * mu_r * N ** 2 * A / (2.0 * PI * r_m)


@formel("Induktivitaet aus AL-Wert", "L = AL * N^2", {"AL": "H", "N": "1"}, "H", K2, dict(AL=250e-9, N=40.0))
def induktivitaet_al(AL, N):
    return AL * N ** 2


def kapazitaet_parallel(*C):
    """C = C1 + C2 + ..."""
    return sum(np.asarray(c, dtype=float) for c in C)


def kapazitaet_reihe(*C):
    """1/C = 1/C1 + 1/C2 + ..."""
    return 1.0 / sum(1.0 / np.asarray(c, dtype=float) for c in C)


def induktivitaet_reihe(*L):
    """L = L1 + L2 + ... (ohne Kopplung)"""
    return sum(np.asarray(x, dtype=float) for x in L)


def induktivitaet_parallel(*L):
    """1/L = 1/L1 + 1/L2 + ... (ohne Kopplung)"""
    return 1.0 / sum(1.0 / np.asarray(x, dtype=float) for x in L)


@formel("Reihenschaltung C (2 Stueck)", "C = C1*C2/(C1+C2)", {"C1": "F", "C2": "F"}, "F", K2,
        dict(C1=10e-6, C2=40e-6))
def kapazitaet_reihe2(C1, C2):
    return C1 * C2 / (C1 + C2)


@formel("Gekoppelte Spulen (Reihe, gleichsinnig)", "L = L1 + L2 + 2 k sqrt(L1 L2)",
        {"L1": "H", "L2": "H", "k": "1"}, "H", K2, dict(L1=1e-3, L2=4e-3, k=0.5))
def induktivitaet_reihe_gekoppelt(L1, L2, k):
    return L1 + L2 + 2.0 * k * np.sqrt(L1 * L2)


@formel("Gegeninduktivitaet", "M = k * sqrt(L1*L2)", {"k": "1", "L1": "H", "L2": "H"}, "H", K2,
        dict(k=0.8, L1=1e-3, L2=4e-3))
def gegeninduktivitaet(k, L1, L2):
    return k * np.sqrt(L1 * L2)


@formel("Kondensatorladung", "Q = C * U", {"C": "F", "U": "V"}, "C", K2, dict(C=100e-6, U=10.0))
def ladung_kondensator(C, U):
    return C * U


@formel("Energie im Kondensator", "W = 1/2 C U^2", {"C": "F", "U": "V"}, "J", K2, dict(C=100e-6, U=10.0))
def energie_kondensator(C, U):
    return 0.5 * C * U ** 2


@formel("Energie in der Spule", "W = 1/2 L I^2", {"L": "H", "I": "A"}, "J", K2, dict(L=10e-3, I=2.0))
def energie_spule(L, I):
    return 0.5 * L * I ** 2


@formel("Kondensatorstrom", "i = C * dU/dt", {"C": "F", "dU": "V", "dt": "s"}, "A", K2,
        dict(C=100e-6, dU=5.0, dt=1e-3))
def strom_kondensator(C, dU, dt):
    return C * dU / dt


@formel("Spulenspannung", "u = L * dI/dt", {"L": "H", "dI": "A", "dt": "s"}, "V", K2,
        dict(L=10e-3, dI=2.0, dt=1e-3))
def spannung_spule(L, dI, dt):
    return L * dI / dt


@formel("Zeitkonstante RC", "tau = R * C", {"R": "ohm", "C": "F"}, "s", K2, dict(R=1000.0, C=1e-6))
def zeitkonstante_rc(R, C):
    return R * C


@formel("Zeitkonstante RL", "tau = L / R", {"L": "H", "R": "ohm"}, "s", K2, dict(L=10e-3, R=10.0))
def zeitkonstante_rl(L, R):
    return L / R


# ===========================================================================
# 3. ELEKTRISCHE UND MAGNETISCHE FELDER
# ===========================================================================
K3 = "Felder"


@formel("Coulomb-Kraft", "F = q1 q2 / (4 pi eps0 eps_r r^2)",
        {"q1": "C", "q2": "C", "r": "m", "eps_r": "1"}, "N", K3, dict(q1=1e-6, q2=2e-6, r=0.1, eps_r=1.0))
def coulomb_kraft(q1, q2, r, eps_r=1.0):
    return q1 * q2 / (4.0 * PI * EPS0 * eps_r * r ** 2)


@formel("Feldstaerke Punktladung", "E = Q / (4 pi eps0 eps_r r^2)", {"Q": "C", "r": "m", "eps_r": "1"}, "V/m", K3,
        dict(Q=1e-9, r=0.05, eps_r=1.0))
def feldstaerke_punktladung(Q, r, eps_r=1.0):
    return Q / (4.0 * PI * EPS0 * eps_r * r ** 2)


@formel("Potential Punktladung", "phi = Q / (4 pi eps0 eps_r r)", {"Q": "C", "r": "m", "eps_r": "1"}, "V", K3,
        dict(Q=1e-9, r=0.05, eps_r=1.0))
def potential_punktladung(Q, r, eps_r=1.0):
    return Q / (4.0 * PI * EPS0 * eps_r * r)


@formel("Feld im Plattenkondensator", "E = U / d", {"U": "V", "d": "m"}, "V/m", K3, dict(U=100.0, d=1e-3))
def feldstaerke_platten(U, d):
    return U / d


@formel("Elektrische Flussdichte", "D = eps0 eps_r E", {"eps_r": "1", "E": "V/m"}, "C/m^2",
        K3, dict(eps_r=4.0, E=1e5))
def verschiebungsdichte(eps_r, E):
    return EPS0 * eps_r * E


@formel("Energiedichte E-Feld", "w = 1/2 eps0 eps_r E^2", {"eps_r": "1", "E": "V/m"}, "J/m^3", K3,
        dict(eps_r=1.0, E=1e6))
def energiedichte_e(eps_r, E):
    return 0.5 * EPS0 * eps_r * E ** 2


@formel("H-Feld gerader Leiter", "H = I / (2 pi r)", {"I": "A", "r": "m"}, "A/m", K3, dict(I=10.0, r=0.05))
def h_feld_leiter(I, r):
    return I / (2.0 * PI * r)


@formel("H-Feld lange Spule", "H = N I / l", {"N": "1", "I": "A", "l": "m"}, "A/m", K3,
        dict(N=500.0, I=2.0, l=0.25))
def h_feld_spule(N, I, l):
    return N * I / l


@formel("Magnetische Flussdichte", "B = mu0 mu_r H", {"mu_r": "1", "H": "A/m"}, "T", K3, dict(mu_r=1.0, H=795.8))
def flussdichte(mu_r, H):
    return MU0 * mu_r * H


@formel("Magnetischer Fluss", "Phi = B * A", {"B": "T", "A": "m^2"}, "Wb", K3, dict(B=1.5, A=2e-4))
def fluss(B, A):
    return B * A


@formel("Energiedichte B-Feld", "w = B^2 / (2 mu0 mu_r)", {"B": "T", "mu_r": "1"}, "J/m^3", K3, dict(B=1.0, mu_r=1.0))
def energiedichte_b(B, mu_r=1.0):
    return B ** 2 / (2.0 * MU0 * mu_r)


@formel("Lorentzkraft (v senkrecht B)", "F = q v B", {"q": "C", "v": "m/s", "B": "T"}, "N", K3,
        dict(q=1.602176634e-19, v=1e6, B=1.0))
def lorentzkraft(q, v, B):
    return q * v * B


@formel("Kraft auf stromdurchflossenen Leiter", "F = B I l sin(alpha)",
        {"B": "T", "I": "A", "l": "m", "alpha": "1"}, "N", K3, dict(B=0.5, I=10.0, l=0.2, alpha=PI / 2))
def kraft_leiter(B, I, l, alpha=PI / 2):
    return B * I * l * np.sin(alpha)


@formel("Bewegungsinduktion", "u = B l v", {"B": "T", "l": "m", "v": "m/s"}, "V", K3, dict(B=0.8, l=0.5, v=10.0))
def induktionsspannung_bewegt(B, l, v):
    return B * l * v


@formel("Induktionsgesetz (Betrag)", "|u| = N dPhi/dt", {"N": "1", "dPhi": "Wb", "dt": "s"}, "V", K3,
        dict(N=200.0, dPhi=1e-3, dt=0.01))
def induktionsspannung(N, dPhi, dt):
    return N * dPhi / dt


@formel("Magnetischer Widerstand", "Rm = l / (mu0 mu_r A)", {"l": "m", "mu_r": "1", "A": "m^2"}, "1/H", K3,
        dict(l=0.2, mu_r=1000.0, A=4e-4))
def magn_widerstand(l, mu_r, A):
    return l / (MU0 * mu_r * A)


@formel("Durchflutung", "Theta = N * I", {"N": "1", "I": "A"}, "A", K3, dict(N=500.0, I=2.0))
def durchflutung(N, I):
    return N * I


@formel("Hopkinsonsches Gesetz", "Phi = Theta / Rm", {"Theta": "A", "Rm": "1/H"}, "Wb", K3, dict(Theta=1000.0, Rm=1e6))
def magn_fluss_kreis(Theta, Rm):
    return Theta / Rm


@formel("Hall-Spannung", "UH = I B / (n e d)", {"I": "A", "B": "T", "n": "m^-3", "d": "m"}, "V", K3,
        dict(I=0.01, B=0.5, n=1e22, d=1e-4))
def hall_spannung(I, B, n, d):
    return I * B / (n * E_ELEM * d)


@formel("Skintiefe", "delta = sqrt(rho / (pi f mu0 mu_r))", {"f": "Hz", "rho": "ohm*m", "mu_r": "1"}, "m", K3,
        dict(f=1e6, rho=1.72e-8, mu_r=1.0))
def skintiefe(f, rho, mu_r=1.0):
    return np.sqrt(rho / (PI * f * MU0 * mu_r))


@formel("Wellenwiderstand des Mediums", "Z = sqrt(mu0 mu_r / (eps0 eps_r))", {"eps_r": "1", "mu_r": "1"}, "ohm", K3,
        dict(eps_r=1.0, mu_r=1.0))
def wellenwiderstand_medium(eps_r=1.0, mu_r=1.0):
    return np.sqrt(MU0 * mu_r / (EPS0 * eps_r))


@formel("Phasengeschwindigkeit", "v = c / sqrt(eps_r mu_r)", {"eps_r": "1", "mu_r": "1"}, "m/s", K3,
        dict(eps_r=4.0, mu_r=1.0))
def phasengeschwindigkeit(eps_r=1.0, mu_r=1.0):
    return C0 / np.sqrt(eps_r * mu_r)


@formel("Wellenlaenge", "lambda = c / (f sqrt(eps_r))", {"f": "Hz", "eps_r": "1"}, "m", K3, dict(f=1e9, eps_r=1.0))
def wellenlaenge(f, eps_r=1.0):
    return C0 / (f * np.sqrt(eps_r))


@formel("Poynting-Vektor (Betrag)", "S = E * H", {"E": "V/m", "H": "A/m"}, "W/m^2", K3,
        dict(E=377.0, H=1.0))
def poynting(E, H):
    return E * H


# ===========================================================================
# 4. WECHSELSTROM
# ===========================================================================
K4 = "Wechselstrom"


@formel("Kreisfrequenz", "omega = 2 pi f", {"f": "Hz"}, "1/s", K4, dict(f=50.0))
def kreisfrequenz(f):
    return 2.0 * PI * f


@formel("Frequenz aus Periodendauer", "f = 1 / T", {"T": "s"}, "Hz", K4, dict(T=0.02))
def frequenz_aus_T(T):
    return 1.0 / T


@formel("Effektivwert Sinus", "U = Uhat / sqrt(2)", {"u_max": "V"}, "V", K4, dict(u_max=325.27))
def effektivwert_sinus(u_max):
    return u_max / np.sqrt(2.0)


@formel("Scheitelwert Sinus", "Uhat = sqrt(2) * U", {"u_eff": "V"}, "V", K4, dict(u_eff=230.0))
def scheitelwert_sinus(u_eff):
    return np.sqrt(2.0) * u_eff


@formel("Gleichrichtwert Sinus", "|u|mittel = 2 Uhat / pi", {"u_max": "V"}, "V", K4, dict(u_max=10.0))
def gleichrichtwert_sinus(u_max):
    return 2.0 * u_max / PI


@formel("Effektivwert Dreieck", "U = Uhat / sqrt(3)", {"u_max": "V"}, "V", K4, dict(u_max=10.0))
def effektivwert_dreieck(u_max):
    return u_max / np.sqrt(3.0)


@formel("Effektivwert numerisch", "U = sqrt(1/T * int u^2 dt)", {"y": "V", "t": "s"}, "V", K4,
        dict(y=[0.0, 1.0, 0.0, -1.0, 0.0], t=[0.0, 0.25, 0.5, 0.75, 1.0]))
def effektivwert_numerisch(y, t):
    """Trapezregel; ueber ganze Perioden mitteln."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    return np.sqrt(np.sum((y[1:] ** 2 + y[:-1] ** 2) * np.diff(t)) / 2.0 / (t[-1] - t[0]))


@formel("Induktiver Blindwiderstand", "XL = omega L", {"f": "Hz", "L": "H"}, "ohm", K4, dict(f=50.0, L=1.0))
def blindwiderstand_l(f, L):
    return 2.0 * PI * f * L


@formel("Kapazitiver Blindwiderstand (Betrag)", "|XC| = 1 / (omega C)", {"f": "Hz", "C": "F"}, "ohm", K4,
        dict(f=50.0, C=10e-6))
def blindwiderstand_c(f, C):
    return 1.0 / (2.0 * PI * f * C)


@formel("Impedanz Spule (komplex)", "Z = j omega L", {"f": "Hz", "L": "H"}, "ohm", K4, dict(f=50.0, L=0.1))
def z_spule(f, L):
    return 1j * 2.0 * PI * f * L


@formel("Impedanz Kondensator (komplex)", "Z = 1 / (j omega C)", {"f": "Hz", "C": "F"}, "ohm", K4,
        dict(f=50.0, C=10e-6))
def z_kondensator(f, C):
    return 1.0 / (1j * 2.0 * PI * f * C)


@formel("Impedanz Reihen-RLC", "Z = R + j(omega L - 1/(omega C))", {"f": "Hz", "R": "ohm", "L": "H", "C": "F"},
        "ohm", K4, dict(f=1000.0, R=10.0, L=1e-3, C=1e-6))
def z_reihe_rlc(f, R, L, C):
    w = 2.0 * PI * f
    return R + 1j * (w * L - 1.0 / (w * C))


@formel("Impedanz Parallel-RLC", "Z = 1 / (1/R + 1/(j omega L) + j omega C)",
        {"f": "Hz", "R": "ohm", "L": "H", "C": "F"}, "ohm", K4, dict(f=1000.0, R=1000.0, L=1e-3, C=1e-6))
def z_parallel_rlc(f, R, L, C):
    w = 2.0 * PI * f
    return 1.0 / (1.0 / R + 1.0 / (1j * w * L) + 1j * w * C)


@formel("Betrag Impedanz", "|Z| = sqrt(R^2 + X^2)", {"R": "ohm", "X": "ohm"}, "ohm", K4, dict(R=3.0, X=4.0))
def betrag_impedanz(R, X):
    return np.sqrt(R ** 2 + X ** 2)


@formel("Phasenwinkel Impedanz", "phi = atan2(X, R)", {"R": "ohm", "X": "ohm"}, "rad", K4, dict(R=3.0, X=4.0))
def phasenwinkel_impedanz(R, X):
    return np.arctan2(X, R)


@formel("Scheinleistung", "S = U I", {"U": "V", "I": "A"}, "VA", K4, dict(U=230.0, I=10.0))
def scheinleistung(U, I):
    return U * I


@formel("Wirkleistung", "P = U I cos(phi)", {"U": "V", "I": "A", "phi": "rad"}, "W", K4,
        dict(U=230.0, I=10.0, phi=PI / 3))
def wirkleistung(U, I, phi):
    return U * I * np.cos(phi)


@formel("Blindleistung", "Q = U I sin(phi)", {"U": "V", "I": "A", "phi": "rad"}, "var", K4,
        dict(U=230.0, I=10.0, phi=PI / 3))
def blindleistung(U, I, phi):
    return U * I * np.sin(phi)


@formel("Leistungsfaktor", "lambda = P / S", {"P": "W", "S": "VA"}, "1", K4, dict(P=1150.0, S=2300.0))
def leistungsfaktor(P, S):
    return P / S


@formel("Phasenwinkel aus P und Q", "phi = atan2(Q, P)", {"P": "W", "Q": "var"}, "rad", K4, dict(P=1000.0, Q=1000.0))
def phasenwinkel_pq(P, Q):
    return np.arctan2(Q, P)


@formel("Strom aus Wirkleistung (1~)", "I = P / (U cos(phi))", {"P": "W", "U": "V", "cos_phi": "1"}, "A", K4,
        dict(P=2300.0, U=230.0, cos_phi=0.8))
def strom_einphasig(P, U, cos_phi):
    return P / (U * cos_phi)


@formel("Blindleistungskompensation (Kapazitaet)", "C = P (tan phi1 - tan phi2) / (omega U^2)",
        {"P": "W", "cos_phi1": "1", "cos_phi2": "1", "f": "Hz", "U": "V"}, "F", K4,
        dict(P=10000.0, cos_phi1=0.7, cos_phi2=0.95, f=50.0, U=230.0))
def kompensation_kapazitaet(P, cos_phi1, cos_phi2, f, U):
    """C liegt an der Spannung U (einphasig bzw. je Strang)."""
    dQ = P * (np.tan(np.arccos(cos_phi1)) - np.tan(np.arccos(cos_phi2)))
    return dQ / (2.0 * PI * f * U ** 2)


@formel("Kompensations-Blindleistung", "Qc = P (tan phi1 - tan phi2)",
        {"P": "W", "cos_phi1": "1", "cos_phi2": "1"}, "var", K4, dict(P=10000.0, cos_phi1=0.7, cos_phi2=0.95))
def kompensation_blindleistung(P, cos_phi1, cos_phi2):
    return P * (np.tan(np.arccos(cos_phi1)) - np.tan(np.arccos(cos_phi2)))


@formel("Phasenverschiebung aus Zeitdifferenz", "phi = 2 pi f dt", {"f": "Hz", "dt": "s"}, "rad", K4,
        dict(f=50.0, dt=0.005))
def phase_aus_zeit(f, dt):
    return 2.0 * PI * f * dt


# ===========================================================================
# 5. DREHSTROM
# ===========================================================================
K5 = "Drehstrom"


@formel("Verkettete Spannung", "UL = sqrt(3) * Uph", {"U_ph": "V"}, "V", K5, dict(U_ph=230.0))
def spannung_verkettet(U_ph):
    return np.sqrt(3.0) * U_ph


@formel("Strangspannung", "Uph = UL / sqrt(3)", {"U_L": "V"}, "V", K5, dict(U_L=400.0))
def spannung_strang(U_L):
    return U_L / np.sqrt(3.0)


@formel("Scheinleistung Drehstrom", "S = sqrt(3) UL IL", {"U_L": "V", "I_L": "A"}, "VA", K5,
        dict(U_L=400.0, I_L=10.0))
def scheinleistung_drehstrom(U_L, I_L):
    return np.sqrt(3.0) * U_L * I_L


@formel("Wirkleistung Drehstrom", "P = sqrt(3) UL IL cos(phi)", {"U_L": "V", "I_L": "A", "cos_phi": "1"}, "W", K5,
        dict(U_L=400.0, I_L=10.0, cos_phi=0.8))
def wirkleistung_drehstrom(U_L, I_L, cos_phi):
    return np.sqrt(3.0) * U_L * I_L * cos_phi


@formel("Blindleistung Drehstrom", "Q = sqrt(3) UL IL sin(phi)", {"U_L": "V", "I_L": "A", "phi": "rad"}, "var", K5,
        dict(U_L=400.0, I_L=10.0, phi=0.6435))
def blindleistung_drehstrom(U_L, I_L, phi):
    return np.sqrt(3.0) * U_L * I_L * np.sin(phi)


@formel("Leiterstrom aus Wirkleistung", "IL = P / (sqrt(3) UL cos(phi))",
        {"P": "W", "U_L": "V", "cos_phi": "1"}, "A", K5, dict(P=5542.56, U_L=400.0, cos_phi=0.8))
def strom_drehstrom(P, U_L, cos_phi):
    return P / (np.sqrt(3.0) * U_L * cos_phi)


@formel("Leistung Dreieck vs. Stern (gleiche Netzspannung)", "P_delta = 3 P_stern", {"P_stern": "W"}, "W", K5,
        dict(P_stern=1000.0))
def leistung_dreieck_aus_stern(P_stern):
    return 3.0 * P_stern


@formel("Spannungsfall Drehstromleitung", "dU = sqrt(3) I (R cos(phi) + X sin(phi))",
        {"I": "A", "R": "ohm", "X": "ohm", "phi": "rad"}, "V", K5,
        dict(I=100.0, R=0.1, X=0.05, phi=float(np.arccos(0.8))))
def spannungsfall_drehstrom(I, R, X, phi):
    """R, X: Widerstand/Reaktanz eines Aussenleiters (gesamte Leitungslaenge)."""
    return np.sqrt(3.0) * I * (R * np.cos(phi) + X * np.sin(phi))


@formel("Kompensation Drehstrom (Dreieck-Kondensatoren)", "C = Qc / (3 omega UL^2)",
        {"Qc": "var", "f": "Hz", "U_L": "V"}, "F", K5, dict(Qc=10000.0, f=50.0, U_L=400.0))
def kompensation_drehstrom_dreieck(Qc, f, U_L):
    return Qc / (3.0 * 2.0 * PI * f * U_L ** 2)


@formel("Kompensation Drehstrom (Stern-Kondensatoren)", "C = Qc / (omega UL^2)",
        {"Qc": "var", "f": "Hz", "U_L": "V"}, "F", K5, dict(Qc=10000.0, f=50.0, U_L=400.0))
def kompensation_drehstrom_stern(Qc, f, U_L):
    """Je Strang liegt Uph = UL/sqrt(3): C = (Qc/3)/(omega Uph^2) = Qc/(omega UL^2)."""
    return Qc / (2.0 * PI * f * U_L ** 2)


# ===========================================================================
# 6. TRANSFORMATOR
# ===========================================================================
K6 = "Transformator"


@formel("Uebersetzung Spannung", "U2 = U1 * N2/N1", {"U1": "V", "N1": "1", "N2": "1"}, "V", K6,
        dict(U1=230.0, N1=1000.0, N2=100.0))
def trafo_u2(U1, N1, N2):
    return U1 * N2 / N1


@formel("Uebersetzung Strom", "I2 = I1 * N1/N2", {"I1": "A", "N1": "1", "N2": "1"}, "A", K6,
        dict(I1=1.0, N1=1000.0, N2=100.0))
def trafo_i2(I1, N1, N2):
    return I1 * N1 / N2


@formel("Impedanztransformation (auf Primaerseite)", "Z' = Z2 * (N1/N2)^2", {"Z2": "ohm", "N1": "1", "N2": "1"},
        "ohm", K6, dict(Z2=10.0, N1=1000.0, N2=100.0))
def trafo_z_primaer(Z2, N1, N2):
    return Z2 * (N1 / N2) ** 2


@formel("Transformator-Hauptgleichung", "U = 2 pi f N B A / sqrt(2)  (= 4,44 f N B A)",
        {"f": "Hz", "N": "1", "B": "T", "A": "m^2"}, "V", K6, dict(f=50.0, N=500.0, B=1.5, A=1e-3))
def trafo_hauptgleichung(f, N, B, A):
    return 2.0 * PI * f * N * B * A / np.sqrt(2.0)


@formel("Wirkungsgrad Transformator", "eta = P2 / (P2 + Pfe + Pcu)", {"P2": "W", "P_fe": "W", "P_cu": "W"}, "1", K6,
        dict(P2=10000.0, P_fe=100.0, P_cu=200.0))
def trafo_wirkungsgrad(P2, P_fe, P_cu):
    return P2 / (P2 + P_fe + P_cu)


@formel("Dauerkurzschlussstrom", "Ik = In / uk", {"I_n": "A", "u_k": "1"}, "A", K6, dict(I_n=100.0, u_k=0.04))
def trafo_kurzschlussstrom(I_n, u_k):
    return I_n / u_k


# ===========================================================================
# 7. SCHWINGKREISE
# ===========================================================================
K7 = "Schwingkreise"


@formel("Resonanzfrequenz (Thomson)", "f0 = 1 / (2 pi sqrt(L C))", {"L": "H", "C": "F"}, "Hz", K7,
        dict(L=1e-3, C=1e-6))
def resonanzfrequenz(L, C):
    return 1.0 / (2.0 * PI * np.sqrt(L * C))


@formel("Guete Reihenschwingkreis", "Q = (1/R) sqrt(L/C)", {"R": "ohm", "L": "H", "C": "F"}, "1", K7,
        dict(R=10.0, L=1e-3, C=1e-6))
def guete_reihe(R, L, C):
    return np.sqrt(L / C) / R


@formel("Guete Parallelschwingkreis", "Q = R sqrt(C/L)", {"R": "ohm", "L": "H", "C": "F"}, "1", K7,
        dict(R=1000.0, L=1e-3, C=1e-6))
def guete_parallel(R, L, C):
    return R * np.sqrt(C / L)


@formel("Bandbreite (-3 dB)", "B = f0 / Q", {"f0": "Hz", "Q": "1"}, "Hz", K7, dict(f0=5000.0, Q=10.0))
def bandbreite(f0, Q):
    return f0 / Q


@formel("Grenzfrequenzen Reihenkreis (exakt)", "f1,2 = f0 (sqrt(1+1/(4Q^2)) -+ 1/(2Q))",
        {"f0": "Hz", "Q": "1"}, ("Hz", "Hz"), K7, dict(f0=5000.0, Q=10.0))
def grenzfrequenzen_kreis(f0, Q):
    w = np.sqrt(1.0 + 1.0 / (4.0 * Q ** 2))
    return f0 * (w - 1.0 / (2.0 * Q)), f0 * (w + 1.0 / (2.0 * Q))


@formel("Daempfungsgrad Reihen-RLC", "zeta = (R/2) sqrt(C/L)", {"R": "ohm", "L": "H", "C": "F"}, "1", K7,
        dict(R=10.0, L=1e-3, C=1e-6))
def daempfungsgrad_rlc(R, L, C):
    return 0.5 * R * np.sqrt(C / L)


@formel("Gedaempfte Eigenkreisfrequenz", "omega_d = sqrt(1/(LC) - (R/2L)^2)",
        {"R": "ohm", "L": "H", "C": "F"}, "1/s", K7, dict(R=10.0, L=1e-3, C=1e-6))
def eigenkreisfrequenz_gedaempft(R, L, C):
    return np.sqrt(1.0 / (L * C) - (R / (2.0 * L)) ** 2 + 0j).real


@formel("Aperiodischer Grenzfall (Widerstand)", "R_ap = 2 sqrt(L/C)", {"L": "H", "C": "F"}, "ohm", K7,
        dict(L=1e-3, C=1e-6))
def widerstand_aperiodisch(L, C):
    return 2.0 * np.sqrt(L / C)


# ===========================================================================
# 8. SCHALTVORGAENGE
# ===========================================================================
K8 = "Schaltvorgaenge"


@formel("RC Aufladung: uC(t)", "uC = U0 (1 - exp(-t/RC))", {"t": "s", "U0": "V", "R": "ohm", "C": "F"}, "V", K8,
        dict(t=1e-3, U0=5.0, R=1000.0, C=1e-6))
def rc_ladespannung(t, U0, R, C):
    return U0 * (1.0 - np.exp(-t / (R * C)))


@formel("RC Entladung: uC(t)", "uC = U0 exp(-t/RC)", {"t": "s", "U0": "V", "R": "ohm", "C": "F"}, "V", K8,
        dict(t=1e-3, U0=5.0, R=1000.0, C=1e-6))
def rc_entladespannung(t, U0, R, C):
    return U0 * np.exp(-t / (R * C))


@formel("RC Ladestrom: i(t)", "i = U0/R exp(-t/RC)", {"t": "s", "U0": "V", "R": "ohm", "C": "F"}, "A", K8,
        dict(t=1e-3, U0=5.0, R=1000.0, C=1e-6))
def rc_ladestrom(t, U0, R, C):
    return U0 / R * np.exp(-t / (R * C))


@formel("RC Zeit bis Anteil x des Endwerts", "t = -RC ln(1-x)", {"R": "ohm", "C": "F", "x": "1"}, "s", K8,
        dict(R=1000.0, C=1e-6, x=0.5))
def rc_zeit_bis_anteil(R, C, x):
    return -R * C * np.log(1.0 - x)


@formel("RL Einschalten: i(t)", "i = U/R (1 - exp(-tR/L))", {"t": "s", "U": "V", "R": "ohm", "L": "H"}, "A", K8,
        dict(t=1e-3, U=10.0, R=10.0, L=10e-3))
def rl_einschaltstrom(t, U, R, L):
    return U / R * (1.0 - np.exp(-t * R / L))


@formel("RL Abschalten (Freilauf): i(t)", "i = I0 exp(-tR/L)", {"t": "s", "I0": "A", "R": "ohm", "L": "H"}, "A", K8,
        dict(t=1e-3, I0=1.0, R=10.0, L=10e-3))
def rl_freilaufstrom(t, I0, R, L):
    return I0 * np.exp(-t * R / L)


@formel("Reihen-RLC Sprungantwort uC(t)", "uC'' + (R/L) uC' + uC/(LC) = U0/(LC)",
        {"t": "s", "R": "ohm", "L": "H", "C": "F", "U0": "V"}, "V", K8,
        dict(t=1e-4, R=10.0, L=1e-3, C=1e-6, U0=1.0))
def rlc_sprung_uc(t, R, L, C, U0=1.0):
    """Kondensatorspannung nach Spannungssprung U0 (Anfangsbedingungen 0).
    Drei Faelle: schwingend (delta<w0), aperiodischer Grenzfall, kriechend."""
    t = np.asarray(t, dtype=float)
    delta = R / (2.0 * L)
    w0sq = 1.0 / (L * C)
    disc = delta ** 2 - w0sq
    if abs(disc) <= 1e-12 * w0sq:                         # aperiodischer Grenzfall
        return U0 * (1.0 - (1.0 + delta * t) * np.exp(-delta * t))
    if disc < 0.0:                                         # schwingfall
        wd = np.sqrt(-disc)
        return U0 * (1.0 - np.exp(-delta * t) * (np.cos(wd * t) + delta / wd * np.sin(wd * t)))
    a = np.sqrt(disc)                                      # kriechfall
    s1, s2 = -delta + a, -delta - a
    return U0 * (1.0 + (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s1 - s2))


@formel("Ueberschwingweite (PT2)", "e = exp(-pi zeta / sqrt(1 - zeta^2))", {"zeta": "1"}, "1", K8, dict(zeta=0.5))
def ueberschwingweite(zeta):
    z = np.asarray(zeta, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.exp(-PI * z / np.sqrt(1.0 - z ** 2))
    return np.where(z < 1.0, e, 0.0)


# ===========================================================================
# 9. FILTER UND UEBERTRAGUNGSFUNKTIONEN
# ===========================================================================
K9 = "Filter"


@formel("Grenzfrequenz RC-Filter", "fc = 1 / (2 pi R C)", {"R": "ohm", "C": "F"}, "Hz", K9, dict(R=1000.0, C=159.155e-9))
def grenzfrequenz_rc(R, C):
    return 1.0 / (2.0 * PI * R * C)


@formel("Grenzfrequenz RL-Filter", "fc = R / (2 pi L)", {"R": "ohm", "L": "H"}, "Hz", K9, dict(R=100.0, L=15.9e-3))
def grenzfrequenz_rl(R, L):
    return R / (2.0 * PI * L)


@formel("Tiefpass 1. Ordnung", "H = 1 / (1 + j f/fc)", {"f": "Hz", "fc": "Hz"}, "1", K9, dict(f=1000.0, fc=1000.0))
def tiefpass_1(f, fc):
    return 1.0 / (1.0 + 1j * f / fc)


@formel("Hochpass 1. Ordnung", "H = j(f/fc) / (1 + j f/fc)", {"f": "Hz", "fc": "Hz"}, "1", K9,
        dict(f=1000.0, fc=1000.0))
def hochpass_1(f, fc):
    x = 1j * f / fc
    return x / (1.0 + x)


@formel("Tiefpass 2. Ordnung", "H = 1 / (1 - (f/f0)^2 + j f/(f0 Q))", {"f": "Hz", "f0": "Hz", "Q": "1"}, "1", K9,
        dict(f=1000.0, f0=1000.0, Q=0.7071))
def tiefpass_2(f, f0, Q):
    x = f / f0
    return 1.0 / (1.0 - x ** 2 + 1j * x / Q)


@formel("Butterworth-Betrag", "|H| = 1 / sqrt(1 + (f/fc)^(2n))", {"f": "Hz", "fc": "Hz", "n": "1"}, "1", K9,
        dict(f=2000.0, fc=1000.0, n=3.0))
def butterworth_betrag(f, fc, n):
    return 1.0 / np.sqrt(1.0 + (f / fc) ** (2.0 * n))


@formel("Butterworth-Ordnung (Bedarf, reell)", "n >= lg[(10^(As/10)-1)/(10^(Ap/10)-1)] / (2 lg(fs/fp))",
        {"fp": "Hz", "fs": "Hz", "A_p_db": "1", "A_s_db": "1"}, "1", K9,
        dict(fp=1000.0, fs=2000.0, A_p_db=1.0, A_s_db=40.0))
def butterworth_ordnung(fp, fs, A_p_db, A_s_db):
    """Ergebnis aufrunden (math.ceil) fuer die noetige ganzzahlige Ordnung."""
    return np.log10((10 ** (A_s_db / 10) - 1) / (10 ** (A_p_db / 10) - 1)) / (2.0 * np.log10(fs / fp))


@formel("Flankensteilheit n-ter Ordnung", "S = 20 n dB/Dekade", {"n": "1"}, "1", K9, dict(n=2.0))
def steilheit_db_dekade(n):
    return 20.0 * n


@formel("Betrag in dB", "a = 20 lg|H|", {"H": "1"}, "1", K9, dict(H=0.7071))
def betrag_db(H):
    return 20.0 * np.log10(np.abs(H))


@formel("Phase in Grad", "phi = arg(H)", {"H": "1"}, "1", K9, dict(H=1 - 1j))
def phase_grad(H):
    return np.degrees(np.angle(H))


# ===========================================================================
# 10. VERSTAERKER, OPV, RAUSCHEN
# ===========================================================================
K10 = "Verstaerker"


@formel("OPV invertierend", "V = -Rf / Rin", {"Rf": "ohm", "Rin": "ohm"}, "1", K10, dict(Rf=10e3, Rin=1e3))
def opv_invertierend(Rf, Rin):
    return -Rf / Rin


@formel("OPV nichtinvertierend", "V = 1 + Rf / Rg", {"Rf": "ohm", "Rg": "ohm"}, "1", K10, dict(Rf=10e3, Rg=1e3))
def opv_nichtinvertierend(Rf, Rg):
    return 1.0 + Rf / Rg


@formel("OPV Bandbreite (GBW)", "fg = GBW / V", {"GBW": "Hz", "V": "1"}, "Hz", K10, dict(GBW=1e6, V=100.0))
def opv_grenzfrequenz(GBW, V):
    return GBW / np.abs(V)


@formel("OPV Slew-Rate-Grenze (Grossignal)", "f_max = SR / (2 pi Uhat)", {"SR": "V/s", "u_max": "V"}, "Hz", K10,
        dict(SR=1e6, u_max=10.0))
def opv_slew_fmax(SR, u_max):
    return SR / (2.0 * PI * u_max)


@formel("Thermisches Rauschen (Spannung)", "Un = sqrt(4 kB T R B)", {"T": "K", "R": "ohm", "B": "Hz"}, "V", K10,
        dict(T=300.0, R=1000.0, B=1.0))
def rauschspannung_thermisch(T, R, B):
    return np.sqrt(4.0 * K_B * T * R * B)


@formel("Schrotrauschen (Strom)", "In = sqrt(2 q I B)", {"I": "A", "B": "Hz"}, "A", K10, dict(I=1e-3, B=1.0))
def rauschstrom_schrot(I, B):
    return np.sqrt(2.0 * E_ELEM * I * B)


# ===========================================================================
# 11. HALBLEITER
# ===========================================================================
K11 = "Halbleiter"


@formel("Temperaturspannung", "UT = kB T / e", {"T": "K"}, "V", K11, dict(T=300.0))
def thermospannung(T):
    return K_B * T / E_ELEM


@formel("Shockley-Diodengleichung", "I = Is (exp(U/(n UT)) - 1)", {"U": "V", "I_s": "A", "n": "1", "T": "K"}, "A",
        K11, dict(U=0.6, I_s=1e-12, n=1.0, T=300.0))
def diode_strom(U, I_s, n=1.0, T=300.0):
    return I_s * np.expm1(U / (n * thermospannung(T)))


@formel("LED-Vorwiderstand", "R = (Uq - Uf) / I", {"Uq": "V", "Uf": "V", "I": "A"}, "ohm", K11,
        dict(Uq=5.0, Uf=2.0, I=0.02))
def led_vorwiderstand(Uq, Uf, I):
    return (Uq - Uf) / I


@formel("Brummspannung Siebkondensator", "dU = I / (f_r C)", {"I": "A", "f_r": "Hz", "C": "F"}, "V", K11,
        dict(I=0.1, f_r=100.0, C=1e-3))
def brummspannung(I, f_r, C):
    """f_r: Brummfrequenz (Einweg: f, Zweiweg: 2 f)."""
    return I / (f_r * C)


@formel("MOSFET Sattigungsstrom (Quadratmodell)", "ID = 1/2 k (UGS - Uth)^2",
        {"k": "A/V^2", "U_GS": "V", "U_th": "V"}, "A", K11, dict(k=2e-3, U_GS=3.0, U_th=1.0))
def mosfet_id_saettigung(k, U_GS, U_th):
    return 0.5 * k * np.maximum(U_GS - U_th, 0.0) ** 2


@formel("BJT Kollektorstrom", "IC = beta IB", {"beta": "1", "I_B": "A"}, "A", K11, dict(beta=100.0, I_B=20e-6))
def bjt_ic(beta, I_B):
    return beta * I_B


@formel("BJT Steilheit", "gm = IC / UT", {"I_C": "A", "T": "K"}, "S", K11, dict(I_C=1e-3, T=300.0))
def bjt_gm(I_C, T=300.0):
    return I_C / thermospannung(T)


# ===========================================================================
# 12. LEITUNGEN UND HOCHFREQUENZ
# ===========================================================================
K12 = "Leitungen"


@formel("Wellenwiderstand Koaxialkabel", "Z = sqrt(mu0/eps0)/(2 pi sqrt(eps_r)) ln(D/d)",
        {"eps_r": "1", "D": "m", "d": "m"}, "ohm", K12, dict(eps_r=2.25, D=3.5e-3, d=1e-3))
def wellenwiderstand_koax(eps_r, D, d):
    return np.sqrt(MU0 / EPS0) / (2.0 * PI * np.sqrt(eps_r)) * np.log(D / d)


@formel("Reflexionsfaktor", "r = (ZL - Z0)/(ZL + Z0)", {"Z_L": "ohm", "Z_0": "ohm"}, "1", K12, dict(Z_L=100.0, Z_0=50.0))
def reflexionsfaktor(Z_L, Z_0):
    return (Z_L - Z_0) / (Z_L + Z_0)


@formel("Stehwellenverhaeltnis (VSWR)", "s = (1+|r|)/(1-|r|)", {"r": "1"}, "1", K12, dict(r=1.0 / 3.0))
def stehwellenverhaeltnis(r):
    r = np.abs(r)
    with np.errstate(divide="ignore"):
        return (1.0 + r) / (1.0 - r)


@formel("Rueckflussdaempfung", "aR = -20 lg|r|", {"r": "1"}, "1", K12, dict(r=1.0 / 3.0))
def rueckflussdaempfung_db(r):
    with np.errstate(divide="ignore"):
        return -20.0 * np.log10(np.abs(r))


@formel("Freiraumdaempfung", "FSPL = 20 lg(4 pi d f / c)", {"d": "m", "f": "Hz"}, "1", K12, dict(d=1000.0, f=1e9))
def freiraumdaempfung_db(d, f):
    return 20.0 * np.log10(4.0 * PI * d * f / C0)


@formel("Leitungs-Eingangsimpedanz (verlustfrei)", "Zin = Z0 (ZL + j Z0 tan(bl)) / (Z0 + j ZL tan(bl))",
        {"Z_L": "ohm", "Z_0": "ohm", "beta_l": "1"}, "ohm", K12, dict(Z_L=100.0, Z_0=50.0, beta_l=PI / 4))
def leitung_eingangsimpedanz(Z_L, Z_0, beta_l):
    t = np.tan(beta_l)
    return Z_0 * (Z_L + 1j * Z_0 * t) / (Z_0 + 1j * Z_L * t)


# ===========================================================================
# 13. ELEKTRISCHE MASCHINEN
# ===========================================================================
K13 = "Maschinen"


@formel("Synchrondrehzahl", "ns = f / p", {"f": "Hz", "p": "1"}, "1/s", K13, dict(f=50.0, p=2.0))
def synchrondrehzahl(f, p):
    """Ergebnis in 1/s (x 60 = 1/min)."""
    return f / p


@formel("Schlupf", "s = (ns - n) / ns", {"n_s": "1/s", "n": "1/s"}, "1", K13, dict(n_s=25.0, n=24.0))
def schlupf(n_s, n):
    return (n_s - n) / n_s


@formel("Laeuferfrequenz", "f2 = s f1", {"s": "1", "f1": "Hz"}, "Hz", K13, dict(s=0.04, f1=50.0))
def laeuferfrequenz(s, f1):
    return s * f1


@formel("Drehmoment aus Leistung", "M = P / (2 pi n)", {"P": "W", "n": "1/s"}, "J", K13,
        dict(P=7500.0, n=24.0))
def drehmoment(P, n):
    return P / (2.0 * PI * n)


@formel("Mechanische Leistung", "P = 2 pi n M", {"n": "1/s", "M": "J"}, "W", K13, dict(n=24.0, M=49.74))
def leistung_mechanisch(n, M):
    return 2.0 * PI * n * M


@formel("Kloss'sche Formel (ASM)", "M = 2 Mk / (s/sk + sk/s)", {"M_k": "J", "s": "1", "s_k": "1"}, "J", K13,
        dict(M_k=100.0, s=0.1, s_k=0.2))
def kloss_moment(M_k, s, s_k):
    return 2.0 * M_k / (s / s_k + s_k / s)


@formel("Wirkungsgrad", "eta = P_ab / P_zu", {"P_ab": "W", "P_zu": "W"}, "1", K13, dict(P_ab=7500.0, P_zu=8500.0))
def wirkungsgrad(P_ab, P_zu):
    return P_ab / P_zu


@formel("Motorstrom Drehstrom", "I = P_ab / (sqrt(3) UL eta cos(phi))",
        {"P_ab": "W", "U_L": "V", "eta": "1", "cos_phi": "1"}, "A", K13,
        dict(P_ab=7500.0, U_L=400.0, eta=0.9, cos_phi=0.85))
def motor_strom_drehstrom(P_ab, U_L, eta, cos_phi):
    return P_ab / (np.sqrt(3.0) * U_L * eta * cos_phi)


@formel("GM Ankerspannungsgleichung", "U = k*omega + Ra*Ia", {"k": "V*s", "omega": "1/s", "R_a": "ohm", "I_a": "A"},
        "V", K13, dict(k=1.0, omega=100.0, R_a=0.5, I_a=10.0))
def dcm_ankerspannung(k, omega, R_a, I_a):
    return k * omega + R_a * I_a


@formel("GM Drehmoment", "M = k * Ia", {"k": "V*s", "I_a": "A"}, "J", K13, dict(k=1.0, I_a=10.0))
def dcm_drehmoment(k, I_a):
    return k * I_a


@formel("GM Winkelgeschwindigkeit", "omega = (U - Ra Ia) / k", {"U": "V", "R_a": "ohm", "I_a": "A", "k": "V*s"},
        "1/s", K13, dict(U=105.0, R_a=0.5, I_a=10.0, k=1.0))
def dcm_omega(U, R_a, I_a, k):
    return (U - R_a * I_a) / k


@formel("Hochlaufzeit", "t = J omega / M_b", {"J_tr": "kg*m^2", "omega": "1/s", "M_b": "J"}, "s", K13,
        dict(J_tr=0.05, omega=150.0, M_b=15.0))
def hochlaufzeit(J_tr, omega, M_b):
    return J_tr * omega / M_b


# ===========================================================================
# 14. EINHEITEN- UND GROESSENUMRECHNUNGEN
# ===========================================================================
K14 = "Einheiten"

# --- SI-Praefixe ----------------------------------------------------------
SI_PRAEFIX: Dict[str, float] = {
    "Q": 1e30, "R": 1e27, "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3,
    "h": 1e2, "da": 1e1, "": 1.0, "d": 1e-1, "c": 1e-2, "m": 1e-3, "µ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12,
    "f": 1e-15, "a": 1e-18, "z": 1e-21, "y": 1e-24, "r": 1e-27, "q": 1e-30,
}
_ENG_PRAEFIX = ["q", "r", "y", "z", "a", "f", "p", "n", "µ", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"]


def si_praefix(wert, von: str = "", nach: str = ""):
    """Wert von Praefix 'von' nach 'nach' umrechnen, z. B. si_praefix(4700, 'n', 'µ') -> 4.7"""
    return wert * SI_PRAEFIX[von] / SI_PRAEFIX[nach]


def auto_praefix(wert: float) -> Tuple[float, str]:
    """Ingenieurnotation: 0.00047 -> (470.0, 'µ')  (Exponent Vielfaches von 3)."""
    if wert == 0:
        return 0.0, ""
    exp3 = int(math.floor(math.log10(abs(wert)) / 3.0)) * 3
    exp3 = max(-30, min(30, exp3))
    praefix = _ENG_PRAEFIX[exp3 // 3 + 10]
    return wert / 10.0 ** exp3, praefix


def formatiere_si(wert: float, einheit: str = "", stellen: int = 4) -> str:
    """4700 -> '4.7 kΩ' (wenn einheit='Ω')."""
    m, p = auto_praefix(wert)
    return f"{m:.{stellen}g} {p}{einheit}".strip()


# --- Logarithmische Groessen ---------------------------------------------
@formel("Leistungsverhaeltnis -> dB", "a = 10 lg(P2/P1)", {"v": "1"}, "1", K14, dict(v=2.0), dimcheck=False)
def db_leistung(v):
    return 10.0 * np.log10(v)


@formel("Spannungs-/Stromverhaeltnis -> dB", "a = 20 lg(U2/U1)", {"v": "1"}, "1", K14, dict(v=10.0), dimcheck=False)
def db_spannung(v):
    return 20.0 * np.log10(v)


@formel("dB -> Leistungsverhaeltnis", "v = 10^(a/10)", {"a": "dB"}, "1", K14, dict(a=3.0103), dimcheck=False)
def von_db_leistung(a):
    return 10.0 ** (np.asarray(a) / 10.0)


@formel("dB -> Spannungsverhaeltnis", "v = 10^(a/20)", {"a": "dB"}, "1", K14, dict(a=20.0), dimcheck=False)
def von_db_spannung(a):
    return 10.0 ** (np.asarray(a) / 20.0)


@formel("W -> dBm", "L = 10 lg(P / 1 mW)", {"P": "W"}, "dBm", K14, dict(P=1.0), dimcheck=False)
def watt_zu_dbm(P):
    return 10.0 * np.log10(np.asarray(P) / 1e-3)


@formel("dBm -> W", "P = 1 mW * 10^(L/10)", {"L": "dBm"}, "W", K14, dict(L=30.0), dimcheck=False)
def dbm_zu_watt(L):
    return 1e-3 * 10.0 ** (np.asarray(L) / 10.0)


@formel("V -> dBV", "L = 20 lg(U / 1 V)", {"U": "V"}, "dBV", K14, dict(U=1.0), dimcheck=False)
def volt_zu_dbv(U):
    return 20.0 * np.log10(np.asarray(U))


@formel("dBV -> V", "U = 10^(L/20) V", {"L": "dBV"}, "V", K14, dict(L=-20.0), dimcheck=False)
def dbv_zu_volt(L):
    return 10.0 ** (np.asarray(L) / 20.0)


@formel("V -> dBuV", "L = 20 lg(U / 1 uV)", {"U": "V"}, "dBuV", K14, dict(U=1e-3), dimcheck=False)
def volt_zu_dbuv(U):
    return 20.0 * np.log10(np.asarray(U) / 1e-6)


@formel("dBuV -> V", "U = 1 uV * 10^(L/20)", {"L": "dBuV"}, "V", K14, dict(L=60.0), dimcheck=False)
def dbuv_zu_volt(L):
    return 1e-6 * 10.0 ** (np.asarray(L) / 20.0)


@formel("dBm -> dBuV an R", "dBuV = dBm + 90 + 10 lg(R/1 ohm)", {"dBm": "dBm", "R": "ohm"}, "dBuV", K14,
        dict(dBm=0.0, R=50.0), dimcheck=False)
def dbm_zu_dbuv(dBm, R=50.0):
    return np.asarray(dBm) + 90.0 + 10.0 * np.log10(R)


@formel("Neper -> dB", "1 Np = 20 lg(e) dB = 8,6859 dB", {"Np": "Np"}, "dB", K14, dict(Np=1.0), dimcheck=False)
def neper_zu_db(Np):
    return 20.0 * np.log10(np.e) * np.asarray(Np)


@formel("dB -> Neper", "1 dB = 0,11513 Np", {"dB": "dB"}, "Np", K14, dict(dB=8.685889638), dimcheck=False)
def db_zu_neper(dB):
    return np.asarray(dB) / (20.0 * np.log10(np.e))


# --- Energie / Leistung / Ladung -----------------------------------------
@formel("kWh -> J", "1 kWh = 3,6 MJ", {"kWh": "kWh"}, "J", K14, dict(kWh=1.0), dimcheck=False)
def kwh_zu_joule(kWh):
    return np.asarray(kWh) * 3.6e6


@formel("J -> kWh", "1 J = 2,7778e-7 kWh", {"J": "J"}, "kWh", K14, dict(J=3.6e6), dimcheck=False)
def joule_zu_kwh(J):
    return np.asarray(J) / 3.6e6


@formel("Wh -> J", "1 Wh = 3600 J", {"Wh": "Wh"}, "J", K14, dict(Wh=1.0), dimcheck=False)
def wh_zu_joule(Wh):
    return np.asarray(Wh) * 3600.0


@formel("eV -> J", "1 eV = 1,602176634e-19 J", {"eV": "eV"}, "J", K14, dict(eV=1.0), dimcheck=False)
def ev_zu_joule(eV):
    return np.asarray(eV) * E_ELEM


@formel("J -> eV", "1 J = 6,2415e18 eV", {"J": "J"}, "eV", K14, dict(J=1.602176634e-19), dimcheck=False)
def joule_zu_ev(J):
    return np.asarray(J) / E_ELEM


@formel("cal -> J (thermochemisch)", "1 cal = 4,184 J", {"cal": "cal"}, "J", K14, dict(cal=1.0), dimcheck=False)
def cal_zu_joule(cal):
    return np.asarray(cal) * 4.184


@formel("Ah -> C", "1 Ah = 3600 C", {"Ah": "Ah"}, "C", K14, dict(Ah=1.0), dimcheck=False)
def ah_zu_coulomb(Ah):
    return np.asarray(Ah) * 3600.0


@formel("C -> Ah", "1 C = 2,7778e-4 Ah", {"C": "C"}, "Ah", K14, dict(C=3600.0), dimcheck=False)
def coulomb_zu_ah(C):
    return np.asarray(C) / 3600.0


@formel("PS (metrisch) -> W", "1 PS = 735,49875 W", {"PS": "PS"}, "W", K14, dict(PS=1.0), dimcheck=False)
def ps_zu_watt(PS):
    return np.asarray(PS) * 735.49875


@formel("hp (mechanisch) -> W", "1 hp = 745,69987 W", {"hp": "hp"}, "W", K14, dict(hp=1.0), dimcheck=False)
def hp_zu_watt(hp):
    return np.asarray(hp) * 745.699871582


@formel("W -> PS", "1 W = 1,35962e-3 PS", {"W": "W"}, "PS", K14, dict(W=735.49875), dimcheck=False)
def watt_zu_ps(W):
    return np.asarray(W) / 735.49875


# --- Temperatur -----------------------------------------------------------
@formel("Celsius -> Kelvin", "T = theta + 273,15", {"theta": "degC"}, "K", K14, dict(theta=25.0), dimcheck=False)
def celsius_zu_kelvin(theta):
    return np.asarray(theta) + T_NULL_C


@formel("Kelvin -> Celsius", "theta = T - 273,15", {"T": "K"}, "degC", K14, dict(T=300.0), dimcheck=False)
def kelvin_zu_celsius(T):
    return np.asarray(T) - T_NULL_C


@formel("Celsius -> Fahrenheit", "F = 9/5 theta + 32", {"theta": "degC"}, "degF", K14, dict(theta=100.0), dimcheck=False)
def celsius_zu_fahrenheit(theta):
    return np.asarray(theta) * 9.0 / 5.0 + 32.0


@formel("Fahrenheit -> Celsius", "theta = 5/9 (F - 32)", {"F": "degF"}, "degC", K14, dict(F=212.0), dimcheck=False)
def fahrenheit_zu_celsius(F):
    return (np.asarray(F) - 32.0) * 5.0 / 9.0


# --- Winkel / Frequenz / Drehzahl ----------------------------------------
@formel("Grad -> Radiant", "rad = deg * pi/180", {"deg": "deg"}, "rad", K14, dict(deg=180.0), dimcheck=False)
def grad_zu_rad(deg):
    return np.radians(deg)


@formel("Radiant -> Grad", "deg = rad * 180/pi", {"rad": "rad"}, "deg", K14, dict(rad=PI), dimcheck=False)
def rad_zu_grad(rad):
    return np.degrees(rad)


@formel("Hz -> rad/s", "omega = 2 pi f", {"f": "Hz"}, "rad/s", K14, dict(f=50.0), dimcheck=False)
def hz_zu_rads(f):
    return 2.0 * PI * np.asarray(f)


@formel("rad/s -> Hz", "f = omega / (2 pi)", {"omega": "rad/s"}, "Hz", K14, dict(omega=314.159265), dimcheck=False)
def rads_zu_hz(omega):
    return np.asarray(omega) / (2.0 * PI)


@formel("1/min -> rad/s", "omega = 2 pi n / 60", {"n": "1/min"}, "rad/s", K14, dict(n=60.0), dimcheck=False)
def rpm_zu_rads(n):
    return 2.0 * PI * np.asarray(n) / 60.0


@formel("rad/s -> 1/min", "n = 60 omega / (2 pi)", {"omega": "rad/s"}, "1/min", K14, dict(omega=2 * PI), dimcheck=False)
def rads_zu_rpm(omega):
    return 60.0 * np.asarray(omega) / (2.0 * PI)


# --- Laenge / Querschnitt / Leiter ---------------------------------------
@formel("inch -> m", "1 in = 25,4 mm", {"inch": "in"}, "m", K14, dict(inch=1.0), dimcheck=False)
def inch_zu_meter(inch):
    return np.asarray(inch) * 0.0254


@formel("mil -> m", "1 mil = 0,0254 mm", {"mil": "mil"}, "m", K14, dict(mil=1.0), dimcheck=False)
def mil_zu_meter(mil):
    return np.asarray(mil) * 2.54e-5


@formel("circular mil -> m^2", "1 cmil = pi/4 mil^2 = 5,0671e-4 mm^2", {"cmil": "cmil"}, "m^2", K14, dict(cmil=1.0),
        dimcheck=False)
def cmil_zu_m2(cmil):
    return np.asarray(cmil) * PI / 4.0 * (2.54e-5) ** 2


@formel("AWG -> Drahtdurchmesser", "d = 0,127 mm * 92^((36-n)/39)", {"n": "AWG"}, "m", K14, dict(n=10), dimcheck=False)
def awg_durchmesser(n):
    """n: AWG-Nummer; 00 = -1, 000 = -2, 0000 = -3."""
    return 0.127e-3 * 92.0 ** ((36.0 - np.asarray(n, dtype=float)) / 39.0)


@formel("AWG -> Querschnitt", "A = pi/4 d^2", {"n": "AWG"}, "m^2", K14, dict(n=10), dimcheck=False)
def awg_querschnitt(n):
    return PI / 4.0 * awg_durchmesser(n) ** 2


@formel("Querschnitt -> Durchmesser", "d = sqrt(4 A / pi)", {"A": "m^2"}, "m", K14, dict(A=1.5e-6), dimcheck=False)
def durchmesser_aus_querschnitt(A):
    return np.sqrt(4.0 * np.asarray(A) / PI)


# --- Leitfaehigkeit / spez. Widerstand -----------------------------------
@formel("rho [ohm mm^2/m] -> [ohm m]", "1 ohm mm^2/m = 1e-6 ohm m", {"rho": "ohm mm^2/m"}, "ohm m", K14,
        dict(rho=0.0172), dimcheck=False)
def rho_mm2_zu_si(rho):
    return np.asarray(rho) * 1e-6


@formel("rho [ohm m] -> [ohm mm^2/m]", "1 ohm m = 1e6 ohm mm^2/m", {"rho": "ohm m"}, "ohm mm^2/m", K14,
        dict(rho=1.72e-8), dimcheck=False)
def rho_si_zu_mm2(rho):
    return np.asarray(rho) * 1e6


@formel("kappa [m/(ohm mm^2)] <-> [S/m]", "1 m/(ohm mm^2) = 1e6 S/m", {"kappa": "m/(ohm mm^2)"}, "S/m", K14,
        dict(kappa=58.0), dimcheck=False)
def kappa_mm2_zu_si(kappa):
    return np.asarray(kappa) * 1e6


@formel("S/m -> % IACS", "100 % IACS = 5,8e7 S/m", {"sigma": "S/m"}, "%IACS", K14, dict(sigma=5.8e7), dimcheck=False)
def sigma_zu_iacs(sigma):
    return np.asarray(sigma) / 5.8e7 * 100.0


# --- Magnetische Einheiten (CGS <-> SI) ----------------------------------
@formel("Gauss -> Tesla", "1 G = 1e-4 T", {"G": "G"}, "T", K14, dict(G=10000.0), dimcheck=False)
def gauss_zu_tesla(G):
    return np.asarray(G) * 1e-4


@formel("Tesla -> Gauss", "1 T = 1e4 G", {"T": "T"}, "G", K14, dict(T=1.0), dimcheck=False)
def tesla_zu_gauss(T):
    return np.asarray(T) * 1e4


@formel("Oersted -> A/m", "1 Oe = 1000/(4 pi) A/m = 79,577 A/m", {"Oe": "Oe"}, "A/m", K14, dict(Oe=1.0),
        dimcheck=False)
def oersted_zu_a_pro_m(Oe):
    return np.asarray(Oe) * 1000.0 / (4.0 * PI)


@formel("A/m -> Oersted", "1 A/m = 4 pi/1000 Oe", {"H": "A/m"}, "Oe", K14, dict(H=79.577), dimcheck=False)
def a_pro_m_zu_oersted(H):
    return np.asarray(H) * 4.0 * PI / 1000.0


@formel("Maxwell -> Weber", "1 Mx = 1e-8 Wb", {"Mx": "Mx"}, "Wb", K14, dict(Mx=1e8), dimcheck=False)
def maxwell_zu_weber(Mx):
    return np.asarray(Mx) * 1e-8


@formel("Gilbert -> Ampere", "1 Gb = 10/(4 pi) A", {"Gb": "Gb"}, "A", K14, dict(Gb=1.0), dimcheck=False)
def gilbert_zu_ampere(Gb):
    return np.asarray(Gb) * 10.0 / (4.0 * PI)


# --- E-Reihen und Widerstands-Farbcode -----------------------------------
E_REIHEN: Dict[str, Tuple[float, ...]] = {
    "E3": (1.0, 2.2, 4.7),
    "E6": (1.0, 1.5, 2.2, 3.3, 4.7, 6.8),
    "E12": (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2),
    "E24": (1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2,
            6.8, 7.5, 8.2, 9.1),
}


def naechster_e_wert(wert: float, reihe: str = "E12") -> float:
    """Naechster Normwert der E-Reihe (logarithmisch naechster Nachbar)."""
    dekade = 10.0 ** math.floor(math.log10(wert))
    kandidaten = [v * dekade * m for v in E_REIHEN[reihe] for m in (1.0, 10.0)] + [E_REIHEN[reihe][0] * dekade / 10.0]
    return min(kandidaten, key=lambda k: abs(math.log(k / wert)))


_FARBEN_ZIFFER = {"schwarz": 0, "braun": 1, "rot": 2, "orange": 3, "gelb": 4, "gruen": 5, "blau": 6, "violett": 7,
                  "grau": 8, "weiss": 9}
_FARBEN_MULT = {**{k: 10.0 ** v for k, v in _FARBEN_ZIFFER.items()}, "gold": 0.1, "silber": 0.01}
_FARBEN_TOL = {"braun": 1.0, "rot": 2.0, "gruen": 0.5, "blau": 0.25, "violett": 0.1, "grau": 0.05, "gold": 5.0,
               "silber": 10.0}


def widerstand_farbcode(*farben: str) -> Tuple[float, float]:
    """4-Band (Z Z Mult Tol) oder 5-Band (Z Z Z Mult Tol) -> (Ohm, Toleranz in %)."""
    f = [x.lower() for x in farben]
    if len(f) == 4:
        wert = (10 * _FARBEN_ZIFFER[f[0]] + _FARBEN_ZIFFER[f[1]]) * _FARBEN_MULT[f[2]]
    elif len(f) == 5:
        wert = (100 * _FARBEN_ZIFFER[f[0]] + 10 * _FARBEN_ZIFFER[f[1]] + _FARBEN_ZIFFER[f[2]]) * _FARBEN_MULT[f[3]]
    else:
        raise ValueError("4 oder 5 Farbbaender erwartet")
    return wert, _FARBEN_TOL[f[-1]]


# ===========================================================================
# Katalog
# ===========================================================================
def katalog(kategorie: str | None = None) -> None:
    """Gibt alle registrierten Formeln (optional einer Kategorie) tabellarisch aus."""
    aktuell = None
    for name, fn in REGISTRY.items():
        m = fn.meta
        if kategorie and m["kat"].lower() != kategorie.lower():
            continue
        if m["kat"] != aktuell:
            aktuell = m["kat"]
            print(f"\n=== {aktuell.upper()} ===")
        ein = ", ".join(f"{k} [{u}]" for k, u in m["ein"].items())
        aus = m["aus"] if isinstance(m["aus"], str) else ", ".join(m["aus"])
        print(f"{name:34s} {m['gleichung']:60s} ({ein}) -> [{aus}]")


def kategorien() -> Tuple[str, ...]:
    return tuple(dict.fromkeys(fn.meta["kat"] for fn in REGISTRY.values()))


if __name__ == "__main__":
    katalog()
    print(f"\n{len(REGISTRY)} registrierte Formeln, Kategorien: {', '.join(kategorien())}")
    print("Dimensionspruefung:", "bestanden" if pruefe_alle_dimensionen() else "FEHLER")
