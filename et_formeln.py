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
* Kategorien: Gleichstrom | Spannungsquellen | Bauelemente | Felder | Wechselstrom | Drehstrom |
              Transformator | Schwingkreise | Schaltvorgaenge | Filter | Verstaerker | Halbleiter |
              Leitungen | Maschinen | Einheiten | Nichtlineare Widerstaende | Messtechnik |
              Leistungselektronik | Netzgeraete | Kippschaltungen | Mechanik | Waerme | Geometrie |
              Datenuebertragung
* Ohne Einheit (nicht in der Registry): Digitaltechnik (Gatter, Zahlensysteme, Dualarithmetik,
  Schaltalgebra) und Mathematik (Prozent, Runden, Trigonometrie, Ableitung/Integral, Vektoren, umstellen)
* Abgleich mit der Formelsammlung von Elektronik Kompendium: ek_abdeckung.py

Qualitaetssicherung
-------------------
1. ``pruefe_alle_dimensionen()``  - Dimensionshomogenitaet (Skalierungstest je SI-Basiseinheit)
2. ``test_et_formeln.py``         - Referenzwerte, physikalische Identitaeten, numerische Gegenrechnung
3. ``test_ek_abdeckung.py``       - EK-Abdeckung (343 Seiten) + Referenzwerte der Ergaenzungen

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
FARADAY = 96485.33212         # C/mol    Faraday-Konstante (N_A e)
G_ERDE = 9.80665              # m/s^2    Normfallbeschleunigung
GRAV = 6.67430e-11            # m^3/(kg s^2) Gravitationskonstante
SIGMA_SB = 5.670374419e-8     # W/(m^2 K^4) Stefan-Boltzmann-Konstante

KONST_DIM: Dict[str, str] = {
    "C0": "m/s",
    "MU0": "H/m",
    "EPS0": "F/m",
    "E_ELEM": "C",
    "K_B": "J/K",
    "H_PLANCK": "J*s",
    "FARADAY": "C",           # pro mol - kuerzt sich gegen molare Massen in kg/mol
    "G_ERDE": "m/s^2",
    "GRAV": "m^3/kg/s^2",
    "SIGMA_SB": "W/m^2/K^4",
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


@formel("Elektrische Arbeit aus U, I, t", "W = U * I * t", {"U": "V", "I": "A", "t": "s"}, "J", K1,
        dict(U=230.0, I=2.0, t=3600.0))
def arbeit_uit(U, I, t):
    return U * I * t


@formel("Elektrische Arbeit aus I, R, t", "W = I^2 * R * t", {"I": "A", "R": "ohm", "t": "s"}, "J", K1,
        dict(I=2.0, R=10.0, t=60.0))
def arbeit_i2rt(I, R, t):
    return I ** 2 * R * t


@formel("Elektrische Arbeit aus Ladung", "W = Q * U", {"Q": "C", "U": "V"}, "J", K1, dict(Q=10.0, U=12.0))
def arbeit_qu(Q, U):
    return Q * U


@formel("Leitwert eines Leiters", "G = kappa * A / l", {"kappa": "S/m", "A": "m^2", "l": "m"}, "S", K1,
        dict(kappa=5.8e7, A=1.5e-6, l=100.0))
def leitwert_leiter(kappa, A, l):
    return kappa * A / l


@formel("Stromdichte aus Feldstaerke", "J = kappa * E", {"kappa": "S/m", "E": "V/m"}, "A/m^2", K1,
        dict(kappa=5.8e7, E=0.1))
def stromdichte_feld(kappa, E):
    return kappa * E


@formel("Temperaturkoeffizient aus Messung", "alpha = (R_T - R0) / (R0 * dT)", {"R0": "ohm", "R_T": "ohm", "dT": "K"},
        "1/K", K1, dict(R0=100.0, R_T=138.5, dT=100.0))
def temperaturkoeffizient(R0, R_T, dT):
    return (R_T - R0) / (R0 * dT)


@formel("Belasteter Spannungsteiler", "U_L = U * (R2||RL) / (R1 + R2||RL)",
        {"U": "V", "R1": "ohm", "R2": "ohm", "R_L": "ohm"}, "V", K1, dict(U=12.0, R1=1000.0, R2=1000.0, R_L=1000.0))
def spannungsteiler_belastet(U, R1, R2, R_L):
    r2l = R2 * R_L / (R2 + R_L)
    return U * r2l / (R1 + r2l)


@formel("Brueckenspannung (unabgeglichen)", "Ud = U (R2/(R1+R2) - R4/(R3+R4))",
        {"U": "V", "R1": "ohm", "R2": "ohm", "R3": "ohm", "R4": "ohm"}, "V", K1,
        dict(U=10.0, R1=1000.0, R2=1100.0, R3=1000.0, R4=1000.0))
def brueckenspannung(U, R1, R2, R3, R4):
    """Linker Zweig R1 (oben)/R2 (unten), rechter Zweig R3 (oben)/R4 (unten).
    Ud = Potential Mitte links minus Mitte rechts (Leerlauf, ohne Brueckenzweig)."""
    return U * (R2 / (R1 + R2) - R4 / (R3 + R4))


@formel("Bruecke: Innenwiderstand der Diagonale", "Ri = R1||R2 + R3||R4",
        {"R1": "ohm", "R2": "ohm", "R3": "ohm", "R4": "ohm"}, "ohm", K1,
        dict(R1=1000.0, R2=1100.0, R3=1000.0, R4=1000.0))
def bruecke_innenwiderstand(R1, R2, R3, R4):
    return R1 * R2 / (R1 + R2) + R3 * R4 / (R3 + R4)


@formel("Bruecke: Strom im Brueckenzweig", "I_Br = Ud / (Ri + R_Br)",
        {"U": "V", "R1": "ohm", "R2": "ohm", "R3": "ohm", "R4": "ohm", "R_Br": "ohm"}, "A", K1,
        dict(U=10.0, R1=1000.0, R2=1100.0, R3=1000.0, R4=1000.0, R_Br=500.0))
def bruecke_strom(U, R1, R2, R3, R4, R_Br):
    """Ersatzspannungsquelle der Diagonale (Ud, Ri) belastet mit R_Br."""
    return brueckenspannung(U, R1, R2, R3, R4) / (bruecke_innenwiderstand(R1, R2, R3, R4) + R_Br)


@formel("Bruecke: Gesamtwiderstand (ohne Brueckenzweig)", "R = (R1+R2)||(R3+R4)",
        {"R1": "ohm", "R2": "ohm", "R3": "ohm", "R4": "ohm"}, "ohm", K1,
        dict(R1=1000.0, R2=1100.0, R3=1000.0, R4=1000.0))
def bruecke_gesamtwiderstand(R1, R2, R3, R4):
    return widerstand_parallel2(R1 + R2, R3 + R4)


# ===========================================================================
# 1b. SPANNUNGSQUELLEN, AKKUS, FOTOVOLTAIK, GALVANIK
# ===========================================================================
K1B = "Spannungsquellen"


@formel("Gruppenschaltung gleicher Quellen", "U = n U0 ; Ri = n Ri0 / m  (n in Reihe, m parallel)",
        {"U0": "V", "Ri0": "ohm", "n": "1", "m": "1"}, ("V", "ohm"), K1B, dict(U0=1.5, Ri0=0.2, n=4.0, m=2.0))
def quellen_gruppenschaltung(U0, Ri0, n, m):
    """Reihenschaltung: m = 1; Parallelschaltung: n = 1."""
    return n * U0, n * Ri0 / m


@formel("Laststrom Gruppenschaltung", "I = n U0 / (n Ri0 / m + RL)",
        {"U0": "V", "Ri0": "ohm", "n": "1", "m": "1", "R_L": "ohm"}, "A", K1B,
        dict(U0=1.5, Ri0=0.2, n=4.0, m=2.0, R_L=10.0))
def strom_gruppenschaltung(U0, Ri0, n, m, R_L):
    U, Ri = quellen_gruppenschaltung(U0, Ri0, n, m)
    return U / (Ri + R_L)


@formel("Kurzschlussstrom (Ersatzstromquelle)", "Ik = U0 / Ri", {"U0": "V", "Ri": "ohm"}, "A", K1B,
        dict(U0=12.0, Ri=0.5))
def kurzschlussstrom(U0, Ri):
    return U0 / Ri


@formel("Innenwiderstand aus Leerlauf/Belastung", "Ri = (U0 - U) / I", {"U0": "V", "U": "V", "I": "A"}, "ohm", K1B,
        dict(U0=12.6, U=12.0, I=10.0))
def innenwiderstand(U0, U, I):
    return (U0 - U) / I


@formel("Innenwiderstand aus Leerlauf/Kurzschluss", "Ri = U0 / Ik", {"U0": "V", "I_k": "A"}, "ohm", K1B,
        dict(U0=12.0, I_k=24.0))
def innenwiderstand_lk(U0, I_k):
    return U0 / I_k


@formel("Laststrom an realer Quelle", "I = U0 / (Ri + RL)", {"U0": "V", "Ri": "ohm", "R_L": "ohm"}, "A", K1B,
        dict(U0=12.0, Ri=0.5, R_L=5.5))
def laststrom_quelle(U0, Ri, R_L):
    return U0 / (Ri + R_L)


@formel("Ladezeit Akku (mit Ladefaktor)", "t = k * Q / I", {"Q": "C", "I": "A", "k": "1"}, "s", K1B,
        dict(Q=7200.0, I=0.2, k=1.4))
def ladezeit_akku(Q, I, k):
    """k ~ 1,2...1,4 (NiMH/Blei), ~1,0 (Li-Ion)."""
    return k * Q / I


@formel("Ladungswirkungsgrad (Ah)", "eta_Ah = Q_ent / Q_lad", {"Q_ent": "C", "Q_lad": "C"}, "1", K1B,
        dict(Q_ent=7200.0, Q_lad=8640.0))
def wirkungsgrad_ah(Q_ent, Q_lad):
    return Q_ent / Q_lad


@formel("Energiewirkungsgrad (Wh)", "eta_Wh = W_ent / W_lad", {"W_ent": "J", "W_lad": "J"}, "1", K1B,
        dict(W_ent=86400.0, W_lad=120000.0))
def wirkungsgrad_wh(W_ent, W_lad):
    return W_ent / W_lad


@formel("Nernst-Gleichung", "E = E0 - (UT/z) ln(Q)", {"E0": "V", "T": "K", "z": "1", "Q_r": "1"}, "V", K1B,
        dict(E0=1.10, T=298.15, z=2.0, Q_r=0.1))
def nernst(E0, T, z, Q_r):
    """RT/(zF) = kB T/(z e) = UT/z.  Q_r: Reaktionsquotient (25 °C: 0,059 V/z * lg Q)."""
    return E0 - K_B * T / (z * E_ELEM) * np.log(Q_r)


@formel("Zellspannung galvanisches Element", "E = E_Kathode - E_Anode", {"E_K": "V", "E_A": "V"}, "V", K1B,
        dict(E_K=0.34, E_A=-0.76))
def zellspannung(E_K, E_A):
    return E_K - E_A


@formel("Faradaysches Gesetz (Elektrolyse)", "m = Q M / (z F)", {"Q": "C", "M": "kg", "z": "1"}, "kg", K1B,
        dict(Q=96485.0, M=63.546e-3, z=2.0))
def elektrolyse_masse(Q, M, z):
    """M: molare Masse in kg/mol (das 'pro mol' kuerzt sich gegen F in C/mol)."""
    return Q * M / (z * FARADAY)


@formel("Einstrahlungsleistung PV", "P_in = E * A", {"E": "W/m^2", "A": "m^2"}, "W", K1B, dict(E=1000.0, A=1.6))
def solar_einstrahlung(E, A):
    return E * A


@formel("Wirkungsgrad Solarmodul", "eta = P_max / (E A)", {"P_max": "W", "E": "W/m^2", "A": "m^2"}, "1", K1B,
        dict(P_max=320.0, E=1000.0, A=1.6))
def solar_wirkungsgrad(P_max, E, A):
    return P_max / (E * A)


@formel("Fuellfaktor Solarzelle", "FF = P_mpp / (U_oc I_sc)", {"P_mpp": "W", "U_oc": "V", "I_sc": "A"}, "1", K1B,
        dict(P_mpp=4.0, U_oc=0.65, I_sc=7.5))
def fuellfaktor(P_mpp, U_oc, I_sc):
    return P_mpp / (U_oc * I_sc)


@formel("Leerlaufspannung Solarzelle", "U_oc = n UT ln(I_ph/I_0 + 1)", {"I_ph": "A", "I_0": "A", "n": "1", "T": "K"},
        "V", K1B, dict(I_ph=7.5, I_0=1e-10, n=1.0, T=298.15))
def solar_leerlaufspannung(I_ph, I_0, n, T):
    return n * thermospannung(T) * np.log(I_ph / I_0 + 1.0)


@formel("Modulleistung bei Temperatur", "P = P_STC (1 + gamma (T - T_STC))",
        {"P_stc": "W", "gamma": "1/K", "T": "K", "T_stc": "K"}, "W", K1B,
        dict(P_stc=320.0, gamma=-0.0035, T=333.15, T_stc=298.15))
def solar_leistung_temperatur(P_stc, gamma, T, T_stc):
    return P_stc * (1.0 + gamma * (T - T_stc))


@formel("Jahresertrag PV", "E = P_peak * t_voll * PR", {"P_peak": "W", "t_voll": "s", "PR": "1"}, "J", K1B,
        dict(P_peak=10e3, t_voll=950.0 * 3600.0, PR=0.85))
def solar_jahresertrag(P_peak, t_voll, PR):
    """t_voll: spezifische Volllaststunden (kWh/kWp) als Zeit; PR: Performance Ratio."""
    return P_peak * t_voll * PR


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


@formel("Energie im Kondensator (aus Ladung)", "W = Q^2 / (2 C)", {"Q": "C", "C": "F"}, "J", K2,
        dict(Q=1e-3, C=100e-6))
def energie_kondensator_q(Q, C):
    return Q ** 2 / (2.0 * C)


@formel("Teilspannung C in Reihe", "U_i = Q / C_i", {"Q": "C", "C_i": "F"}, "V", K2, dict(Q=1e-4, C_i=10e-6))
def teilspannung_kondensator(Q, C_i):
    return Q / C_i


@formel("Kapazitiver Spannungsteiler", "U2 = U * C1 / (C1 + C2)", {"U": "V", "C1": "F", "C2": "F"}, "V", K2,
        dict(U=10.0, C1=1e-6, C2=4e-6))
def spannungsteiler_kapazitiv(U, C1, C2):
    """U2: Spannung an C2 (kleinere Kapazitaet -> groessere Teilspannung)."""
    return U * C1 / (C1 + C2)


@formel("Induktiver Spannungsteiler", "U2 = U * L2 / (L1 + L2)", {"U": "V", "L1": "H", "L2": "H"}, "V", K2,
        dict(U=10.0, L1=1e-3, L2=4e-3))
def spannungsteiler_induktiv(U, L1, L2):
    """U2: Spannung an L2 (ungekoppelte Spulen)."""
    return U * L2 / (L1 + L2)


@formel("Kapazitaet aus Ladung und Spannung", "C = Q / U", {"Q": "C", "U": "V"}, "F", K2, dict(Q=1e-3, U=10.0))
def kapazitaet_qu(Q, U):
    return Q / U


@formel("Induktivitaet aus Verkettungsfluss", "L = N Phi / I", {"N": "1", "Phi": "Wb", "I": "A"}, "H", K2,
        dict(N=100.0, Phi=1e-4, I=1.0))
def induktivitaet_fluss(N, Phi, I):
    return N * Phi / I


@formel("Kapazitaet aus Zeitkonstante (Messung)", "C = tau / R", {"tau": "s", "R": "ohm"}, "F", K2,
        dict(tau=1e-3, R=1000.0))
def kapazitaet_aus_tau(tau, R):
    return tau / R


@formel("Kapazitaet aus Blindwiderstand (Messung)", "C = 1 / (2 pi f Xc)", {"f": "Hz", "X_C": "ohm"}, "F", K2,
        dict(f=50.0, X_C=318.3))
def kapazitaet_aus_xc(f, X_C):
    return 1.0 / (2.0 * PI * f * X_C)


@formel("Induktivitaet aus Blindwiderstand (Messung)", "L = XL / (2 pi f)", {"f": "Hz", "X_L": "ohm"}, "H", K2,
        dict(f=50.0, X_L=314.16))
def induktivitaet_aus_xl(f, X_L):
    return X_L / (2.0 * PI * f)


@formel("Kapazitaet aus Resonanzfrequenz", "C = 1 / ((2 pi f0)^2 L)", {"f0": "Hz", "L": "H"}, "F", K2,
        dict(f0=5032.9, L=1e-3))
def kapazitaet_aus_resonanz(f0, L):
    return 1.0 / ((2.0 * PI * f0) ** 2 * L)


@formel("Induktivitaet aus Resonanzfrequenz", "L = 1 / ((2 pi f0)^2 C)", {"f0": "Hz", "C": "F"}, "H", K2,
        dict(f0=5032.9, C=1e-6))
def induktivitaet_aus_resonanz(f0, C):
    return 1.0 / ((2.0 * PI * f0) ** 2 * C)


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


@formel("Feldstaerke aus Kraft", "E = F / q", {"F": "N", "q": "C"}, "V/m", K3, dict(F=1e-3, q=1e-6))
def feldstaerke_kraft(F, q):
    return F / q


@formel("Kraft auf Ladung im E-Feld", "F = q E", {"q": "C", "E": "V/m"}, "N", K3, dict(q=1e-6, E=1000.0))
def kraft_e_feld(q, E):
    return q * E


@formel("Flaechenladungsdichte", "sigma = Q / A", {"Q": "C", "A": "m^2"}, "C/m^2", K3, dict(Q=1e-6, A=1e-2))
def flaechenladungsdichte(Q, A):
    return Q / A


@formel("Feldstaerke aus Flaechenladung", "E = sigma / (eps0 eps_r)", {"sigma": "C/m^2", "eps_r": "1"}, "V/m", K3,
        dict(sigma=1e-6, eps_r=1.0))
def feldstaerke_flaechenladung(sigma, eps_r):
    return sigma / (EPS0 * eps_r)


@formel("Flussdichte um Punktladung", "D = Q / (4 pi r^2)", {"Q": "C", "r": "m"}, "C/m^2", K3, dict(Q=1e-9, r=0.1))
def verschiebungsdichte_punktladung(Q, r):
    return Q / (4.0 * PI * r ** 2)


@formel("Flussdichte gerader Leiter", "B = mu0 mu_r I / (2 pi r)", {"I": "A", "r": "m", "mu_r": "1"}, "T", K3,
        dict(I=10.0, r=0.05, mu_r=1.0))
def b_feld_leiter(I, r, mu_r):
    return MU0 * mu_r * I / (2.0 * PI * r)


@formel("Flussdichte aus Leiterkraft", "B = F / (I l)", {"F": "N", "I": "A", "l": "m"}, "T", K3,
        dict(F=0.5, I=10.0, l=0.1))
def flussdichte_aus_kraft(F, I, l):
    return F / (I * l)


@formel("Fluss bei schraeger Flaeche", "Phi = B A cos(theta)", {"B": "T", "A": "m^2", "theta": "rad"}, "Wb", K3,
        dict(B=1.0, A=1e-2, theta=PI / 3))
def fluss_winkel(B, A, theta):
    """theta: Winkel zwischen Feldlinien und Flaechennormale."""
    return B * A * np.cos(theta)


@formel("Magnetischer Kreis mit Luftspalt (Fluss)", "Phi = N I / (l_Fe/(mu0 mu_r A) + l_L/(mu0 A))",
        {"N": "1", "I": "A", "l_fe": "m", "mu_r": "1", "l_luft": "m", "A": "m^2"}, "Wb", K3,
        dict(N=500.0, I=1.0, l_fe=0.2, mu_r=2000.0, l_luft=1e-3, A=4e-4))
def magn_fluss_luftspalt(N, I, l_fe, mu_r, l_luft, A):
    Rm = magn_widerstand(l_fe, mu_r, A) + magn_widerstand(l_luft, 1.0, A)
    return N * I / Rm


@formel("Scheitelwert Generatorspannung (rotierende Spule)", "Uhat = N B A omega", {"N": "1", "B": "T", "A": "m^2",
        "omega": "1/s"}, "V", K3, dict(N=100.0, B=0.5, A=1e-2, omega=314.16))
def generatorspannung_scheitel(N, B, A, omega):
    return N * B * A * omega


@formel("Wellenzahl", "k = 2 pi / lambda", {"lam": "m"}, "1/m", K3, dict(lam=0.5))
def wellenzahl(lam):
    return 2.0 * PI / lam


@formel("Wellenlaenge aus Ausbreitungsgeschwindigkeit", "lambda = v / f", {"v": "m/s", "f": "Hz"}, "m", K3,
        dict(v=343.0, f=1000.0))
def wellenlaenge_v(v, f):
    return v / f


@formel("Photonenenergie", "E = h f", {"f": "Hz"}, "J", K3, dict(f=5e14))
def photonenenergie(f):
    return H_PLANCK * f


@formel("Wellenlaenge aus Photonenenergie", "lambda = h c / E", {"E": "J"}, "m", K3, dict(E=3.2e-19))
def wellenlaenge_photon(E):
    return H_PLANCK * C0 / E


@formel("Brechungsgesetz (Snellius)", "theta2 = asin(n1 sin(theta1) / n2)", {"n1": "1", "theta1": "rad", "n2": "1"},
        "rad", K3, dict(n1=1.0, theta1=PI / 6, n2=1.5))
def snellius(n1, theta1, n2):
    with np.errstate(invalid="ignore"):
        return np.arcsin(n1 * np.sin(theta1) / n2)      # nan bei Totalreflexion


@formel("Doppler-Effekt", "f' = f (v + v_B) / (v - v_Q)", {"f": "Hz", "v": "m/s", "v_B": "m/s", "v_Q": "m/s"}, "Hz",
        K3, dict(f=1000.0, v=343.0, v_B=0.0, v_Q=30.0))
def doppler(f, v, v_B, v_Q):
    """v_B: Beobachter auf Quelle zu (+), v_Q: Quelle auf Beobachter zu (+)."""
    return f * (v + v_B) / (v - v_Q)


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


@formel("Periodendauer", "T = 1 / f", {"f": "Hz"}, "s", K4, dict(f=50.0))
def periodendauer(f):
    return 1.0 / f


@formel("Kreisfrequenz aus Periodendauer", "omega = 2 pi / T", {"T": "s"}, "1/s", K4, dict(T=0.02))
def kreisfrequenz_aus_T(T):
    return 2.0 * PI / T


@formel("Augenblickswert Sinus", "u(t) = Uhat sin(omega t + phi)", {"u_max": "V", "f": "Hz", "t": "s", "phi": "rad"},
        "V", K4, dict(u_max=325.0, f=50.0, t=0.004, phi=0.0))
def augenblickswert(u_max, f, t, phi):
    """Gilt ebenso fuer Stroeme (u_max in A -> Ergebnis in A)."""
    return u_max * np.sin(2.0 * PI * f * t + phi)


@formel("Addition Sinusgroessen gleicher Frequenz", "U = |U1 e^(j phi1) + U2 e^(j phi2)|, phi = arg(...)",
        {"U1": "V", "phi1": "rad", "U2": "V", "phi2": "rad"}, ("V", "rad"), K4,
        dict(U1=10.0, phi1=0.0, U2=10.0, phi2=PI / 2))
def zeigeraddition(U1, phi1, U2, phi2):
    z = U1 * np.exp(1j * phi1) + U2 * np.exp(1j * phi2)
    return np.abs(z), np.angle(z)


@formel("Effektivwert Rechteckpuls", "U = Uhat sqrt(D)", {"u_max": "V", "D": "1"}, "V", K4, dict(u_max=5.0, D=0.25))
def effektivwert_puls(u_max, D):
    """D: Tastgrad t_ein/T (D = 1 bzw. symmetrisches +-Rechteck: U = Uhat)."""
    return u_max * np.sqrt(D)


@formel("Mittelwert Rechteckpuls", "U = Uhat D", {"u_max": "V", "D": "1"}, "V", K4, dict(u_max=5.0, D=0.25))
def mittelwert_puls(u_max, D):
    return u_max * D


@formel("Effektivwert Einweggleichrichtung", "U = Uhat / 2", {"u_max": "V"}, "V", K4, dict(u_max=10.0))
def effektivwert_einweg(u_max):
    return u_max / 2.0


@formel("Gleichrichtwert Einweggleichrichtung", "U = Uhat / pi", {"u_max": "V"}, "V", K4, dict(u_max=10.0))
def gleichrichtwert_einweg(u_max):
    return u_max / PI


@formel("Scheitelfaktor", "CF = Uhat / U_eff", {"u_max": "V", "u_eff": "V"}, "1", K4, dict(u_max=325.27, u_eff=230.0))
def scheitelfaktor(u_max, u_eff):
    return u_max / u_eff


@formel("Formfaktor", "FF = U_eff / U_gl", {"u_eff": "V", "u_gl": "V"}, "1", K4, dict(u_eff=230.0, u_gl=207.0))
def formfaktor(u_eff, u_gl):
    return u_eff / u_gl


@formel("Impedanz aus U und I", "Z = U / I", {"U": "V", "I": "A"}, "ohm", K4, dict(U=230.0, I=2.0))
def impedanz_ui(U, I):
    return U / I


@formel("Scheinwiderstand R || X (Betrag)", "Z = R X / sqrt(R^2 + X^2)", {"R": "ohm", "X": "ohm"}, "ohm", K4,
        dict(R=100.0, X=100.0))
def z_parallel_betrag(R, X):
    return R * np.abs(X) / np.sqrt(R ** 2 + X ** 2)


@formel("Phasenwinkel R || X", "tan(phi) = R / X", {"R": "ohm", "X": "ohm"}, "rad", K4, dict(R=100.0, X=100.0))
def phasenwinkel_parallel(R, X):
    """Phase von Z (Spannung gegen Gesamtstrom); X > 0 induktiv, X < 0 kapazitiv.
    cos(phi) = Z/R = |X|/sqrt(R^2+X^2)."""
    return np.arctan2(R * np.sign(X), np.abs(X))


@formel("Gesamtstrom R || X", "I = sqrt(I_R^2 + I_X^2)", {"I_R": "A", "I_X": "A"}, "A", K4, dict(I_R=3.0, I_X=4.0))
def strom_gesamt_parallel(I_R, I_X):
    """I_X = I_L - I_C bei R||L||C."""
    return np.sqrt(I_R ** 2 + I_X ** 2)


@formel("Gesamtspannung R + X in Reihe", "U = sqrt(U_R^2 + U_X^2)", {"U_R": "V", "U_X": "V"}, "V", K4,
        dict(U_R=3.0, U_X=4.0))
def spannung_gesamt_reihe(U_R, U_X):
    """U_X = U_L - U_C bei R+L+C."""
    return np.sqrt(U_R ** 2 + U_X ** 2)


@formel("Teilspannung in Reihen-Wechselstromkreis", "U_x = U * X / Z", {"U": "V", "X": "ohm", "Z": "ohm"}, "V", K4,
        dict(U=10.0, X=4.0, Z=5.0))
def teilspannung_wechselstrom(U, X, Z):
    """Betrag: U_R = U R/Z, U_C = U X_C/Z, U_L = U X_L/Z."""
    return U * X / Z


@formel("Scheinleistung aus P und Q", "S = sqrt(P^2 + Q^2)", {"P": "W", "Q": "var"}, "VA", K4,
        dict(P=3000.0, Q=4000.0))
def scheinleistung_pq(P, Q):
    return np.sqrt(P ** 2 + Q ** 2)


@formel("Blindleistung aus S und P", "Q = sqrt(S^2 - P^2)", {"S": "VA", "P": "W"}, "var", K4,
        dict(S=5000.0, P=3000.0))
def blindleistung_sp(S, P):
    return np.sqrt(S ** 2 - P ** 2)


@formel("Blindleistung einer Spule", "Q_L = U^2 / (omega L)", {"U": "V", "f": "Hz", "L": "H"}, "var", K4,
        dict(U=230.0, f=50.0, L=0.5))
def blindleistung_spule(U, f, L):
    return U ** 2 / (2.0 * PI * f * L)


@formel("Blindleistung eines Kondensators", "Q_C = U^2 omega C", {"U": "V", "f": "Hz", "C": "F"}, "var", K4,
        dict(U=230.0, f=50.0, C=10e-6))
def blindleistung_kondensator(U, f, C):
    return U ** 2 * 2.0 * PI * f * C


@formel("Kompensationskapazitaet aus Blindleistung", "C = Q / (omega U^2)", {"Q": "var", "f": "Hz", "U": "V"}, "F",
        K4, dict(Q=1000.0, f=50.0, U=230.0))
def kapazitaet_aus_blindleistung(Q, f, U):
    return Q / (2.0 * PI * f * U ** 2)


@formel("Kompensationsinduktivitaet aus Blindleistung", "L = U^2 / (omega Q)", {"Q": "var", "f": "Hz", "U": "V"}, "H",
        K4, dict(Q=1000.0, f=50.0, U=230.0))
def induktivitaet_aus_blindleistung(Q, f, U):
    return U ** 2 / (2.0 * PI * f * Q)


@formel("Verlustfaktor Spule", "tan(delta) = R / (omega L)", {"f": "Hz", "R": "ohm", "L": "H"}, "1", K4,
        dict(f=1000.0, R=2.0, L=10e-3))
def verlustfaktor_spule(f, R, L):
    return R / (2.0 * PI * f * L)


@formel("Guete Spule", "Q = omega L / R", {"f": "Hz", "R": "ohm", "L": "H"}, "1", K4, dict(f=1000.0, R=2.0, L=10e-3))
def guete_spule(f, R, L):
    return 2.0 * PI * f * L / R


@formel("Verlustfaktor Kondensator (Reihen-ESR)", "tan(delta) = omega R_s C", {"f": "Hz", "R_s": "ohm", "C": "F"}, "1",
        K4, dict(f=100e3, R_s=0.05, C=10e-6))
def verlustfaktor_kondensator_reihe(f, R_s, C):
    return 2.0 * PI * f * R_s * C


@formel("Verlustfaktor Kondensator (Parallel-R)", "tan(delta) = 1 / (omega R_p C)", {"f": "Hz", "R_p": "ohm", "C": "F"},
        "1", K4, dict(f=50.0, R_p=10e6, C=1e-6))
def verlustfaktor_kondensator_parallel(f, R_p, C):
    return 1.0 / (2.0 * PI * f * R_p * C)


@formel("Wirkverluste Kondensator", "P = U^2 omega C tan(delta)", {"U": "V", "f": "Hz", "C": "F", "tan_delta": "1"},
        "W", K4, dict(U=230.0, f=50.0, C=10e-6, tan_delta=0.001))
def verlustleistung_kondensator(U, f, C, tan_delta):
    return U ** 2 * 2.0 * PI * f * C * tan_delta


@formel("Guete aus Verlustfaktor", "Q = 1 / tan(delta)", {"tan_delta": "1"}, "1", K4, dict(tan_delta=0.01))
def guete_aus_verlustfaktor(tan_delta):
    return 1.0 / tan_delta


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


@formel("Strangstrom Dreieck (symmetrisch)", "I_str = IL / sqrt(3)", {"I_L": "A"}, "A", K5, dict(I_L=17.32))
def strangstrom_dreieck(I_L):
    return I_L / np.sqrt(3.0)


@formel("Leiterstrom Dreieck (symmetrisch)", "IL = sqrt(3) I_str", {"I_str": "A"}, "A", K5, dict(I_str=10.0))
def leiterstrom_dreieck(I_str):
    return np.sqrt(3.0) * I_str


@formel("Wirkleistung Drehstrom aus Stranggroessen", "P = 3 U_str I_str cos(phi)",
        {"U_str": "V", "I_str": "A", "cos_phi": "1"}, "W", K5, dict(U_str=230.0, I_str=10.0, cos_phi=0.8))
def wirkleistung_drehstrom_strang(U_str, I_str, cos_phi):
    return 3.0 * U_str * I_str * cos_phi


@formel("Phasenversatz n-phasiges System", "dphi = 2 pi / n", {"n": "1"}, "rad", K5, dict(n=3.0))
def phasenversatz(n):
    return 2.0 * PI / n


@formel("Generatorfrequenz", "f = p n", {"p": "1", "n": "1/s"}, "Hz", K5, dict(p=2.0, n=25.0))
def generatorfrequenz(p, n):
    """p: Polpaarzahl, n in 1/s (n_min / 60)."""
    return p * n


_A120 = np.exp(-2j * PI / 3)          # Drehoperator: L2 = U e^(-j120), L3 = U e^(+j120)


@formel("Neutralleiterstrom (Zeigersumme)", "IN = |I1 e^(-j phi1) + I2 e^(-j(120+phi2)) + I3 e^(j(120-phi3))|",
        {"I1": "A", "I2": "A", "I3": "A", "phi1": "rad", "phi2": "rad", "phi3": "rad"}, "A", K5,
        dict(I1=10.0, I2=5.0, I3=5.0, phi1=0.0, phi2=0.0, phi3=0.0))
def neutralleiterstrom(I1, I2, I3, phi1, phi2, phi3):
    """Betraege der Strangstroeme und Phasenwinkel der Lasten (phi > 0: induktiv, Strom eilt nach).
    Achtung: NICHT sqrt(I1^2+I2^2+I3^2) - die Stroeme sind um 120 Grad versetzt."""
    s = (I1 * np.exp(-1j * phi1) + I2 * _A120 * np.exp(-1j * phi2) + I3 * np.conj(_A120) * np.exp(-1j * phi3))
    return np.abs(s)


@formel("Sternschaltung unsymmetrisch (mit N-Leiter)", "I_k = U_str,k / Z_k ; IN = -(I1+I2+I3) ; S = sum U I*",
        {"U_str": "V", "Z1": "ohm", "Z2": "ohm", "Z3": "ohm"}, ("A", "A", "A", "A", "VA"), K5,
        dict(U_str=230.0, Z1=23.0, Z2=46.0 + 10j, Z3=20.0 - 15j))
def drehstrom_stern_unsymmetrisch(U_str, Z1, Z2, Z3):
    """Komplexe Strangimpedanzen; Rueckgabe komplexe Leiterstroeme I1..I3, Neutralleiterstrom IN
    und komplexe Gesamtscheinleistung S = P + jQ (abs() -> Betraege)."""
    U = U_str * np.array([1.0, _A120, np.conj(_A120)])
    I = U / np.array([Z1, Z2, Z3])
    S = np.sum(U * np.conj(I))
    return I[0], I[1], I[2], -(I[0] + I[1] + I[2]), S


@formel("Dreieckschaltung unsymmetrisch", "I_12 = U_12 / Z12 ... ; I1 = I_12 - I_31 ; S = sum U I*",
        {"U_L": "V", "Z12": "ohm", "Z23": "ohm", "Z31": "ohm"}, ("A", "A", "A", "VA"), K5,
        dict(U_L=400.0, Z12=40.0, Z23=40.0 + 30j, Z31=80.0))
def drehstrom_dreieck_unsymmetrisch(U_L, Z12, Z23, Z31):
    """Komplexe Leiterstroeme I1..I3 und komplexe Gesamtscheinleistung."""
    U12 = U_L * np.exp(1j * PI / 6)            # verkettete Spannungen eilen U1 um 30 Grad vor
    U23, U31 = U12 * _A120, U12 * np.conj(_A120)
    I12, I23, I31 = U12 / Z12, U23 / Z23, U31 / Z31
    S = U12 * np.conj(I12) + U23 * np.conj(I23) + U31 * np.conj(I31)
    return I12 - I31, I23 - I12, I31 - I23, S


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


@formel("Uebersetzungsverhaeltnis", "ue = N1 / N2", {"N1": "1", "N2": "1"}, "1", K6, dict(N1=1000.0, N2=100.0))
def trafo_uebersetzung(N1, N2):
    return N1 / N2


@formel("Kupferverluste Transformator", "P_Cu = I1^2 R1 + I2^2 R2", {"I1": "A", "R1": "ohm", "I2": "A", "R2": "ohm"},
        "W", K6, dict(I1=1.0, R1=2.0, I2=10.0, R2=0.02))
def trafo_kupferverluste(I1, R1, I2, R2):
    return I1 ** 2 * R1 + I2 ** 2 * R2


@formel("Scheitelwert Kernfluss", "Phi = U / (4,44 f N)", {"U": "V", "f": "Hz", "N": "1"}, "Wb", K6,
        dict(U=230.0, f=50.0, N=500.0))
def trafo_fluss(U, f, N):
    return np.sqrt(2.0) * U / (2.0 * PI * f * N)


@formel("Kurzschlussimpedanz", "Zk = Uk / Ik", {"U_k": "V", "I_k": "A"}, "ohm", K6, dict(U_k=9.2, I_k=4.35))
def trafo_kurzschlussimpedanz(U_k, I_k):
    return U_k / I_k


@formel("Relative Kurzschlussspannung", "uk = Uk / Un", {"U_k": "V", "U_n": "V"}, "1", K6, dict(U_k=9.2, U_n=230.0))
def trafo_uk(U_k, U_n):
    return U_k / U_n


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


@formel("Resonanzkreisfrequenz", "omega0 = 1 / sqrt(L C)", {"L": "H", "C": "F"}, "1/s", K7, dict(L=1e-3, C=1e-6))
def resonanzkreisfrequenz(L, C):
    return 1.0 / np.sqrt(L * C)


@formel("Resonanzwiderstand Parallelkreis (verlustb. Spule)", "Z0 = L / (R_s C)", {"L": "H", "C": "F", "R_s": "ohm"},
        "ohm", K7, dict(L=1e-3, C=1e-6, R_s=1.0))
def resonanzwiderstand_parallel(L, C, R_s):
    """R_s: Verlustwiderstand in Reihe zur Spule (Q >> 1)."""
    return L / (R_s * C)


@formel("Bandbreite Reihenkreis", "B = R / (2 pi L)", {"R": "ohm", "L": "H"}, "Hz", K7, dict(R=10.0, L=1e-3))
def bandbreite_reihe(R, L):
    return R / (2.0 * PI * L)


@formel("Guete aus Bandbreite", "Q = f0 / B", {"f0": "Hz", "B": "Hz"}, "1", K7, dict(f0=5000.0, B=500.0))
def guete_aus_bandbreite(f0, B):
    return f0 / B


@formel("Spannungsueberhoehung Reihenresonanz", "U_L = U_C = Q U", {"U": "V", "Q": "1"}, "V", K7, dict(U=1.0, Q=50.0))
def spannungsueberhoehung(U, Q):
    """Gilt sinngemaess fuer die Stromueberhoehung im Parallelkreis: I_L = I_C = Q I."""
    return Q * U


@formel("Energie im Schwingkreis", "W = 1/2 L i^2 + 1/2 C u^2", {"L": "H", "I": "A", "C": "F", "U": "V"}, "J", K7,
        dict(L=1e-3, I=0.1, C=1e-6, U=1.0))
def energie_schwingkreis(L, I, C, U):
    return 0.5 * L * I ** 2 + 0.5 * C * U ** 2


@formel("Anzapfung: Uebersetzung", "ue = sqrt(L_ges / L_teil)", {"L_ges": "H", "L_teil": "H"}, "1", K7,
        dict(L_ges=100e-6, L_teil=25e-6))
def anzapfung_uebersetzung(L_ges, L_teil):
    """Fest gekoppelte Spule: L ~ N^2  ->  ue = N_ges/N_teil."""
    return np.sqrt(L_ges / L_teil)


@formel("Anzapfung: transformierter Widerstand", "R' = R_teil * L_ges / L_teil", {"R": "ohm", "L_ges": "H",
        "L_teil": "H"}, "ohm", K7, dict(R=50.0, L_ges=100e-6, L_teil=25e-6))
def anzapfung_widerstand(R, L_ges, L_teil):
    """An der Anzapfung angeschlossener Widerstand R erscheint am ganzen Kreis als R ue^2."""
    return R * L_ges / L_teil


@formel("Eigenfrequenz Feder-Masse-Schwinger", "f0 = sqrt(k/m) / (2 pi)", {"k": "N/m", "m": "kg"}, "Hz", K7,
        dict(k=100.0, m=1.0))
def eigenfrequenz_feder(k, m):
    return np.sqrt(k / m) / (2.0 * PI)


@formel("Daempfungsgrad mechanisch", "zeta = c / (2 sqrt(m k))", {"c": "N*s/m", "m": "kg", "k": "N/m"}, "1", K7,
        dict(c=2.0, m=1.0, k=100.0))
def daempfungsgrad_mech(c, m, k):
    return c / (2.0 * np.sqrt(m * k))


@formel("Daempfungsgrad aus Guete", "zeta = 1 / (2 Q)", {"Q": "1"}, "1", K7, dict(Q=10.0))
def daempfungsgrad_aus_guete(Q):
    return 1.0 / (2.0 * Q)


@formel("Logarithmisches Dekrement", "Lambda = (1/n) ln(x(t) / x(t + nT))", {"x1": "1", "x2": "1", "n": "1"}, "1", K7,
        dict(x1=1.0, x2=0.5, n=2.0))
def log_dekrement(x1, x2, n):
    """x1, x2: Amplituden im Abstand von n Perioden (beliebige, gleiche Einheit)."""
    return np.log(x1 / x2) / n


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


@formel("RC Entladestrom: i(t)", "i = -U0/R exp(-t/RC)", {"t": "s", "U0": "V", "R": "ohm", "C": "F"}, "A", K8,
        dict(t=1e-3, U0=5.0, R=1000.0, C=1e-6))
def rc_entladestrom(t, U0, R, C):
    return -U0 / R * np.exp(-t / (R * C))


@formel("RL Einschalten: uL(t)", "uL = U exp(-tR/L)", {"t": "s", "U": "V", "R": "ohm", "L": "H"}, "V", K8,
        dict(t=1e-3, U=10.0, R=10.0, L=10e-3))
def rl_spulenspannung(t, U, R, L):
    return U * np.exp(-t * R / L)


@formel("RC Ladezeit zwischen zwei Spannungen", "t = RC ln((U0 - U1) / (U0 - U2))",
        {"R": "ohm", "C": "F", "U0": "V", "U1": "V", "U2": "V"}, "s", K8,
        dict(R=1000.0, C=1e-6, U0=10.0, U1=2.0, U2=8.0))
def rc_ladezeit(R, C, U0, U1, U2):
    """Laden Richtung Endwert U0 von U1 auf U2 (gilt auch fuer Entladen: U0 = 0)."""
    return R * C * np.log((U0 - U1) / (U0 - U2))


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


@formel("RC-Tiefpass Betrag", "|H| = 1 / sqrt(1 + (omega R C)^2)", {"f": "Hz", "R": "ohm", "C": "F"}, "1", K9,
        dict(f=1000.0, R=1000.0, C=159.155e-9))
def rc_tiefpass_betrag(f, R, C):
    return 1.0 / np.sqrt(1.0 + (2.0 * PI * f * R * C) ** 2)


@formel("RC-Tiefpass Phase", "phi = -atan(omega R C)", {"f": "Hz", "R": "ohm", "C": "F"}, "rad", K9,
        dict(f=1000.0, R=1000.0, C=159.155e-9))
def rc_tiefpass_phase(f, R, C):
    return -np.arctan(2.0 * PI * f * R * C)


@formel("CR-Hochpass Betrag", "|H| = omega R C / sqrt(1 + (omega R C)^2)", {"f": "Hz", "R": "ohm", "C": "F"}, "1", K9,
        dict(f=1000.0, R=1000.0, C=159.155e-9))
def rc_hochpass_betrag(f, R, C):
    x = 2.0 * PI * f * R * C
    return x / np.sqrt(1.0 + x ** 2)


@formel("CR-Hochpass Phase", "phi = atan(1 / (omega R C))", {"f": "Hz", "R": "ohm", "C": "F"}, "rad", K9,
        dict(f=1000.0, R=1000.0, C=159.155e-9))
def rc_hochpass_phase(f, R, C):
    return np.arctan(1.0 / (2.0 * PI * f * R * C))


@formel("LR-Tiefpass Betrag", "|H| = R / sqrt(R^2 + (omega L)^2)", {"f": "Hz", "R": "ohm", "L": "H"}, "1", K9,
        dict(f=1000.0, R=100.0, L=15.9e-3))
def rl_tiefpass_betrag(f, R, L):
    return R / np.sqrt(R ** 2 + (2.0 * PI * f * L) ** 2)


@formel("RL-Hochpass Betrag", "|H| = omega L / sqrt(R^2 + (omega L)^2)", {"f": "Hz", "R": "ohm", "L": "H"}, "1", K9,
        dict(f=1000.0, R=100.0, L=15.9e-3))
def rl_hochpass_betrag(f, R, L):
    wl = 2.0 * PI * f * L
    return wl / np.sqrt(R ** 2 + wl ** 2)


@formel("Siebfaktor RC-Glied", "s = sqrt(1 + (omega R C)^2)", {"f": "Hz", "R": "ohm", "C": "F"}, "1", K9,
        dict(f=100.0, R=100.0, C=1e-3))
def siebfaktor_rc(f, R, C):
    """Brummspannung vor / nach dem Siebglied (Kehrwert des Tiefpassbetrags)."""
    return np.sqrt(1.0 + (2.0 * PI * f * R * C) ** 2)


@formel("Siebfaktor LC-Glied", "s = omega^2 L C - 1", {"f": "Hz", "L": "H", "C": "F"}, "1", K9,
        dict(f=100.0, L=0.1, C=1e-3))
def siebfaktor_lc(f, L, C):
    """Unbelastetes LC-Siebglied oberhalb der Resonanz (verlustfrei)."""
    return (2.0 * PI * f) ** 2 * L * C - 1.0


@formel("LC-Tiefpass (unbelastet)", "H = Z_C / (Z_L + Z_C) = 1 / (1 - omega^2 L C)", {"f": "Hz", "L": "H", "C": "F"},
        "1", K9, dict(f=100.0, L=0.1, C=1e-3))
def lc_tiefpass(f, L, C):
    return z_kondensator(f, C) / (z_spule(f, L) + z_kondensator(f, C))


@formel("Mittenfrequenz (geometrisch)", "f0 = sqrt(f_u f_o)", {"f_u": "Hz", "f_o": "Hz"}, "Hz", K9,
        dict(f_u=300.0, f_o=3400.0))
def mittenfrequenz(f_u, f_o):
    return np.sqrt(f_u * f_o)


@formel("Bandbreite aus Grenzfrequenzen", "B = f_o - f_u", {"f_u": "Hz", "f_o": "Hz"}, "Hz", K9,
        dict(f_u=300.0, f_o=3400.0))
def bandbreite_grenzen(f_u, f_o):
    return f_o - f_u


@formel("Bandpass 2. Ordnung", "H = 1 / (1 + j Q (f/f0 - f0/f))", {"f": "Hz", "f0": "Hz", "Q": "1"}, "1", K9,
        dict(f=1100.0, f0=1000.0, Q=5.0))
def bandpass_2(f, f0, Q):
    return 1.0 / (1.0 + 1j * Q * (f / f0 - f0 / f))


@formel("Bandsperre 2. Ordnung", "H = (1 - (f/f0)^2) / (1 - (f/f0)^2 + j f/(f0 Q))", {"f": "Hz", "f0": "Hz", "Q": "1"},
        "1", K9, dict(f=1100.0, f0=1000.0, Q=5.0))
def bandsperre_2(f, f0, Q):
    """= 1 - bandpass_2 (Kerbfilter, Nullstelle bei f0)."""
    x = f / f0
    return (1.0 - x ** 2) / (1.0 - x ** 2 + 1j * x / Q)


@formel("Bandpass aus Hoch- und Tiefpass 1. Ordnung (Betrag)", "|H| = (f/f_u)/sqrt(1+(f/f_u)^2) / sqrt(1+(f/f_o)^2)",
        {"f": "Hz", "f_u": "Hz", "f_o": "Hz"}, "1", K9, dict(f=1000.0, f_u=300.0, f_o=3400.0))
def bandpass_1_betrag(f, f_u, f_o):
    return np.abs(hochpass_1(f, f_u) * tiefpass_1(f, f_o))


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


@formel("Spannungsverstaerkung", "Vu = Ua / Ue", {"U_a": "V", "U_e": "V"}, "1", K10, dict(U_a=5.0, U_e=0.05))
def spannungsverstaerkung(U_a, U_e):
    return U_a / U_e


@formel("Stromverstaerkung", "Vi = Ia / Ie", {"I_a": "A", "I_e": "A"}, "1", K10, dict(I_a=0.1, I_e=1e-3))
def stromverstaerkung(I_a, I_e):
    return I_a / I_e


@formel("Leistungsverstaerkung", "Vp = Pa / Pe", {"P_a": "W", "P_e": "W"}, "1", K10, dict(P_a=10.0, P_e=0.01))
def leistungsverstaerkung(P_a, P_e):
    return P_a / P_e


@formel("Leistungsverstaerkung aus Vu", "Vp = Vu^2 Re / Ra", {"V_u": "1", "R_e": "ohm", "R_a": "ohm"}, "1", K10,
        dict(V_u=100.0, R_e=10e3, R_a=8.0))
def leistungsverstaerkung_vu(V_u, R_e, R_a):
    return V_u ** 2 * R_e / R_a


@formel("Verstaerkung mit Gegenkopplung", "V' = V / (1 + k V)", {"V": "1", "k": "1"}, "1", K10, dict(V=1e5, k=0.01))
def verstaerkung_gegenkopplung(V, k):
    return V / (1.0 + k * V)


@formel("Rueckkopplungsfaktor Spannungsteiler", "k = Rg / (Rf + Rg)", {"Rf": "ohm", "Rg": "ohm"}, "1", K10,
        dict(Rf=9e3, Rg=1e3))
def rueckkopplungsfaktor(Rf, Rg):
    return Rg / (Rf + Rg)


@formel("Eingangswiderstand mit Gegenkopplung (Serien-)", "Ze' = Ze (1 + k V)", {"Z_e": "ohm", "V": "1", "k": "1"},
        "ohm", K10, dict(Z_e=1e6, V=1e5, k=0.01))
def eingangswiderstand_gegenkopplung(Z_e, V, k):
    return Z_e * (1.0 + k * V)


@formel("Ausgangswiderstand mit Gegenkopplung (Spannungs-)", "Za' = Za / (1 + k V)", {"Z_a": "ohm", "V": "1", "k": "1"},
        "ohm", K10, dict(Z_a=100.0, V=1e5, k=0.01))
def ausgangswiderstand_gegenkopplung(Z_a, V, k):
    return Z_a / (1.0 + k * V)


@formel("Bandbreite mit Gegenkopplung", "B' = B (1 + k V)", {"B": "Hz", "V": "1", "k": "1"}, "Hz", K10,
        dict(B=10.0, V=1e5, k=0.01))
def bandbreite_gegenkopplung(B, V, k):
    return B * (1.0 + k * V)


def verstaerkung_kette(*V):
    """Mehrstufiger Verstaerker: V_ges = V1 * V2 * ... (in dB: Summe der Stufen-dB)."""
    return np.prod([np.asarray(v, dtype=float) for v in V], axis=0)


@formel("Verstaerkung ueber Frequenz (1 Pol)", "V(f) = V0 / sqrt(1 + (f/fg)^2)", {"V0": "1", "f": "Hz", "fg": "Hz"},
        "1", K10, dict(V0=100.0, f=10e3, fg=10e3))
def verstaerkung_frequenz(V0, f, fg):
    return V0 / np.sqrt(1.0 + (f / fg) ** 2)


@formel("Phasengang Verstaerker (1 Pol)", "phi = -atan(f / fg)", {"f": "Hz", "fg": "Hz"}, "rad", K10,
        dict(f=10e3, fg=10e3))
def phase_verstaerker(f, fg):
    return -np.arctan(f / fg)


@formel("Slew-Rate", "SR = dU / dt", {"dU": "V", "dt": "s"}, "V/s", K10, dict(dU=10.0, dt=1e-5))
def slew_rate(dU, dt):
    return dU / dt


@formel("OPV Ausgang invertierend", "Ua = -Rf/Rin Ue", {"Rf": "ohm", "Rin": "ohm", "U_e": "V"}, "V", K10,
        dict(Rf=10e3, Rin=1e3, U_e=0.5))
def opv_ua_invertierend(Rf, Rin, U_e):
    return -Rf / Rin * U_e


@formel("OPV Ausgang nichtinvertierend", "Ua = (1 + Rf/Rg) Ue", {"Rf": "ohm", "Rg": "ohm", "U_e": "V"}, "V", K10,
        dict(Rf=10e3, Rg=1e3, U_e=0.5))
def opv_ua_nichtinvertierend(Rf, Rg, U_e):
    return (1.0 + Rf / Rg) * U_e


@formel("OPV Differenzverstaerker (Subtrahierer)", "Ua = Rf/R1 (U2 - U1)  (Rf/R1 = R3/R2)",
        {"Rf": "ohm", "R1": "ohm", "U1": "V", "U2": "V"}, "V", K10, dict(Rf=10e3, R1=1e3, U1=1.0, U2=1.2))
def opv_differenz(Rf, R1, U1, U2):
    return Rf / R1 * (U2 - U1)


@formel("OPV Summierer (2 Eingaenge)", "Ua = -Rf (U1/R1 + U2/R2)",
        {"Rf": "ohm", "R1": "ohm", "R2": "ohm", "U1": "V", "U2": "V"}, "V", K10,
        dict(Rf=10e3, R1=10e3, R2=5e3, U1=0.5, U2=0.25))
def opv_summierer(Rf, R1, R2, U1, U2):
    return -Rf * (U1 / R1 + U2 / R2)


def opv_summierer_n(Rf, R, U):
    """Summierer mit beliebig vielen Eingaengen: Ua = -Rf * sum(U_i / R_i)."""
    return -Rf * np.sum(np.asarray(U, dtype=float) / np.asarray(R, dtype=float))


@formel("OPV Integrator (Zeitverlauf)", "ua(t) = -1/(RC) int ue dt + ua(0)",
        {"t": "s", "u_e": "V", "R": "ohm", "C": "F", "U0": "V"}, "V", K10,
        dict(t=[0.0, 1e-3, 2e-3], u_e=[1.0, 1.0, 1.0], R=10e3, C=100e-9, U0=0.0))
def opv_integrator(t, u_e, R, C, U0):
    """Numerisch (Trapezregel) fuer abgetastetes Eingangssignal."""
    t = np.asarray(t, dtype=float)
    u = np.asarray(u_e, dtype=float)
    integral = np.concatenate(([0.0], np.cumsum((u[1:] + u[:-1]) / 2.0 * np.diff(t))))
    return U0 - integral / (R * C)


@formel("OPV Integrator (Betrag)", "|V| = 1 / (omega R C)", {"f": "Hz", "R": "ohm", "C": "F"}, "1", K10,
        dict(f=1000.0, R=10e3, C=100e-9))
def opv_integrator_betrag(f, R, C):
    return 1.0 / (2.0 * PI * f * R * C)


@formel("OPV Differenzierer (Zeitverlauf)", "ua = -R C due/dt", {"t": "s", "u_e": "V", "R": "ohm", "C": "F"}, "V",
        K10, dict(t=[0.0, 1e-3, 2e-3], u_e=[0.0, 1.0, 2.0], R=10e3, C=100e-9))
def opv_differenzierer(t, u_e, R, C):
    return -R * C * np.gradient(np.asarray(u_e, dtype=float), np.asarray(t, dtype=float))


@formel("OPV Differenzierer (Betrag)", "|V| = omega R C", {"f": "Hz", "R": "ohm", "C": "F"}, "1", K10,
        dict(f=1000.0, R=10e3, C=100e-9))
def opv_differenzierer_betrag(f, R, C):
    return 2.0 * PI * f * R * C


@formel("OPV aktiver Tiefpass (invertierend, C || Rf)", "|V| = (Rf/Rin) / sqrt(1 + (f/fg)^2), fg = 1/(2 pi Rf C)",
        {"f": "Hz", "Rf": "ohm", "Rin": "ohm", "C": "F"}, "1", K10, dict(f=1000.0, Rf=10e3, Rin=1e3, C=15.9e-9))
def opv_tiefpass_betrag(f, Rf, Rin, C):
    return Rf / Rin / np.sqrt(1.0 + (2.0 * PI * f * Rf * C) ** 2)


@formel("OPV aktiver Hochpass (invertierend, C in Reihe Rin)",
        "|V| = (Rf/Rin) (f/fg) / sqrt(1 + (f/fg)^2), fg = 1/(2 pi Rin C)",
        {"f": "Hz", "Rf": "ohm", "Rin": "ohm", "C": "F"}, "1", K10, dict(f=1000.0, Rf=10e3, Rin=1e3, C=159e-9))
def opv_hochpass_betrag(f, Rf, Rin, C):
    x = 2.0 * PI * f * Rin * C
    return Rf / Rin * x / np.sqrt(1.0 + x ** 2)


@formel("Sinus-Ausgangsleistung", "P = Uhat^2 / (2 RL)", {"u_max": "V", "R_L": "ohm"}, "W", K10,
        dict(u_max=10.0, R_L=8.0))
def ausgangsleistung_sinus(u_max, R_L):
    return u_max ** 2 / (2.0 * R_L)


@formel("Wirkungsgrad Gegentakt-B-Endstufe", "eta = pi/4 Uhat / U_b", {"u_max": "V", "U_b": "V"}, "1", K10,
        dict(u_max=10.0, U_b=12.0))
def wirkungsgrad_b_endstufe(u_max, U_b):
    """U_b: Betriebsspannung je Halbwelle (+-U_b). Maximal pi/4 = 78,5 % (A-Betrieb: max. 25 % bzw. 50 % mit Trafo)."""
    return PI / 4.0 * u_max / U_b


@formel("Verlustleistung Endstufe", "Pv = P_zu - P_ab", {"P_zu": "W", "P_ab": "W"}, "W", K10, dict(P_zu=20.0, P_ab=15.0))
def verlustleistung(P_zu, P_ab):
    return P_zu - P_ab


def klirrfaktor(U1, *U_harm):
    """THD (bezogen auf Grundschwingung): k = sqrt(U2^2 + U3^2 + ...) / U1."""
    return np.sqrt(np.sum(np.asarray(U_harm, dtype=float) ** 2)) / U1


def klirrfaktor_gesamt(U1, *U_harm):
    """Klirrfaktor nach DIN (bezogen auf Gesamteffektivwert): sqrt(sum Un^2) / sqrt(U1^2 + sum Un^2)."""
    h = np.sum(np.asarray(U_harm, dtype=float) ** 2)
    return np.sqrt(h) / np.sqrt(U1 ** 2 + h)


@formel("Signal-Rausch-Abstand", "SNR = 10 lg(P_S / P_N)", {"P_S": "W", "P_N": "W"}, "1", K10,
        dict(P_S=1e-3, P_N=1e-9))
def snr_db(P_S, P_N):
    return 10.0 * np.log10(P_S / P_N)


@formel("Rauschleistung (verfuegbar)", "P_N = kB T B", {"T": "K", "B": "Hz"}, "W", K10, dict(T=290.0, B=1e6))
def rauschleistung(T, B):
    return K_B * T * B


@formel("Rauschzahl", "F = SNR_ein / SNR_aus (linear)", {"snr_ein": "1", "snr_aus": "1"}, "1", K10,
        dict(snr_ein=1000.0, snr_aus=500.0))
def rauschzahl(snr_ein, snr_aus):
    """Lineare Verhaeltnisse; Rauschmass NF = 10 lg(F) (db_leistung)."""
    return snr_ein / snr_aus


@formel("Rauschtemperatur", "T_e = T0 (F - 1)", {"F": "1", "T0": "K"}, "K", K10, dict(F=2.0, T0=290.0))
def rauschtemperatur(F, T0):
    return T0 * (F - 1.0)


def friis(F, G):
    """Gesamtrauschzahl einer Kette (lineare Werte): F = F1 + (F2-1)/G1 + (F3-1)/(G1 G2) + ..."""
    F = np.asarray(F, dtype=float)
    G = np.asarray(G, dtype=float)
    g = np.concatenate(([1.0], np.cumprod(G[:len(F) - 1])))
    return F[0] + np.sum((F[1:] - 1.0) / g[1:])


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


@formel("Verlustleistung LED-Vorwiderstand", "Pv = (Uq - Uf) I", {"Uq": "V", "Uf": "V", "I": "A"}, "W", K11,
        dict(Uq=12.0, Uf=2.0, I=0.02))
def led_vorwiderstand_leistung(Uq, Uf, I):
    return (Uq - Uf) * I


@formel("Sperrschichtkapazitaet / Kapazitaetsdiode", "C = C0 / (1 + U_R/U_D)^n",
        {"C0": "F", "U_R": "V", "U_D": "V", "n": "1"}, "F", K11, dict(C0=30e-12, U_R=4.0, U_D=0.7, n=0.5))
def sperrschichtkapazitaet(C0, U_R, U_D, n):
    """n = 0,5 abrupter, 1/3 linearer Uebergang, ~2 hyperabrupt (Abstimmdiode)."""
    return C0 / (1.0 + U_R / U_D) ** n


@formel("Kapazitaetsverhaeltnis Kapazitaetsdiode", "C1/C2 = ((1 + U_R2/U_D) / (1 + U_R1/U_D))^n",
        {"U_R1": "V", "U_R2": "V", "U_D": "V", "n": "1"}, "1", K11, dict(U_R1=1.0, U_R2=8.0, U_D=0.7, n=0.5))
def kapazitaetsverhaeltnis(U_R1, U_R2, U_D, n):
    return ((1.0 + U_R2 / U_D) / (1.0 + U_R1 / U_D)) ** n


@formel("Verlustleistung Z-Diode", "P_Z = U_Z I_Z", {"U_Z": "V", "I_Z": "A"}, "W", K11, dict(U_Z=5.1, I_Z=0.02))
def zdiode_leistung(U_Z, I_Z):
    return U_Z * I_Z


@formel("Differentieller Widerstand Z-Diode", "r_Z = dU_Z / dI_Z", {"dU_Z": "V", "dI_Z": "A"}, "ohm", K11,
        dict(dU_Z=0.1, dI_Z=0.01))
def zdiode_rz(dU_Z, dI_Z):
    return dU_Z / dI_Z


@formel("Vorwiderstand Z-Diode", "R_V = (U_e - U_Z) / (I_Z + I_L)", {"U_e": "V", "U_Z": "V", "I_Z": "A", "I_L": "A"},
        "ohm", K11, dict(U_e=12.0, U_Z=5.1, I_Z=0.01, I_L=0.02))
def zdiode_vorwiderstand(U_e, U_Z, I_Z, I_L):
    return (U_e - U_Z) / (I_Z + I_L)


@formel("Vorwiderstand Z-Diode (zulaessiger Bereich)",
        "R_min = (U_e,max - U_Z)/(I_Z,max + I_L,min) ; R_max = (U_e,min - U_Z)/(I_Z,min + I_L,max)",
        {"U_e_min": "V", "U_e_max": "V", "U_Z": "V", "I_Z_min": "A", "I_Z_max": "A", "I_L_min": "A", "I_L_max": "A"},
        ("ohm", "ohm"), K11,
        dict(U_e_min=11.0, U_e_max=14.0, U_Z=5.1, I_Z_min=0.005, I_Z_max=0.08, I_L_min=0.0, I_L_max=0.03))
def zdiode_vorwiderstand_bereich(U_e_min, U_e_max, U_Z, I_Z_min, I_Z_max, I_L_min, I_L_max):
    """R_V muss zwischen R_min und R_max liegen (sonst Ueberlast bzw. Stabilisierung reisst ab)."""
    return (U_e_max - U_Z) / (I_Z_max + I_L_min), (U_e_min - U_Z) / (I_Z_min + I_L_max)


@formel("Z-Strom bei Parallelstabilisierung", "I_Z = (U_e - U_Z)/R_V - I_L", {"U_e": "V", "U_Z": "V", "R_V": "ohm",
        "I_L": "A"}, "A", K11, dict(U_e=12.0, U_Z=5.1, R_V=230.0, I_L=0.02))
def zdiode_strom(U_e, U_Z, R_V, I_L):
    return (U_e - U_Z) / R_V - I_L


@formel("Max. Laststrom Parallelstabilisierung", "I_L,max = (U_e - U_Z)/R_V - I_Z,min",
        {"U_e": "V", "U_Z": "V", "R_V": "ohm", "I_Z_min": "A"}, "A", K11,
        dict(U_e=12.0, U_Z=5.1, R_V=230.0, I_Z_min=0.005))
def zdiode_laststrom_max(U_e, U_Z, R_V, I_Z_min):
    return (U_e - U_Z) / R_V - I_Z_min


@formel("Ausgangsspannung Reihenstabilisierung", "U_a = U_Z - U_BE", {"U_Z": "V", "U_BE": "V"}, "V", K11,
        dict(U_Z=5.6, U_BE=0.7))
def reihenstabilisierung_ua(U_Z, U_BE):
    """Laengstransistor als Emitterfolger an der Z-Diode."""
    return U_Z - U_BE


@formel("Welligkeit", "w = U_Br,eff / U_gl", {"U_br": "V", "U_gl": "V"}, "1", K11, dict(U_br=0.5, U_gl=12.0))
def welligkeit(U_br, U_gl):
    return U_br / U_gl


@formel("Effektivwert Brummspannung (Saegezahn)", "U_Br,eff = U_Br,ss / (2 sqrt(3))", {"U_ss": "V"}, "V", K11,
        dict(U_ss=1.0))
def brummspannung_effektiv(U_ss):
    return U_ss / (2.0 * np.sqrt(3.0))


@formel("Gleichspannung nach Glaettung", "U_gl = Uhat - U_Br,ss / 2", {"u_max": "V", "U_ss": "V"}, "V", K11,
        dict(u_max=16.3, U_ss=1.0))
def gleichspannung_geglaettet(u_max, U_ss):
    """Mittelwert der saegezahnfoermigen Spannung am Ladekondensator."""
    return u_max - U_ss / 2.0


@formel("Stromverstaerkung B aus alpha", "B = alpha / (1 - alpha)", {"alpha": "1"}, "1", K11, dict(alpha=0.99))
def bjt_beta_aus_alpha(alpha):
    return alpha / (1.0 - alpha)


@formel("Stromverstaerkung alpha aus B", "alpha = B / (B + 1)", {"beta": "1"}, "1", K11, dict(beta=99.0))
def bjt_alpha_aus_beta(beta):
    return beta / (beta + 1.0)


@formel("Emitterstrom", "I_E = I_C + I_B", {"I_C": "A", "I_B": "A"}, "A", K11, dict(I_C=10e-3, I_B=0.1e-3))
def bjt_ie(I_C, I_B):
    return I_C + I_B


@formel("Verlustleistung BJT", "P = U_CE I_C + U_BE I_B", {"U_CE": "V", "I_C": "A", "U_BE": "V", "I_B": "A"}, "W",
        K11, dict(U_CE=5.0, I_C=0.1, U_BE=0.7, I_B=1e-3))
def bjt_verlustleistung(U_CE, I_C, U_BE, I_B):
    return U_CE * I_C + U_BE * I_B


@formel("Differentieller Emitterwiderstand", "r_e = UT / I_E", {"I_E": "A", "T": "K"}, "ohm", K11,
        dict(I_E=1e-3, T=300.0))
def bjt_re(I_E, T):
    """Bei Raumtemperatur ~ 26 mV / I_E."""
    return thermospannung(T) / I_E


@formel("Basisvorwiderstand", "R_B = (U_B - U_BE) / I_B", {"U_B": "V", "U_BE": "V", "I_B": "A"}, "ohm", K11,
        dict(U_B=12.0, U_BE=0.7, I_B=50e-6))
def bjt_basisvorwiderstand(U_B, U_BE, I_B):
    return (U_B - U_BE) / I_B


@formel("Kollektor-Emitter-Spannung im Arbeitspunkt", "U_CE = U_CC - I_C R_C - I_E R_E",
        {"U_CC": "V", "I_C": "A", "R_C": "ohm", "I_E": "A", "R_E": "ohm"}, "V", K11,
        dict(U_CC=12.0, I_C=2e-3, R_C=2.2e3, I_E=2.02e-3, R_E=1e3))
def bjt_uce(U_CC, I_C, R_C, I_E, R_E):
    """Ohne Emitterwiderstand R_E = 0. Gilt auch fuer Optokoppler-/Fototransistorausgang."""
    return U_CC - I_C * R_C - I_E * R_E


@formel("Emitterstrom bei Basisspannungsteiler", "I_E = (U_Th - U_BE) / (R_E + R_Th/(B+1)), U_Th = U R2/(R1+R2)",
        {"U_CC": "V", "R1": "ohm", "R2": "ohm", "R_E": "ohm", "beta": "1", "U_BE": "V"}, "A", K11,
        dict(U_CC=12.0, R1=47e3, R2=10e3, R_E=1e3, beta=200.0, U_BE=0.7))
def bjt_ie_spannungsteiler(U_CC, R1, R2, R_E, beta, U_BE):
    """R1 oben (an U_CC), R2 unten; exakt ueber Ersatzspannungsquelle des Teilers."""
    U_th = U_CC * R2 / (R1 + R2)
    R_th = R1 * R2 / (R1 + R2)
    return (U_th - U_BE) / (R_E + R_th / (beta + 1.0))


@formel("Emitterschaltung Spannungsverstaerkung", "Vu = -R_C / (R_E + r_e)", {"R_C": "ohm", "R_E": "ohm", "r_e": "ohm"},
        "1", K11, dict(R_C=4.7e3, R_E=470.0, r_e=13.0))
def emitterschaltung_vu(R_C, R_E, r_e):
    """R_E: nicht ueberbrueckter Emitterwiderstand (Stromgegenkopplung); R_E = 0 -> Vu = -R_C/r_e.
    R_C ggf. || Last."""
    return -R_C / (R_E + r_e)


@formel("Emitterschaltung Eingangswiderstand (Transistor)", "r_ein = (B + 1)(R_E + r_e)",
        {"beta": "1", "R_E": "ohm", "r_e": "ohm"}, "ohm", K11, dict(beta=200.0, R_E=470.0, r_e=13.0))
def emitterschaltung_rin(beta, R_E, r_e):
    """Parallel dazu liegt noch der Basisspannungsteiler."""
    return (beta + 1.0) * (R_E + r_e)


@formel("Kollektorschaltung Spannungsverstaerkung", "Vu = R_E / (R_E + r_e)", {"R_E": "ohm", "r_e": "ohm"}, "1", K11,
        dict(R_E=1e3, r_e=13.0))
def kollektorschaltung_vu(R_E, r_e):
    return R_E / (R_E + r_e)


@formel("Kollektorschaltung Ausgangswiderstand", "r_aus = R_E || (r_e + R_G/(B+1))",
        {"R_E": "ohm", "r_e": "ohm", "R_G": "ohm", "beta": "1"}, "ohm", K11,
        dict(R_E=1e3, r_e=13.0, R_G=10e3, beta=200.0))
def kollektorschaltung_rout(R_E, r_e, R_G, beta):
    return widerstand_parallel2(R_E, r_e + R_G / (beta + 1.0))


@formel("Basisschaltung Spannungsverstaerkung", "Vu = R_C / r_e", {"R_C": "ohm", "r_e": "ohm"}, "1", K11,
        dict(R_C=4.7e3, r_e=26.0))
def basisschaltung_vu(R_C, r_e):
    """Eingangswiderstand der Basisschaltung ~ r_e (sehr klein), Ausgangswiderstand ~ R_C."""
    return R_C / r_e


@formel("Koppelkondensator (untere Grenzfrequenz)", "C = 1 / (2 pi f_u R)", {"f_u": "Hz", "R": "ohm"}, "F", K11,
        dict(f_u=20.0, R=10e3))
def koppelkondensator(f_u, R):
    """R: Summe aus Quellen- und Eingangswiderstand (bzw. Ausgangs- und Lastwiderstand)."""
    return 1.0 / (2.0 * PI * f_u * R)


@formel("Transistorschalter: Kollektorstrom", "I_C = (U_CC - U_CEsat) / R_C", {"U_CC": "V", "U_CEsat": "V",
        "R_C": "ohm"}, "A", K11, dict(U_CC=12.0, U_CEsat=0.2, R_C=120.0))
def schalter_ic(U_CC, U_CEsat, R_C):
    return (U_CC - U_CEsat) / R_C


@formel("Transistorschalter: Basiswiderstand", "R_B = (U_e - U_BE) / (ue I_C / B)",
        {"U_e": "V", "U_BE": "V", "I_C": "A", "beta": "1", "ue": "1"}, "ohm", K11,
        dict(U_e=5.0, U_BE=0.7, I_C=0.1, beta=100.0, ue=3.0))
def schalter_rb(U_e, U_BE, I_C, beta, ue):
    """ue: Uebersteuerungsfaktor (typ. 2...5) fuer sichere Saettigung."""
    return (U_e - U_BE) / (ue * I_C / beta)


@formel("Schaltzeit induktive Last", "t = L I / U", {"L": "H", "I": "A", "U": "V"}, "s", K11,
        dict(L=0.1, I=0.5, U=12.0))
def schaltzeit_induktiv(L, I, U):
    """Zeit fuer Stromauf-/-abbau bei konstanter Spannung an L (z. B. Freilauf-/Klemmspannung)."""
    return L * I / U


@formel("Umladeverluste kapazitive Last", "P = C U^2 f", {"C": "F", "U": "V", "f": "Hz"}, "W", K11,
        dict(C=1e-9, U=12.0, f=100e3))
def umladeleistung(C, U, f):
    """Pro Lade-/Entladezyklus wird C U^2 im Schaltwiderstand umgesetzt (auch CMOS: P = C U^2 f)."""
    return C * U ** 2 * f


@formel("Fan-Out", "N = I_aus / I_ein", {"I_aus": "A", "I_ein": "A"}, "1", K11, dict(I_aus=16e-3, I_ein=1.6e-3))
def fan_out(I_aus, I_ein):
    """Fuer H- und L-Pegel getrennt bilden, der kleinere Wert gilt (abrunden)."""
    return I_aus / I_ein


@formel("Mittlere Gatterlaufzeit", "t_pd = (t_PHL + t_PLH) / 2", {"t_PHL": "s", "t_PLH": "s"}, "s", K11,
        dict(t_PHL=8e-9, t_PLH=12e-9))
def gatterlaufzeit(t_PHL, t_PLH):
    return (t_PHL + t_PLH) / 2.0


@formel("FET Drainstrom ohmscher Bereich", "I_D = k ((U_GS - U_th) U_DS - U_DS^2/2)",
        {"k": "A/V^2", "U_GS": "V", "U_th": "V", "U_DS": "V"}, "A", K11, dict(k=2e-3, U_GS=3.0, U_th=1.0, U_DS=0.5))
def fet_id_ohmsch(k, U_GS, U_th, U_DS):
    """Gilt fuer 0 <= U_DS <= U_GS - U_th (sonst mosfet_id_saettigung)."""
    return k * ((U_GS - U_th) * U_DS - U_DS ** 2 / 2.0)


@formel("FET Saettigungsgrenze", "U_DS,sat = U_GS - U_th", {"U_GS": "V", "U_th": "V"}, "V", K11,
        dict(U_GS=3.0, U_th=1.0))
def fet_uds_sat(U_GS, U_th):
    return U_GS - U_th


@formel("FET Steilheit", "g_m = dI_D/dU_GS = k (U_GS - U_th)", {"k": "A/V^2", "U_GS": "V", "U_th": "V"}, "S", K11,
        dict(k=2e-3, U_GS=3.0, U_th=1.0))
def fet_gm(k, U_GS, U_th):
    return k * (U_GS - U_th)


@formel("FET Ausgangswiderstand", "r_ds = 1 / (lambda I_D)", {"lam": "1/V", "I_D": "A"}, "ohm", K11,
        dict(lam=0.02, I_D=4e-3))
def fet_rds(lam, I_D):
    return 1.0 / (lam * I_D)


@formel("FET Prozessparameter", "k = mu_n C_ox W / L", {"mu": "m^2/V/s", "C_ox": "F/m^2", "W": "m", "L": "m"}, "A/V^2",
        K11, dict(mu=0.05, C_ox=3.45e-3, W=10e-6, L=1e-6))
def fet_k(mu, C_ox, W, L):
    return mu * C_ox * W / L


@formel("Sourceschaltung Spannungsverstaerkung", "Vu = -g_m (R_D || r_ds)", {"g_m": "S", "R_D": "ohm", "r_ds": "ohm"},
        "1", K11, dict(g_m=4e-3, R_D=4.7e3, r_ds=12.5e3))
def sourceschaltung_vu(g_m, R_D, r_ds):
    return -g_m * widerstand_parallel2(R_D, r_ds)


@formel("Drainschaltung Spannungsverstaerkung", "Vu = g_m R_S / (1 + g_m R_S)", {"g_m": "S", "R_S": "ohm"}, "1", K11,
        dict(g_m=4e-3, R_S=1e3))
def drainschaltung_vu(g_m, R_S):
    """Ausgangswiderstand ~ R_S || 1/g_m."""
    return g_m * R_S / (1.0 + g_m * R_S)


@formel("FET Restspannung (durchgesteuert)", "U_DS = I_D R_DS(on)", {"I_D": "A", "R_DSon": "ohm"}, "V", K11,
        dict(I_D=5.0, R_DSon=0.02))
def fet_restspannung(I_D, R_DSon):
    return I_D * R_DSon


@formel("FET Durchlassverluste", "P = I_D^2 R_DS(on)", {"I_D": "A", "R_DSon": "ohm"}, "W", K11,
        dict(I_D=5.0, R_DSon=0.02))
def fet_leitverluste(I_D, R_DSon):
    """I_D als Effektivwert bei getaktetem Betrieb."""
    return I_D ** 2 * R_DSon


@formel("FET Schaltverluste", "P = 1/2 U_DS I_D f (t_r + t_f)", {"U_DS": "V", "I_D": "A", "f": "Hz", "t_r": "s",
        "t_f": "s"}, "W", K11, dict(U_DS=48.0, I_D=5.0, f=100e3, t_r=20e-9, t_f=30e-9))
def fet_schaltverluste(U_DS, I_D, f, t_r, t_f):
    return 0.5 * U_DS * I_D * f * (t_r + t_f)


@formel("Koppelfaktor Optokoppler (CTR)", "CTR = I_C / I_F", {"I_C": "A", "I_F": "A"}, "1", K11,
        dict(I_C=5e-3, I_F=10e-3))
def ctr(I_C, I_F):
    """x 100 fuer Angabe in %."""
    return I_C / I_F


@formel("Fotostrom aus Empfindlichkeit", "I_ph = S P_opt", {"S": "A/W", "P_opt": "W"}, "A", K11,
        dict(S=0.5, P_opt=1e-3))
def fotostrom(S, P_opt):
    """S: spektrale Empfindlichkeit (Fotodiode ~0,5 A/W; Fototransistor: zusaetzlich x B)."""
    return S * P_opt


@formel("Fotostrom aus Quantenwirkungsgrad", "I_ph = eta e P_opt / (h f)", {"eta": "1", "P_opt": "W", "f": "Hz"}, "A",
        K11, dict(eta=0.8, P_opt=1e-3, f=3.53e14))
def fotostrom_quanten(eta, P_opt, f):
    return eta * E_ELEM * P_opt / (H_PLANCK * f)


@formel("Hall-Konstante", "R_H = 1 / (n e)", {"n": "m^-3"}, "m^3/C", K11, dict(n=1e22))
def hall_konstante(n):
    return 1.0 / (n * E_ELEM)


@formel("Flussdichte aus Hallspannung", "B = U_H n e d / I", {"U_H": "V", "n": "m^-3", "d": "m", "I": "A"}, "T", K11,
        dict(U_H=0.05, n=1e22, d=1e-4, I=0.01))
def hall_flussdichte(U_H, n, d, I):
    return U_H * n * E_ELEM * d / I


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


# --- Geschwindigkeit / weitere Temperaturen -------------------------------
@formel("km/h -> m/s", "v = v_kmh / 3,6", {"v": "km/h"}, "m/s", K14, dict(v=36.0), dimcheck=False)
def kmh_zu_ms(v):
    return np.asarray(v) / 3.6


@formel("m/s -> km/h", "v_kmh = 3,6 v", {"v": "m/s"}, "km/h", K14, dict(v=10.0), dimcheck=False)
def ms_zu_kmh(v):
    return np.asarray(v) * 3.6


@formel("Kelvin -> Fahrenheit", "F = 9/5 (T - 273,15) + 32", {"T": "K"}, "degF", K14, dict(T=373.15), dimcheck=False)
def kelvin_zu_fahrenheit(T):
    return celsius_zu_fahrenheit(kelvin_zu_celsius(T))


@formel("Fahrenheit -> Kelvin", "T = 5/9 (F - 32) + 273,15", {"F": "degF"}, "K", K14, dict(F=212.0), dimcheck=False)
def fahrenheit_zu_kelvin(F):
    return celsius_zu_kelvin(fahrenheit_zu_celsius(F))


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
# 15. NICHTLINEARE WIDERSTAENDE (NTC, PTC, VDR, LDR, Feldplatte)
# ===========================================================================
K15 = "Nichtlineare Widerstaende"


@formel("NTC: Widerstand bei Temperatur (B-Wert)", "R = R_N exp(B (1/T - 1/T_N))",
        {"T": "K", "R_N": "ohm", "B": "K", "T_N": "K"}, "ohm", K15, dict(T=323.15, R_N=10e3, B=3950.0, T_N=298.15))
def ntc_widerstand(T, R_N, B, T_N):
    """R_N: Nennwiderstand bei T_N (meist 25 °C = 298,15 K)."""
    return R_N * np.exp(B * (1.0 / T - 1.0 / T_N))


@formel("NTC: B-Wert aus zwei Messpunkten", "B = T1 T2 / (T2 - T1) ln(R1 / R2)",
        {"T1": "K", "R1": "ohm", "T2": "K", "R2": "ohm"}, "K", K15, dict(T1=298.15, R1=10e3, T2=358.15, R2=1.07e3))
def ntc_b_wert(T1, R1, T2, R2):
    return T1 * T2 / (T2 - T1) * np.log(R1 / R2)


@formel("NTC: Temperatur aus Widerstand", "T = 1 / (ln(R/R_N)/B + 1/T_N)",
        {"R": "ohm", "R_N": "ohm", "B": "K", "T_N": "K"}, "K", K15, dict(R=3.6e3, R_N=10e3, B=3950.0, T_N=298.15))
def ntc_temperatur(R, R_N, B, T_N):
    return 1.0 / (np.log(R / R_N) / B + 1.0 / T_N)


@formel("NTC: Temperaturkoeffizient", "alpha = -B / T^2", {"B": "K", "T": "K"}, "1/K", K15, dict(B=3950.0, T=298.15))
def ntc_tk(B, T):
    return -B / T ** 2


@formel("Eigenerwaermung (NTC/PTC)", "dT = P / delta_th", {"P": "W", "delta_th": "W/K"}, "K", K15,
        dict(P=10e-3, delta_th=1.5e-3))
def eigenerwaermung(P, delta_th):
    """delta_th: Waermeleitwert / Dissipationskonstante aus dem Datenblatt (mW/K)."""
    return P / delta_th


@formel("PTC/Kaltleiter: linearer Bereich", "R = R0 (1 + alpha (T - T0))", {"R0": "ohm", "alpha": "1/K", "T": "K",
        "T0": "K"}, "ohm", K15, dict(R0=100.0, alpha=0.00385, T=373.15, T0=273.15))
def ptc_widerstand_linear(R0, alpha, T, T0):
    """Metall-PTC (Pt100: alpha = 0,00385 1/K). Fuer T in °C gleiche Formel mit T - T0 als Differenz."""
    return R0 * (1.0 + alpha * (T - T0))


@formel("PTC/Kaltleiter: steiler Bereich", "R = R_ref exp(alpha (T - T_ref))",
        {"R_ref": "ohm", "alpha": "1/K", "T": "K", "T_ref": "K"}, "ohm", K15,
        dict(R_ref=100.0, alpha=0.15, T=403.15, T_ref=393.15))
def ptc_widerstand_exp(R_ref, alpha, T, T_ref):
    """Keramik-PTC oberhalb der Bezugstemperatur (Curie-Punkt): exponentieller Anstieg."""
    return R_ref * np.exp(alpha * (T - T_ref))


@formel("VDR/Varistor: Strom", "I = I_ref (U / U_ref)^alpha", {"U": "V", "U_ref": "V", "I_ref": "A", "alpha": "1"}, "A",
        K15, dict(U=300.0, U_ref=270.0, I_ref=1e-3, alpha=30.0))
def vdr_strom(U, U_ref, I_ref, alpha):
    """Kennlinie I = K U^alpha mit Bezugspunkt (U_ref, I_ref), z. B. Varistorspannung bei 1 mA."""
    return I_ref * (np.abs(U) / U_ref) ** alpha * np.sign(U)


@formel("VDR/Varistor: Spannung", "U = U_ref (I / I_ref)^(1/alpha)", {"I": "A", "U_ref": "V", "I_ref": "A",
        "alpha": "1"}, "V", K15, dict(I=10.0, U_ref=270.0, I_ref=1e-3, alpha=30.0))
def vdr_spannung(I, U_ref, I_ref, alpha):
    return U_ref * (np.abs(I) / I_ref) ** (1.0 / alpha) * np.sign(I)


@formel("VDR/Varistor: Widerstand", "R = U / I = (U_ref/I_ref) (U/U_ref)^(1-alpha)",
        {"U": "V", "U_ref": "V", "I_ref": "A", "alpha": "1"}, "ohm", K15,
        dict(U=300.0, U_ref=270.0, I_ref=1e-3, alpha=30.0))
def vdr_widerstand(U, U_ref, I_ref, alpha):
    """Entspricht der Form R = R0 (U/U0)^n mit R0 = U_ref/I_ref und n = 1 - alpha."""
    return U / vdr_strom(U, U_ref, I_ref, alpha)


@formel("VDR/Varistor: Nichtlinearitaetsexponent", "alpha = ln(I2/I1) / ln(U2/U1)",
        {"U1": "V", "I1": "A", "U2": "V", "I2": "A"}, "1", K15, dict(U1=270.0, I1=1e-3, U2=350.0, I2=1.0))
def vdr_alpha(U1, I1, U2, I2):
    return np.log(I2 / I1) / np.log(U2 / U1)


@formel("LDR: Widerstand aus Beleuchtungsstaerke", "R = R_ref (E_ref / E)^gamma",
        {"E": "lx", "R_ref": "ohm", "E_ref": "lx", "gamma": "1"}, "ohm", K15,
        dict(E=100.0, R_ref=10e3, E_ref=10.0, gamma=0.7), dimcheck=False)
def ldr_widerstand(E, R_ref, E_ref, gamma):
    """E und E_ref in gleicher Einheit (lx); gamma typ. 0,5...1. Dunkelwiderstand: E_ref ~ 0 -> Datenblatt."""
    return R_ref * (E_ref / E) ** gamma


@formel("LDR: Beleuchtungsstaerke aus Widerstand", "E = E_ref (R_ref / R)^(1/gamma)",
        {"R": "ohm", "R_ref": "ohm", "E_ref": "lx", "gamma": "1"}, "lx", K15,
        dict(R=2e3, R_ref=10e3, E_ref=10.0, gamma=0.7), dimcheck=False)
def ldr_beleuchtungsstaerke(R, R_ref, E_ref, gamma):
    return E_ref * (R_ref / R) ** (1.0 / gamma)


@formel("LDR: gamma aus zwei Messpunkten", "gamma = ln(R1/R2) / ln(E2/E1)",
        {"E1": "lx", "R1": "ohm", "E2": "lx", "R2": "ohm"}, "1", K15,
        dict(E1=10.0, R1=10e3, E2=100.0, R2=2e3), dimcheck=False)
def ldr_gamma(E1, R1, E2, R2):
    return np.log(R1 / R2) / np.log(E2 / E1)


@formel("Feldplatte (magnetfeldabh. Widerstand)", "R_B = R_0 (1 + (mu B)^2)", {"B": "T", "R_0": "ohm", "mu": "m^2/V/s"},
        "ohm", K15, dict(B=0.5, R_0=100.0, mu=7.7))
def feldplatte_widerstand(B, R_0, mu):
    """Gauss-Effekt (InSb, mu ~ 7,7 m^2/Vs), Naeherung fuer kleine bis mittlere Flussdichten;
    fuer grosse B steigt R_B/R_0 etwa linear. Genaue Kennlinie -> Datenblatt."""
    return R_0 * (1.0 + (mu * B) ** 2)


# ===========================================================================
# 16. MESSTECHNIK
# ===========================================================================
K16 = "Messtechnik"


@formel("Vorwiderstand Spannungsmesser", "R_V = (U_max - U_m) / I_m", {"U_max": "V", "U_m": "V", "I_m": "A"}, "ohm",
        K16, dict(U_max=10.0, U_m=0.1, I_m=1e-3))
def vorwiderstand_messbereich(U_max, U_m, I_m):
    """U_m = I_m R_m: Spannung am Messwerk bei Vollausschlag."""
    return (U_max - U_m) / I_m


@formel("Nebenwiderstand (Shunt) Strommesser", "R_P = R_m I_m / (I_max - I_m)", {"I_max": "A", "I_m": "A", "R_m": "ohm"},
        "ohm", K16, dict(I_max=1.0, I_m=1e-3, R_m=100.0))
def shuntwiderstand(I_max, I_m, R_m):
    return R_m * I_m / (I_max - I_m)


@formel("Widerstandsmessung spannungsrichtig", "R_x = U / (I - U/R_iV)", {"U": "V", "I": "A", "R_iV": "ohm"}, "ohm",
        K16, dict(U=10.0, I=0.0101, R_iV=1e5))
def widerstand_spannungsrichtig(U, I, R_iV):
    """Voltmeter direkt an R_x, Amperemeter misst Voltmeterstrom mit (fuer kleine R_x)."""
    return U / (I - U / R_iV)


@formel("Widerstandsmessung stromrichtig", "R_x = U / I - R_iA", {"U": "V", "I": "A", "R_iA": "ohm"}, "ohm", K16,
        dict(U=10.0, I=0.01, R_iA=1.0))
def widerstand_stromrichtig(U, I, R_iA):
    """Voltmeter misst Spannungsfall am Amperemeter mit (fuer grosse R_x)."""
    return U / I - R_iA


@formel("Leistung aus Zaehlerumdrehungen", "P = n / (c_z t)", {"n": "1", "c_z": "1/J", "t": "s"}, "W", K16,
        dict(n=30.0, c_z=75.0 / 3.6e6, t=60.0))
def zaehler_leistung(n, c_z, t):
    """c_z: Zaehlerkonstante (Umdrehungen bzw. Impulse je kWh) / 3,6e6 -> je Joule."""
    return n / (c_z * t)


@formel("Absoluter Messfehler", "F = x_mess - x_wahr", {"x_mess": "V", "x_wahr": "V"}, "V", K16,
        dict(x_mess=10.1, x_wahr=10.0))
def messfehler_absolut(x_mess, x_wahr):
    """Fuer jede Groesse verwendbar (Einheit des Ergebnisses = Einheit der Messgroesse)."""
    return x_mess - x_wahr


@formel("Relativer Messfehler", "f = (x_mess - x_wahr) / x_wahr", {"x_mess": "V", "x_wahr": "V"}, "1", K16,
        dict(x_mess=10.1, x_wahr=10.0))
def messfehler_relativ(x_mess, x_wahr):
    """x 100 fuer %."""
    return (x_mess - x_wahr) / x_wahr


@formel("Max. Anzeigefehler (Genauigkeitsklasse)", "F_max = Klasse/100 * Messbereichsendwert",
        {"klasse": "1", "x_end": "V"}, "V", K16, dict(klasse=1.5, x_end=10.0))
def klassenfehler(klasse, x_end):
    """Klasse in % vom Endwert (z. B. 1,5). Relativer Fehler am Messwert: klassenfehler / Anzeige."""
    return klasse / 100.0 * x_end


@formel("Relativer Fehler am Messwert (Klasse)", "f = Klasse/100 * x_end / x_anzeige",
        {"klasse": "1", "x_end": "V", "x_anz": "V"}, "1", K16, dict(klasse=1.5, x_end=10.0, x_anz=2.0))
def klassenfehler_relativ(klasse, x_end, x_anz):
    return klasse / 100.0 * x_end / x_anz


@formel("Gesamtfehler (systematisch + zufaellig)", "F = sqrt(F_s^2 + F_z^2)", {"F_s": "V", "F_z": "V"}, "V", K16,
        dict(F_s=0.03, F_z=0.04))
def gesamtfehler(F_s, F_z):
    return np.sqrt(F_s ** 2 + F_z ** 2)


def mittelwert(x):
    """Arithmetisches Mittel einer Messreihe."""
    return float(np.mean(np.asarray(x, dtype=float)))


def standardabweichung(x):
    """Empirische Standardabweichung s (Divisor n - 1)."""
    return float(np.std(np.asarray(x, dtype=float), ddof=1))


def standardfehler(x):
    """Standardabweichung des Mittelwerts s / sqrt(n)."""
    return standardabweichung(x) / math.sqrt(len(x))


def mittlerer_absoluter_fehler(x, x_wahr):
    """MAE = 1/n sum |x_i - x_wahr|."""
    return float(np.mean(np.abs(np.asarray(x, dtype=float) - x_wahr)))


def fehlerfortpflanzung(f, x, dx, h=1e-6):
    """Gausssche Fehlerfortpflanzung dF = sqrt(sum (df/dx_i dx_i)^2), Ableitungen numerisch (zentral).
    f: Funktion f(*x); x, dx: Listen der Messwerte und ihrer Unsicherheiten.
    Beispiel: fehlerfortpflanzung(ef.widerstand_ohm, [10.0, 0.5], [0.1, 0.005])."""
    x = np.asarray(x, dtype=float)
    summe = 0.0
    for i, (xi, dxi) in enumerate(zip(x, dx)):
        s = h * max(abs(xi), 1.0)
        xp, xm = x.copy(), x.copy()
        xp[i] += s
        xm[i] -= s
        summe += ((f(*xp) - f(*xm)) / (2.0 * s) * dxi) ** 2
    return float(np.sqrt(summe))


def energiekosten(W_kWh, preis_pro_kWh):
    """Kosten = Energie [kWh] x Arbeitspreis (z. B. EUR/kWh). Joule -> kWh: joule_zu_kwh."""
    return W_kWh * preis_pro_kWh


# ===========================================================================
# 17. LEISTUNGSELEKTRONIK (Thyristor, Triac, Steller)
# ===========================================================================
K17 = "Leistungselektronik"


@formel("Steuerwinkel aus Zuendverzoegerung", "alpha = 2 pi f t_z", {"f": "Hz", "t_z": "s"}, "rad", K17,
        dict(f=50.0, t_z=5e-3))
def steuerwinkel(f, t_z):
    """t_z ab Nulldurchgang; Grad = rad_zu_grad(alpha)."""
    return 2.0 * PI * f * t_z


@formel("Phasenanschnitt: Effektivwert (ohmsche Last)", "U = U_eff sqrt(1 - alpha/pi + sin(2 alpha)/(2 pi))",
        {"u_eff": "V", "alpha": "rad"}, "V", K17, dict(u_eff=230.0, alpha=PI / 2))
def phasenanschnitt_effektivwert(u_eff, alpha):
    """Vollwellen-Wechselstromsteller (Triac / antiparallele Thyristoren), Steuerwinkel 0...pi."""
    return u_eff * np.sqrt(np.maximum(1.0 - alpha / PI + np.sin(2.0 * alpha) / (2.0 * PI), 0.0))


@formel("Phasenanschnitt: Leistung (ohmsche Last)", "P = P_max (1 - alpha/pi + sin(2 alpha)/(2 pi))",
        {"P_max": "W", "alpha": "rad"}, "W", K17, dict(P_max=2000.0, alpha=PI / 2))
def phasenanschnitt_leistung(P_max, alpha):
    return P_max * np.maximum(1.0 - alpha / PI + np.sin(2.0 * alpha) / (2.0 * PI), 0.0)


@formel("Gesteuerte Einpuls-Gleichrichtung (M1, ohmsch)", "U_d = Uhat / (2 pi) (1 + cos(alpha))",
        {"u_max": "V", "alpha": "rad"}, "V", K17, dict(u_max=325.0, alpha=PI / 3))
def gesteuert_m1(u_max, alpha):
    return u_max / (2.0 * PI) * (1.0 + np.cos(alpha))


@formel("Gesteuerte Brueckengleichrichtung (B2, ohmsch)", "U_d = Uhat / pi (1 + cos(alpha))",
        {"u_max": "V", "alpha": "rad"}, "V", K17, dict(u_max=325.0, alpha=PI / 3))
def gesteuert_b2(u_max, alpha):
    return u_max / PI * (1.0 + np.cos(alpha))


@formel("Schwingungspaketsteuerung: Leistung", "P = P_max n / N", {"P_max": "W", "n": "1", "N": "1"}, "W", K17,
        dict(P_max=2000.0, n=3.0, N=10.0))
def schwingungspaket_leistung(P_max, n, N):
    """n eingeschaltete Vollschwingungen von N (Vielperioden-/Schwingungspaketsteuerung)."""
    return P_max * n / N


@formel("Schwingungspaketsteuerung: Effektivwert", "U = U_eff sqrt(n / N)", {"u_eff": "V", "n": "1", "N": "1"}, "V",
        K17, dict(u_eff=230.0, n=3.0, N=10.0))
def schwingungspaket_effektivwert(u_eff, n, N):
    return u_eff * np.sqrt(n / N)


@formel("Thyristor: Durchlassspannung", "U_T = U_T0 + r_T I_T", {"U_T0": "V", "r_T": "ohm", "I_T": "A"}, "V", K17,
        dict(U_T0=0.9, r_T=0.01, I_T=20.0))
def thyristor_durchlassspannung(U_T0, r_T, I_T):
    return U_T0 + r_T * I_T


@formel("Thyristor/Diode: Durchlassverluste", "P = U_T0 I_AV + r_T I_RMS^2", {"U_T0": "V", "r_T": "ohm",
        "I_AV": "A", "I_RMS": "A"}, "W", K17, dict(U_T0=0.9, r_T=0.01, I_AV=10.0, I_RMS=15.7))
def thyristor_verlustleistung(U_T0, r_T, I_AV, I_RMS):
    return U_T0 * I_AV + r_T * I_RMS ** 2


@formel("Zuendwiderstand Gate", "R_G = (U_St - U_GT) / I_GT", {"U_St": "V", "U_GT": "V", "I_GT": "A"}, "ohm", K17,
        dict(U_St=5.0, U_GT=1.5, I_GT=20e-3))
def zuendwiderstand(U_St, U_GT, I_GT):
    return (U_St - U_GT) / I_GT


# ===========================================================================
# 18. NETZGERAETE, REGLER, SCHALTREGLER, KUEHLUNG
# ===========================================================================
K18 = "Netzgeraete"


@formel("Einstellbarer Laengsregler (LM317)", "U_a = U_ref (1 + R2/R1) + I_adj R2",
        {"U_ref": "V", "R1": "ohm", "R2": "ohm", "I_adj": "A"}, "V", K18,
        dict(U_ref=1.25, R1=240.0, R2=720.0, I_adj=50e-6))
def laengsregler_ua(U_ref, R1, R2, I_adj):
    """Mit I_adj = 0 auch fuer OPV-Regler / Rueckkopplungsteiler: U_a = U_ref (1 + R2/R1)."""
    return U_ref * (1.0 + R2 / R1) + I_adj * R2


@formel("Verlustleistung Laengsregler", "P = (U_e - U_a) I", {"U_e": "V", "U_a": "V", "I": "A"}, "W", K18,
        dict(U_e=12.0, U_a=5.0, I=0.5))
def laengsregler_verlust(U_e, U_a, I):
    return (U_e - U_a) * I


@formel("Konstantstromquelle mit Regler", "I = U_ref / R", {"U_ref": "V", "R": "ohm"}, "A", K18,
        dict(U_ref=1.25, R=12.5))
def konstantstrom(U_ref, R):
    return U_ref / R


@formel("Tastgrad", "D = t_ein / T", {"t_ein": "s", "T": "s"}, "1", K18, dict(t_ein=2.5e-6, T=10e-6))
def tastgrad(t_ein, T):
    return t_ein / T


@formel("Tiefsetzsteller (Buck): Ausgang", "U_a = D U_e", {"U_e": "V", "D": "1"}, "V", K18, dict(U_e=12.0, D=0.4))
def tiefsetzsteller_ua(U_e, D):
    return D * U_e


@formel("Hochsetzsteller (Boost): Ausgang", "U_a = U_e / (1 - D)", {"U_e": "V", "D": "1"}, "V", K18,
        dict(U_e=5.0, D=0.5))
def hochsetzsteller_ua(U_e, D):
    return U_e / (1.0 - D)


@formel("Inverswandler (Buck-Boost): Ausgang", "|U_a| = U_e D / (1 - D)", {"U_e": "V", "D": "1"}, "V", K18,
        dict(U_e=12.0, D=0.6))
def inverswandler_ua(U_e, D):
    return U_e * D / (1.0 - D)


@formel("Tiefsetzsteller: Induktivitaet", "L = (U_e - U_a) U_a / (U_e f dI)", {"U_e": "V", "U_a": "V", "f": "Hz",
        "dI": "A"}, "H", K18, dict(U_e=12.0, U_a=5.0, f=100e3, dI=0.3))
def tiefsetzsteller_l(U_e, U_a, f, dI):
    """dI: zulaessige Stromwelligkeit (Spitze-Spitze) der Drossel."""
    return (U_e - U_a) * U_a / (U_e * f * dI)


@formel("Tiefsetzsteller: Ausgangskapazitaet", "C = dI / (8 f dU)", {"dI": "A", "f": "Hz", "dU": "V"}, "F", K18,
        dict(dI=0.3, f=100e3, dU=0.01))
def tiefsetzsteller_c(dI, f, dU):
    return dI / (8.0 * f * dU)


@formel("Hochsetzsteller: Induktivitaet", "L = U_e (U_a - U_e) / (U_a f dI)", {"U_e": "V", "U_a": "V", "f": "Hz",
        "dI": "A"}, "H", K18, dict(U_e=5.0, U_a=12.0, f=100e3, dI=0.3))
def hochsetzsteller_l(U_e, U_a, f, dI):
    return U_e * (U_a - U_e) / (U_a * f * dI)


@formel("Hochsetzsteller: Ausgangskapazitaet", "C = I_a D / (f dU)", {"I_a": "A", "D": "1", "f": "Hz", "dU": "V"}, "F",
        K18, dict(I_a=0.5, D=0.58, f=100e3, dU=0.05))
def hochsetzsteller_c(I_a, D, f, dU):
    return I_a * D / (f * dU)


@formel("Netzausregelung (Line Regulation)", "S_U = dU_a / dU_e", {"dU_a": "V", "dU_e": "V"}, "1", K18,
        dict(dU_a=0.005, dU_e=5.0))
def netzausregelung(dU_a, dU_e):
    """x 100 fuer %."""
    return dU_a / dU_e


@formel("Lastausregelung (Load Regulation)", "S_L = (U_leer - U_voll) / U_voll", {"U_leer": "V", "U_voll": "V"}, "1",
        K18, dict(U_leer=5.05, U_voll=5.0))
def lastausregelung(U_leer, U_voll):
    return (U_leer - U_voll) / U_voll


@formel("Sperrschichttemperatur", "T_j = T_u + P R_th", {"T_u": "K", "P": "W", "R_th": "K/W"}, "K", K18,
        dict(T_u=313.15, P=3.5, R_th=20.0))
def sperrschichttemperatur(T_u, P, R_th):
    """R_th: gesamter Waermewiderstand Sperrschicht -> Umgebung (Summe der Teilwiderstaende)."""
    return T_u + P * R_th


@formel("Erforderlicher Kuehlkoerper", "R_thKU = (T_j - T_u)/P - R_thJG - R_thGK",
        {"T_j": "K", "T_u": "K", "P": "W", "R_thJG": "K/W", "R_thGK": "K/W"}, "K/W", K18,
        dict(T_j=398.15, T_u=313.15, P=10.0, R_thJG=1.5, R_thGK=0.5))
def kuehlkoerper_rth(T_j, T_u, P, R_thJG, R_thGK):
    """Datenblattbezeichnung oft R_thJC (Junction-Case) und R_thCS (Case-Sink/Isolierscheibe)."""
    return (T_j - T_u) / P - R_thJG - R_thGK


# ===========================================================================
# 19. KIPPSCHALTUNGEN UND SIGNALGENERATOREN
# ===========================================================================
K19 = "Kippschaltungen"


@formel("Astabiler Multivibrator (Transistoren)", "T = ln(2) (R_B1 C1 + R_B2 C2)",
        {"R1": "ohm", "C1": "F", "R2": "ohm", "C2": "F"}, "s", K19, dict(R1=47e3, C1=10e-6, R2=47e3, C2=10e-6))
def multivibrator_periode(R1, C1, R2, C2):
    """R1, R2: Basiswiderstaende; Naeherung U_CC >> U_BE (exakt: kippzeit_transistor)."""
    return np.log(2.0) * (R1 * C1 + R2 * C2)


@formel("Kippzeit Transistor-Kippstufe (exakt)", "t = R_B C ln((2 U_CC - U_BE - U_CEsat) / (U_CC - U_BE))",
        {"R_B": "ohm", "C": "F", "U_CC": "V", "U_BE": "V", "U_CEsat": "V"}, "s", K19,
        dict(R_B=47e3, C=10e-6, U_CC=5.0, U_BE=0.7, U_CEsat=0.2))
def kippzeit_transistor(R_B, C, U_CC, U_BE, U_CEsat):
    """Halbperiode des astabilen bzw. Impulsdauer des monostabilen Transistor-Kippglieds."""
    return R_B * C * np.log((2.0 * U_CC - U_BE - U_CEsat) / (U_CC - U_BE))


@formel("NE555 astabil", "t_H = ln2 (R_A + R_B) C ; t_L = ln2 R_B C ; f = 1,44 / ((R_A + 2 R_B) C)",
        {"R_A": "ohm", "R_B": "ohm", "C": "F"}, ("s", "s", "Hz"), K19, dict(R_A=1e3, R_B=10e3, C=100e-9))
def ne555_astabil(R_A, R_B, C):
    t_h = np.log(2.0) * (R_A + R_B) * C
    t_l = np.log(2.0) * R_B * C
    return t_h, t_l, 1.0 / (t_h + t_l)


@formel("NE555 monostabil", "T = ln(3) R C = 1,1 R C", {"R": "ohm", "C": "F"}, "s", K19, dict(R=100e3, C=10e-6))
def ne555_monostabil(R, C):
    return np.log(3.0) * R * C


@formel("OPV-Rechteckgenerator (astabil)", "T = 2 R C ln((1 + k)/(1 - k)), k = R1/(R1 + R2)",
        {"R": "ohm", "C": "F", "R1": "ohm", "R2": "ohm"}, "s", K19, dict(R=10e3, C=100e-9, R1=10e3, R2=10e3))
def opv_astabil_periode(R, C, R1, R2):
    """R, C: Umladeglied; R1 (+Eingang->Masse), R2 (Ausgang->+Eingang). R1 = R2: T = 2 R C ln 3."""
    k = R1 / (R1 + R2)
    return 2.0 * R * C * np.log((1.0 + k) / (1.0 - k))


@formel("OPV-Monoflop", "T = R C ln(1 / (1 - k)), k = R1/(R1 + R2)", {"R": "ohm", "C": "F", "R1": "ohm", "R2": "ohm"},
        "s", K19, dict(R=10e3, C=100e-9, R1=20e3, R2=10e3))
def opv_monoflop(R, C, R1, R2):
    """Klemmdiode an C vernachlaessigt (U_D << U_sat). k = 2/3 -> T = R C ln 3."""
    k = R1 / (R1 + R2)
    return R * C * np.log(1.0 / (1.0 - k))


@formel("Schmitt-Trigger invertierend (OPV)", "U_ein/aus = U_ref R2/(R1+R2) +- U_sat R1/(R1+R2)",
        {"U_sat": "V", "R1": "ohm", "R2": "ohm", "U_ref": "V"}, ("V", "V"), K19,
        dict(U_sat=12.0, R1=10e3, R2=40e3, U_ref=0.0))
def schmitt_invertierend(U_sat, R1, R2, U_ref):
    """R2: Ausgang -> +Eingang (Mitkopplung), R1: +Eingang -> U_ref. Rueckgabe (oberer, unterer Schwellwert);
    Hysterese = Differenz = 2 U_sat R1/(R1+R2)."""
    mitte = U_ref * R2 / (R1 + R2)
    d = U_sat * R1 / (R1 + R2)
    return mitte + d, mitte - d


@formel("Schmitt-Trigger nichtinvertierend (OPV)", "U_ein/aus = +- U_sat R1 / R2", {"U_sat": "V", "R1": "ohm",
        "R2": "ohm"}, ("V", "V"), K19, dict(U_sat=12.0, R1=10e3, R2=40e3))
def schmitt_nichtinvertierend(U_sat, R1, R2):
    """R1: Signal -> +Eingang, R2: Ausgang -> +Eingang, -Eingang an Masse. Hysterese = 2 U_sat R1/R2."""
    return U_sat * R1 / R2, -U_sat * R1 / R2


@formel("Schaltschwellen aus Mitte und Hysterese", "U_ein/aus = U_mitte +- U_H / 2", {"U_mitte": "V", "U_H": "V"},
        ("V", "V"), K19, dict(U_mitte=2.5, U_H=0.5))
def schaltschwellen(U_mitte, U_H):
    return U_mitte + U_H / 2.0, U_mitte - U_H / 2.0


@formel("Saegezahn mit Konstantstrom: Spannungsanstieg", "dU = I dt / C", {"I": "A", "dt": "s", "C": "F"}, "V", K19,
        dict(I=1e-3, dt=1e-3, C=1e-6))
def saegezahn_anstieg(I, dt, C):
    return I * dt / C


@formel("Saegezahn mit Konstantstrom: Frequenz", "f = I / (C dU)", {"I": "A", "C": "F", "dU": "V"}, "Hz", K19,
        dict(I=1e-3, C=1e-6, dU=5.0))
def saegezahn_frequenz(I, C, dU):
    """Ruecklaufzeit vernachlaessigt."""
    return I / (C * dU)


@formel("Steilheit Rampe", "S = A / t_an", {"A": "V", "t_an": "s"}, "V/s", K19, dict(A=5.0, t_an=1e-3))
def rampensteilheit(A, t_an):
    return A / t_an


@formel("UJT-Saegezahngenerator", "T = R C ln(1 / (1 - eta))", {"R": "ohm", "C": "F", "eta": "1"}, "s", K19,
        dict(R=10e3, C=100e-9, eta=0.63))
def ujt_periode(R, C, eta):
    """eta: inneres Spannungsverhaeltnis; Talspannung und Diodenspannung vernachlaessigt."""
    return R * C * np.log(1.0 / (1.0 - eta))


@formel("UJT-Hoeckerspannung", "U_P = eta U_BB + U_D", {"eta": "1", "U_BB": "V", "U_D": "V"}, "V", K19,
        dict(eta=0.63, U_BB=12.0, U_D=0.6))
def ujt_hoeckerspannung(eta, U_BB, U_D):
    return eta * U_BB + U_D


@formel("RC-Phasenschiebergenerator (3 Hochpaesse)", "f = 1 / (2 pi R C sqrt(6)), V >= 29", {"R": "ohm", "C": "F"},
        "Hz", K19, dict(R=10e3, C=10e-9))
def phasenschieber_frequenz(R, C):
    """Bei drei RC-Tiefpaessen (R und C vertauscht): f = sqrt(6) / (2 pi R C)."""
    return 1.0 / (2.0 * PI * R * C * np.sqrt(6.0))


@formel("Wien-Brueckengenerator", "f = 1 / (2 pi sqrt(R1 R2 C1 C2)), V = 3 (bei R1=R2, C1=C2)",
        {"R1": "ohm", "R2": "ohm", "C1": "F", "C2": "F"}, "Hz", K19, dict(R1=10e3, R2=10e3, C1=10e-9, C2=10e-9))
def wien_frequenz(R1, R2, C1, C2):
    return 1.0 / (2.0 * PI * np.sqrt(R1 * R2 * C1 * C2))


# ===========================================================================
# 20. MECHANIK (Kraft, Bewegung, Arbeit, Energie, Leistung)
# ===========================================================================
K20 = "Mechanik"


@formel("Grundgleichung der Mechanik", "F = m a", {"m": "kg", "a": "m/s^2"}, "N", K20, dict(m=10.0, a=2.0))
def kraft(m, a):
    return m * a


@formel("Gewichtskraft", "F_G = m g", {"m": "kg"}, "N", K20, dict(m=10.0))
def gewichtskraft(m):
    """g = 9,80665 m/s^2 (Normfallbeschleunigung)."""
    return m * G_ERDE


@formel("Reibungskraft", "F_R = mu F_N", {"mu": "1", "F_N": "N"}, "N", K20, dict(mu=0.3, F_N=100.0))
def reibungskraft(mu, F_N):
    return mu * F_N


@formel("Federkraft (Hooke, Betrag)", "F = k x", {"k": "N/m", "x": "m"}, "N", K20, dict(k=200.0, x=0.05))
def federkraft(k, x):
    """Rueckstellkraft wirkt entgegen der Auslenkung (F = -k x)."""
    return k * x


@formel("Zentripetalkraft", "F_Z = m v^2 / r", {"m": "kg", "v": "m/s", "r": "m"}, "N", K20, dict(m=1.0, v=10.0, r=5.0))
def zentripetalkraft(m, v, r):
    return m * v ** 2 / r


@formel("Zentripetalbeschleunigung", "a_Z = v^2 / r = omega^2 r", {"v": "m/s", "r": "m"}, "m/s^2", K20,
        dict(v=10.0, r=5.0))
def zentripetalbeschleunigung(v, r):
    return v ** 2 / r


@formel("Kraft aus Impulsaenderung", "F = dp / dt", {"dp": "kg*m/s", "dt": "s"}, "N", K20, dict(dp=10.0, dt=0.1))
def kraft_impuls(dp, dt):
    return dp / dt


@formel("Impuls", "p = m v", {"m": "kg", "v": "m/s"}, "kg*m/s", K20, dict(m=2.0, v=3.0))
def impuls(m, v):
    return m * v


@formel("Gravitationskraft", "F = G m1 m2 / r^2", {"m1": "kg", "m2": "kg", "r": "m"}, "N", K20,
        dict(m1=5.972e24, m2=1.0, r=6.371e6))
def gravitationskraft(m1, m2, r):
    return GRAV * m1 * m2 / r ** 2


@formel("Auftriebskraft", "F_A = rho V g", {"rho": "kg/m^3", "V": "m^3"}, "N", K20, dict(rho=1000.0, V=1e-3))
def auftriebskraft(rho, V):
    return rho * V * G_ERDE


@formel("Druck", "p = F / A", {"F": "N", "A": "m^2"}, "N/m^2", K20, dict(F=100.0, A=0.01))
def druck(F, A):
    return F / A


@formel("Kraftzerlegung", "F_x = F cos(alpha) ; F_y = F sin(alpha)", {"F": "N", "alpha": "rad"}, ("N", "N"), K20,
        dict(F=100.0, alpha=PI / 6))
def kraftzerlegung(F, alpha):
    return F * np.cos(alpha), F * np.sin(alpha)


@formel("Resultierende zweier senkrechter Kraefte", "F = sqrt(F_x^2 + F_y^2)", {"F_x": "N", "F_y": "N"}, "N", K20,
        dict(F_x=30.0, F_y=40.0))
def kraft_resultierende(F_x, F_y):
    return np.sqrt(F_x ** 2 + F_y ** 2)


def kraefte_addieren(*F):
    """Vektorielle Addition beliebig vieler Kraefte (je (Fx, Fy) oder (Fx, Fy, Fz)) -> resultierender Vektor."""
    return np.sum(np.asarray(F, dtype=float), axis=0)


def schwerpunkt(massen, orte):
    """r_S = sum(m_i r_i) / sum(m_i); orte als Liste von Koordinaten (1D/2D/3D)."""
    m = np.asarray(massen, dtype=float)
    r = np.asarray(orte, dtype=float)
    return np.tensordot(m, r, axes=1) / m.sum()


@formel("Drehmoment aus Kraft und Hebelarm", "M = F r sin(alpha)", {"F": "N", "r": "m", "alpha": "rad"}, "J", K20,
        dict(F=100.0, r=0.3, alpha=PI / 2))
def drehmoment_kraft(F, r, alpha):
    """alpha: Winkel zwischen Kraft und Hebelarm (pi/2 = senkrecht). Einheit N m (= J)."""
    return F * r * np.sin(alpha)


@formel("Hebelgesetz", "F2 = F1 l1 / l2", {"F1": "N", "l1": "m", "l2": "m"}, "N", K20, dict(F1=100.0, l1=0.2, l2=1.0))
def hebel_kraft(F1, l1, l2):
    return F1 * l1 / l2


@formel("Traegheitsmoment Punktmasse", "J = m r^2", {"m": "kg", "r": "m"}, "kg*m^2", K20, dict(m=2.0, r=0.5))
def traegheitsmoment_punkt(m, r):
    return m * r ** 2


@formel("Traegheitsmoment Vollzylinder", "J = 1/2 m r^2", {"m": "kg", "r": "m"}, "kg*m^2", K20, dict(m=2.0, r=0.5))
def traegheitsmoment_zylinder(m, r):
    return 0.5 * m * r ** 2


@formel("Drehmoment aus Winkelbeschleunigung", "M = J alpha", {"J": "kg*m^2", "alpha": "1/s^2"}, "J", K20,
        dict(J=0.5, alpha=10.0))
def drehmoment_traegheit(J, alpha):
    return J * alpha


@formel("Drehimpuls", "L = J omega", {"J": "kg*m^2", "omega": "1/s"}, "J*s", K20, dict(J=0.5, omega=100.0))
def drehimpuls(J, omega):
    return J * omega


@formel("Rotationsenergie", "E = 1/2 J omega^2", {"J": "kg*m^2", "omega": "1/s"}, "J", K20, dict(J=0.5, omega=100.0))
def rotationsenergie(J, omega):
    return 0.5 * J * omega ** 2


@formel("Geschwindigkeit (gleichfoermig)", "v = s / t", {"s": "m", "t": "s"}, "m/s", K20, dict(s=100.0, t=10.0))
def geschwindigkeit(s, t):
    return s / t


@formel("Beschleunigung", "a = dv / dt", {"dv": "m/s", "dt": "s"}, "m/s^2", K20, dict(dv=20.0, dt=4.0))
def beschleunigung(dv, dt):
    return dv / dt


@formel("Geschwindigkeit (gleichm. beschleunigt)", "v = v0 + a t", {"v0": "m/s", "a": "m/s^2", "t": "s"}, "m/s", K20,
        dict(v0=5.0, a=2.0, t=3.0))
def geschwindigkeit_beschleunigt(v0, a, t):
    """Verzoegerung: a negativ."""
    return v0 + a * t


@formel("Weg (gleichm. beschleunigt)", "s = s0 + v0 t + 1/2 a t^2", {"s0": "m", "v0": "m/s", "a": "m/s^2", "t": "s"},
        "m", K20, dict(s0=0.0, v0=5.0, a=2.0, t=3.0))
def weg_beschleunigt(s0, v0, a, t):
    return s0 + v0 * t + 0.5 * a * t ** 2


@formel("Endgeschwindigkeit aus Weg", "v = sqrt(v0^2 + 2 a s)", {"v0": "m/s", "a": "m/s^2", "s": "m"}, "m/s", K20,
        dict(v0=5.0, a=2.0, s=24.0))
def geschwindigkeit_weg(v0, a, s):
    return np.sqrt(v0 ** 2 + 2.0 * a * s)


@formel("Freier Fall: Geschwindigkeit nach Fallhoehe", "v = sqrt(2 g h)", {"h": "m"}, "m/s", K20, dict(h=20.0))
def fallgeschwindigkeit(h):
    """Weitere Fallgesetze: v = g t (geschwindigkeit_beschleunigt mit a = G_ERDE), s = 1/2 g t^2."""
    return np.sqrt(2.0 * G_ERDE * h)


@formel("Freier Fall: Fallzeit", "t = sqrt(2 h / g)", {"h": "m"}, "s", K20, dict(h=20.0))
def fallzeit(h):
    return np.sqrt(2.0 * h / G_ERDE)


@formel("Bahngeschwindigkeit", "v = omega r", {"omega": "1/s", "r": "m"}, "m/s", K20, dict(omega=10.0, r=0.5))
def bahngeschwindigkeit(omega, r):
    return omega * r


@formel("Umlaufgeschwindigkeit", "v = 2 pi r / T", {"r": "m", "T": "s"}, "m/s", K20, dict(r=0.5, T=0.1))
def umlaufgeschwindigkeit(r, T):
    return 2.0 * PI * r / T


@formel("Mechanische Arbeit", "W = F s cos(alpha)", {"F": "N", "s": "m", "alpha": "rad"}, "J", K20,
        dict(F=100.0, s=10.0, alpha=0.0))
def arbeit_mechanisch(F, s, alpha):
    """Reibungsarbeit: F = F_R."""
    return F * s * np.cos(alpha)


@formel("Hubarbeit / potentielle Energie", "W = m g h", {"m": "kg", "h": "m"}, "J", K20, dict(m=10.0, h=5.0))
def hubarbeit(m, h):
    return m * G_ERDE * h


@formel("Kinetische Energie", "W = 1/2 m v^2", {"m": "kg", "v": "m/s"}, "J", K20, dict(m=10.0, v=5.0))
def kinetische_energie(m, v):
    return 0.5 * m * v ** 2


@formel("Spannarbeit Feder", "W = 1/2 k x^2", {"k": "N/m", "x": "m"}, "J", K20, dict(k=200.0, x=0.1))
def spannarbeit(k, x):
    return 0.5 * k * x ** 2


@formel("Leistung aus Arbeit", "P = W / t", {"W": "J", "t": "s"}, "W", K20, dict(W=1000.0, t=10.0))
def leistung_arbeit(W, t):
    return W / t


@formel("Leistung aus Kraft und Geschwindigkeit", "P = F v", {"F": "N", "v": "m/s"}, "W", K20, dict(F=100.0, v=2.0))
def leistung_kraft(F, v):
    return F * v


@formel("Leistung aus Drehmoment", "P = M omega", {"M": "J", "omega": "1/s"}, "W", K20, dict(M=50.0, omega=150.0))
def leistung_drehmoment(M, omega):
    return M * omega


@formel("Hubleistung", "P = m g h / t", {"m": "kg", "h": "m", "t": "s"}, "W", K20, dict(m=100.0, h=10.0, t=20.0))
def hubleistung(m, h, t):
    return m * G_ERDE * h / t


@formel("Schiefe Ebene: Hangabtriebskraft", "F_H = m g sin(alpha)", {"m": "kg", "alpha": "rad"}, "N", K20,
        dict(m=10.0, alpha=PI / 6))
def hangabtriebskraft(m, alpha):
    return m * G_ERDE * np.sin(alpha)


@formel("Schiefe Ebene: Normalkraft", "F_N = m g cos(alpha)", {"m": "kg", "alpha": "rad"}, "N", K20,
        dict(m=10.0, alpha=PI / 6))
def normalkraft(m, alpha):
    return m * G_ERDE * np.cos(alpha)


@formel("Schiefe Ebene: Beschleunigung mit Reibung", "a = g (sin(alpha) - mu cos(alpha))", {"alpha": "rad", "mu": "1"},
        "m/s^2", K20, dict(alpha=PI / 6, mu=0.2))
def beschleunigung_schiefe_ebene(alpha, mu):
    """Unabhaengig von der Masse; Ergebnis <= 0 -> Koerper bleibt (Haftreibung vorausgesetzt) liegen."""
    return G_ERDE * (np.sin(alpha) - mu * np.cos(alpha))


@formel("Dichte", "rho = m / V", {"m": "kg", "V": "m^3"}, "kg/m^3", K20, dict(m=7.85, V=1e-3))
def dichte(m, V):
    return m / V


@formel("Masse aus Dichte", "m = rho V", {"rho": "kg/m^3", "V": "m^3"}, "kg", K20, dict(rho=7850.0, V=1e-3))
def masse(rho, V):
    """Volumen: V = m / rho (dichte umgestellt)."""
    return rho * V


@formel("Dichte bei Temperatur", "rho = rho0 (1 - beta dT)", {"rho0": "kg/m^3", "beta": "1/K", "dT": "K"}, "kg/m^3",
        K20, dict(rho0=1000.0, beta=2.1e-4, dT=20.0))
def dichte_temperatur(rho0, beta, dT):
    return rho0 * (1.0 - beta * dT)


@formel("Gasdichte (ideales Gas)", "rho = p / (R_s T) = p M / (R T)", {"p": "N/m^2", "R_s": "J/kg/K", "T": "K"},
        "kg/m^3", K20, dict(p=101325.0, R_s=287.05, T=288.15))
def gasdichte(p, R_s, T):
    """R_s: spezifische Gaskonstante = R/M (Luft: 287 J/(kg K))."""
    return p / (R_s * T)


@formel("Mittlere Dichte / Mischung (2 Stoffe)", "rho = (m1 + m2) / (V1 + V2)", {"m1": "kg", "V1": "m^3", "m2": "kg",
        "V2": "m^3"}, "kg/m^3", K20, dict(m1=1.0, V1=1e-3, m2=7.85, V2=1e-3))
def dichte_mittel(m1, V1, m2, V2):
    return (m1 + m2) / (V1 + V2)


@formel("Massenanteil", "w = m_Teil / m_Gesamt", {"m_teil": "kg", "m_ges": "kg"}, "1", K20, dict(m_teil=0.6, m_ges=2.0))
def massenanteil(m_teil, m_ges):
    return m_teil / m_ges


# ===========================================================================
# 21. WAERME
# ===========================================================================
K21 = "Waerme"


@formel("Waermemenge", "Q = m c dT", {"m": "kg", "c": "J/kg/K", "dT": "K"}, "J", K21, dict(m=1.0, c=4182.0, dT=80.0))
def waermemenge(m, c, dT):
    return m * c * dT


@formel("Spezifische Waermekapazitaet", "c = Q / (m dT)", {"Q": "J", "m": "kg", "dT": "K"}, "J/kg/K", K21,
        dict(Q=334560.0, m=1.0, dT=80.0))
def spez_waermekapazitaet(Q, m, dT):
    return Q / (m * dT)


@formel("Waermekapazitaet", "C = m c", {"m": "kg", "c": "J/kg/K"}, "J/K", K21, dict(m=1.0, c=4182.0))
def waermekapazitaet(m, c):
    return m * c


@formel("Schmelz-/Verdampfungswaerme", "Q = m L", {"m": "kg", "L": "J/kg"}, "J", K21, dict(m=1.0, L=2.257e6))
def latente_waerme(m, L):
    """L_s (Eis: 334 kJ/kg) bzw. L_v (Wasser: 2257 kJ/kg)."""
    return m * L


@formel("Erster Hauptsatz", "dU = Q - W", {"Q": "J", "W": "J"}, "J", K21, dict(Q=1000.0, W=400.0))
def innere_energie(Q, W):
    return Q - W


@formel("Carnot-Wirkungsgrad", "eta = 1 - T_k / T_w", {"T_k": "K", "T_w": "K"}, "1", K21, dict(T_k=300.0, T_w=600.0))
def wirkungsgrad_carnot(T_k, T_w):
    return 1.0 - T_k / T_w


@formel("Waermestrom durch Wand (Leitung)", "Q' = lambda A dT / d", {"lam": "W/m/K", "A": "m^2", "dT": "K", "d": "m"},
        "W", K21, dict(lam=0.04, A=10.0, dT=20.0, d=0.1))
def waermestrom_leitung(lam, A, dT, d):
    """Fuer die Waermemenge in der Zeit t: Q = Q' t."""
    return lam * A * dT / d


@formel("Waermewiderstand (Leitung)", "R_th = d / (lambda A)", {"d": "m", "lam": "W/m/K", "A": "m^2"}, "K/W", K21,
        dict(d=0.1, lam=0.04, A=10.0))
def waermewiderstand(d, lam, A):
    """In Reihe addieren sich Waermewiderstaende wie elektrische (widerstand_reihe)."""
    return d / (lam * A)


@formel("Waermedurchlasswiderstand (flaechenbezogen)", "R = d / lambda", {"d": "m", "lam": "W/m/K"}, "m^2*K/W", K21,
        dict(d=0.1, lam=0.04))
def waermedurchlasswiderstand(d, lam):
    return d / lam


@formel("Waermestrom aus Temperaturdifferenz", "Q' = dT / R_th", {"dT": "K", "R_th": "K/W"}, "W", K21,
        dict(dT=40.0, R_th=4.0))
def waermestrom_rth(dT, R_th):
    return dT / R_th


@formel("Waermestrom Zylinderwand (Rohr)", "Q' = 2 pi lambda L (T1 - T2) / ln(r2 / r1)",
        {"lam": "W/m/K", "L": "m", "T1": "K", "T2": "K", "r1": "m", "r2": "m"}, "W", K21,
        dict(lam=0.04, L=1.0, T1=353.15, T2=293.15, r1=0.02, r2=0.05))
def waermestrom_zylinder(lam, L, T1, T2, r1, r2):
    return 2.0 * PI * lam * L * (T1 - T2) / np.log(r2 / r1)


@formel("Waermestrom Kugelschale", "Q' = 4 pi lambda (T1 - T2) / (1/r1 - 1/r2)",
        {"lam": "W/m/K", "T1": "K", "T2": "K", "r1": "m", "r2": "m"}, "W", K21,
        dict(lam=0.04, T1=353.15, T2=293.15, r1=0.1, r2=0.15))
def waermestrom_kugel(lam, T1, T2, r1, r2):
    return 4.0 * PI * lam * (T1 - T2) / (1.0 / r1 - 1.0 / r2)


@formel("Waermeuebergang (Konvektion)", "Q' = alpha A (T_s - T_f)", {"alpha": "W/m^2/K", "A": "m^2", "dT": "K"}, "W",
        K21, dict(alpha=10.0, A=0.1, dT=40.0))
def waermestrom_konvektion(alpha, A, dT):
    return alpha * A * dT


@formel("Waermeuebergangskoeffizient", "alpha = Q' / (A dT)", {"Q": "W", "A": "m^2", "dT": "K"}, "W/m^2/K", K21,
        dict(Q=40.0, A=0.1, dT=40.0))
def waermeuebergangskoeffizient(Q, A, dT):
    return Q / (A * dT)


@formel("Kuehlflaeche", "A = Q' / (alpha dT)", {"Q": "W", "alpha": "W/m^2/K", "dT": "K"}, "m^2", K21,
        dict(Q=10.0, alpha=10.0, dT=40.0))
def kuehlflaeche(Q, alpha, dT):
    return Q / (alpha * dT)


@formel("Waermestrahlung (Stefan-Boltzmann)", "Q' = eps sigma A (T^4 - T0^4)", {"eps": "1", "A": "m^2", "T": "K",
        "T0": "K"}, "W", K21, dict(eps=0.9, A=0.1, T=373.15, T0=293.15))
def waermestrom_strahlung(eps, A, T, T0):
    return eps * SIGMA_SB * A * (T ** 4 - T0 ** 4)


@formel("Waermedurchgangskoeffizient (U-Wert)", "k = 1 / (1/alpha1 + d/lambda + 1/alpha2)",
        {"alpha1": "W/m^2/K", "d": "m", "lam": "W/m/K", "alpha2": "W/m^2/K"}, "W/m^2/K", K21,
        dict(alpha1=7.7, d=0.24, lam=0.8, alpha2=25.0))
def waermedurchgangskoeffizient(alpha1, d, lam, alpha2):
    return 1.0 / (1.0 / alpha1 + d / lam + 1.0 / alpha2)


@formel("Waermestrom Waermedurchgang", "Q' = k A dT", {"k": "W/m^2/K", "A": "m^2", "dT": "K"}, "W", K21,
        dict(k=2.0, A=10.0, dT=20.0))
def waermestrom_durchgang(k, A, dT):
    return k * A * dT


@formel("Heizzeit (Elektrowaerme)", "t = m c dT / (eta P)", {"m": "kg", "c": "J/kg/K", "dT": "K", "eta": "1", "P": "W"},
        "s", K21, dict(m=1.5, c=4182.0, dT=80.0, eta=0.9, P=2000.0))
def heizzeit(m, c, dT, eta, P):
    """eta: Waermenutzungsgrad = Q_nutz / Q_zu (wirkungsgrad)."""
    return m * c * dT / (eta * P)


# ===========================================================================
# 22. GEOMETRIE (Laengen, Flaechen, Volumen)
# ===========================================================================
K22 = "Geometrie"


@formel("Kreisumfang", "U = 2 pi r", {"r": "m"}, "m", K22, dict(r=1.0))
def umfang_kreis(r):
    return 2.0 * PI * r


@formel("Rechteckumfang", "U = 2 (a + b)", {"a": "m", "b": "m"}, "m", K22, dict(a=2.0, b=3.0))
def umfang_rechteck(a, b):
    """Quadrat: a = b -> U = 4 a."""
    return 2.0 * (a + b)


@formel("Bogenlaenge", "s = r phi", {"r": "m", "phi": "rad"}, "m", K22, dict(r=2.0, phi=PI / 2))
def bogenlaenge(r, phi):
    return r * phi


@formel("Diagonale Rechteck", "d = sqrt(a^2 + b^2)", {"a": "m", "b": "m"}, "m", K22, dict(a=3.0, b=4.0))
def diagonale_rechteck(a, b):
    """Quadrat: d = a sqrt(2). Allgemein: Pythagoras."""
    return np.sqrt(a ** 2 + b ** 2)


@formel("Abstand zweier Punkte (2D)", "d = sqrt((x2-x1)^2 + (y2-y1)^2)", {"x1": "m", "y1": "m", "x2": "m", "y2": "m"},
        "m", K22, dict(x1=0.0, y1=0.0, x2=3.0, y2=4.0))
def abstand_2d(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)


@formel("Abstand zweier Punkte (3D)", "d = sqrt(dx^2 + dy^2 + dz^2)", {"x1": "m", "y1": "m", "z1": "m", "x2": "m",
        "y2": "m", "z2": "m"}, "m", K22, dict(x1=0.0, y1=0.0, z1=0.0, x2=1.0, y2=2.0, z2=2.0))
def abstand_3d(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


@formel("Flaeche Rechteck", "A = a b", {"a": "m", "b": "m"}, "m^2", K22, dict(a=2.0, b=3.0))
def flaeche_rechteck(a, b):
    """Quadrat: a = b."""
    return a * b


@formel("Flaeche Dreieck", "A = 1/2 g h", {"g": "m", "h": "m"}, "m^2", K22, dict(g=4.0, h=3.0))
def flaeche_dreieck(g, h):
    return 0.5 * g * h


@formel("Flaeche Parallelogramm", "A = g h", {"g": "m", "h": "m"}, "m^2", K22, dict(g=4.0, h=3.0))
def flaeche_parallelogramm(g, h):
    return g * h


@formel("Flaeche Trapez", "A = 1/2 (a + c) h", {"a": "m", "c": "m", "h": "m"}, "m^2", K22, dict(a=4.0, c=2.0, h=3.0))
def flaeche_trapez(a, c, h):
    return 0.5 * (a + c) * h


@formel("Flaeche Kreis", "A = pi r^2", {"r": "m"}, "m^2", K22, dict(r=1.0))
def flaeche_kreis(r):
    """Aus Durchmesser: A = pi d^2 / 4 (durchmesser_aus_querschnitt fuer die Umkehrung)."""
    return PI * r ** 2


@formel("Flaeche Ellipse", "A = pi a b", {"a": "m", "b": "m"}, "m^2", K22, dict(a=2.0, b=1.0))
def flaeche_ellipse(a, b):
    return PI * a * b


@formel("Flaeche Raute", "A = 1/2 e f", {"e": "m", "f": "m"}, "m^2", K22, dict(e=4.0, f=3.0))
def flaeche_raute(e, f):
    return 0.5 * e * f


@formel("Flaeche regelmaessiges n-Eck", "A = n s^2 / (4 tan(pi/n))", {"n": "1", "s": "m"}, "m^2", K22,
        dict(n=6.0, s=1.0))
def flaeche_n_eck(n, s):
    return n * s ** 2 / (4.0 * np.tan(PI / n))


@formel("Volumen Quader", "V = a b c", {"a": "m", "b": "m", "c": "m"}, "m^3", K22, dict(a=1.0, b=2.0, c=3.0))
def volumen_quader(a, b, c):
    """Wuerfel: a = b = c -> V = a^3."""
    return a * b * c


@formel("Volumen Prisma / Zylinder (allg.)", "V = A_G h", {"A_G": "m^2", "h": "m"}, "m^3", K22, dict(A_G=2.0, h=3.0))
def volumen_prisma(A_G, h):
    return A_G * h


@formel("Volumen Pyramide / Kegel (allg.)", "V = 1/3 A_G h", {"A_G": "m^2", "h": "m"}, "m^3", K22, dict(A_G=2.0, h=3.0))
def volumen_pyramide(A_G, h):
    return A_G * h / 3.0


@formel("Volumen Zylinder", "V = pi r^2 h", {"r": "m", "h": "m"}, "m^3", K22, dict(r=1.0, h=2.0))
def volumen_zylinder(r, h):
    return PI * r ** 2 * h


@formel("Volumen Kegel", "V = 1/3 pi r^2 h", {"r": "m", "h": "m"}, "m^3", K22, dict(r=1.0, h=3.0))
def volumen_kegel(r, h):
    return PI * r ** 2 * h / 3.0


@formel("Volumen Kugel", "V = 4/3 pi r^3", {"r": "m"}, "m^3", K22, dict(r=1.0))
def volumen_kugel(r):
    return 4.0 / 3.0 * PI * r ** 3


@formel("Oberflaeche Quader", "O = 2 (ab + ac + bc)", {"a": "m", "b": "m", "c": "m"}, "m^2", K22,
        dict(a=1.0, b=2.0, c=3.0))
def oberflaeche_quader(a, b, c):
    """Wuerfel: O = 6 a^2."""
    return 2.0 * (a * b + a * c + b * c)


@formel("Oberflaeche Kugel", "O = 4 pi r^2", {"r": "m"}, "m^2", K22, dict(r=1.0))
def oberflaeche_kugel(r):
    return 4.0 * PI * r ** 2


@formel("Oberflaeche Zylinder", "O = 2 pi r (r + h)", {"r": "m", "h": "m"}, "m^2", K22, dict(r=1.0, h=2.0))
def oberflaeche_zylinder(r, h):
    """Mantelflaeche allein: M = 2 pi r h."""
    return 2.0 * PI * r * (r + h)


@formel("Mantelflaeche Zylinder", "M = 2 pi r h", {"r": "m", "h": "m"}, "m^2", K22, dict(r=1.0, h=2.0))
def mantel_zylinder(r, h):
    return 2.0 * PI * r * h


@formel("Oberflaeche Kegel", "O = pi r (r + s)", {"r": "m", "s": "m"}, "m^2", K22, dict(r=1.0, s=2.0))
def oberflaeche_kegel(r, s):
    """s: Mantellinie; Mantelflaeche allein: M = pi r s."""
    return PI * r * (r + s)


@formel("Mantelflaeche Kegel", "M = pi r s", {"r": "m", "s": "m"}, "m^2", K22, dict(r=1.0, s=2.0))
def mantel_kegel(r, s):
    return PI * r * s


# ===========================================================================
# 23. DATENUEBERTRAGUNG UND LICHTWELLENLEITER
# ===========================================================================
K23 = "Datenuebertragung"


@formel("Kanalkapazitaet (Shannon)", "C = B log2(1 + S/N)", {"B": "Hz", "snr": "1"}, "1/s", K23,
        dict(B=3100.0, snr=1000.0))
def shannon_kapazitaet(B, snr):
    """snr linear (aus dB: von_db_leistung). Ergebnis in bit/s."""
    return B * np.log2(1.0 + snr)


@formel("Kanalkapazitaet (Nyquist, rauschfrei)", "C = 2 B log2(M)", {"B": "Hz", "M": "1"}, "1/s", K23,
        dict(B=3100.0, M=4.0))
def nyquist_kapazitaet(B, M):
    return 2.0 * B * np.log2(M)


@formel("Schrittgeschwindigkeit (Baudrate)", "v_s = R / log2(M)", {"R": "1/s", "M": "1"}, "1/s", K23,
        dict(R=9600.0, M=16.0))
def baudrate(R, M):
    """R: Bitrate in bit/s, M: Anzahl Signalzustaende. Ergebnis in Baud."""
    return R / np.log2(M)


@formel("Datenrate", "R = D / t", {"D": "1", "t": "s"}, "1/s", K23, dict(D=8e6, t=2.0))
def datenrate(D, t):
    """D: Datenmenge in bit."""
    return D / t


@formel("Uebertragungsdauer / Latenz", "t = D/R + d/v + t_p", {"D": "1", "R": "1/s", "d": "m", "v": "m/s", "t_p": "s"},
        "s", K23, dict(D=8e6, R=1e8, d=1e6, v=2e8, t_p=1e-3))
def uebertragungsdauer(D, R, d, v, t_p):
    """Sendedauer + Ausbreitungsverzoegerung + Verarbeitungszeit."""
    return D / R + d / v + t_p


@formel("Zeitmultiplex: Zeitschlitz", "t_s = T / N", {"T": "s", "N": "1"}, "s", K23, dict(T=125e-6, N=32.0))
def tdm_zeitschlitz(T, N):
    return T / N


@formel("Zeitmultiplex: Gesamtbitrate", "R = N m / T", {"N": "1", "m": "1", "T": "s"}, "1/s", K23,
        dict(N=32.0, m=8.0, T=125e-6))
def tdm_bitrate(N, m, T):
    """N Kanaele mit je m bit pro Rahmen der Dauer T (PCM30: 2,048 Mbit/s)."""
    return N * m / T


@formel("Bitfehlerrate", "BER = N_fehler / N_gesamt", {"N_f": "1", "N_ges": "1"}, "1", K23, dict(N_f=3.0, N_ges=1e6))
def bitfehlerrate(N_f, N_ges):
    return N_f / N_ges


@formel("Wahrscheinlichkeit fehlerfreier Block", "P = (1 - p)^n", {"p": "1", "n": "1"}, "1", K23,
        dict(p=1e-6, n=12000.0))
def p_fehlerfrei(p, n):
    """Mindestens ein Fehler: 1 - p_fehlerfrei."""
    return (1.0 - p) ** n


@formel("Entscheidungsgehalt", "H0 = log2(N)", {"N": "1"}, "1", K23, dict(N=26.0))
def entscheidungsgehalt(N):
    """N: Elementvorrat (Anzahl unterscheidbarer Zeichen); Ergebnis in bit/Zeichen."""
    return np.log2(N)


@formel("Relative Redundanz", "r = (n_ges - n_nutz) / n_ges", {"n_ges": "1", "n_nutz": "1"}, "1", K23,
        dict(n_ges=8.0, n_nutz=7.0))
def redundanz_relativ(n_ges, n_nutz):
    """Auch Code-Effizienz eta = n_nutz / n_ges = 1 - r."""
    return (n_ges - n_nutz) / n_ges


@formel("Verfuegbarkeit", "V = MTBF / (MTBF + MTTR)", {"MTBF": "s", "MTTR": "s"}, "1", K23,
        dict(MTBF=8760.0 * 3600.0, MTTR=8.0 * 3600.0))
def verfuegbarkeit(MTBF, MTTR):
    return MTBF / (MTBF + MTTR)


@formel("Zuverlaessigkeit (Ueberlebenswahrscheinlichkeit)", "R(t) = exp(-lambda t)", {"lam": "1/s", "t": "s"}, "1", K23,
        dict(lam=1e-6, t=3600.0))
def zuverlaessigkeit(lam, t):
    """lam: Ausfallrate (MTBF = 1/lam)."""
    return np.exp(-lam * t)


@formel("FM-Bandbreite (Carson)", "B = 2 (df + f_m)", {"df": "Hz", "f_m": "Hz"}, "Hz", K23, dict(df=75e3, f_m=15e3))
def carson_bandbreite(df, f_m):
    return 2.0 * (df + f_m)


@formel("FM-Modulationsindex", "beta = df / f_m", {"df": "Hz", "f_m": "Hz"}, "1", K23, dict(df=75e3, f_m=15e3))
def modulationsindex_fm(df, f_m):
    return df / f_m


@formel("Brechzahl", "n = c0 / v", {"v": "m/s"}, "1", K23, dict(v=2e8))
def brechzahl(v):
    return C0 / v


@formel("Numerische Apertur (Stufenindexfaser)", "NA = sqrt(n1^2 - n2^2) = n0 sin(theta_max)",
        {"n_kern": "1", "n_mantel": "1"}, "1", K23, dict(n_kern=1.48, n_mantel=1.46))
def numerische_apertur(n_kern, n_mantel):
    """Akzeptanzwinkel: theta_max = asin(NA / n0)."""
    return np.sqrt(n_kern ** 2 - n_mantel ** 2)


@formel("Daempfungskoeffizient LWL", "a = 10 lg(P_ein / P_aus) / L", {"P_ein": "W", "P_aus": "W", "L": "m"}, "1/m",
        K23, dict(P_ein=1e-3, P_aus=0.5e-3, L=10e3))
def daempfung_lwl(P_ein, P_aus, L):
    """Ergebnis in dB/m (x 1000 -> dB/km)."""
    return 10.0 * np.log10(P_ein / P_aus) / L


@formel("Dispersion: Impulsverbreiterung", "dt = D L d_lambda", {"D": "s/m^2", "L": "m", "d_lam": "m"}, "s", K23,
        dict(D=17e-6, L=10e3, d_lam=1e-9))
def dispersion_impulsverbreiterung(D, L, d_lam):
    """D in s/m^2 (17 ps/(nm km) = 17e-6 s/m^2)."""
    return D * L * d_lam


@formel("Bandbreite-Laenge-Produkt", "B = BLP / L", {"BLP": "Hz*m", "L": "m"}, "Hz", K23, dict(BLP=500e6 * 1e3, L=2e3))
def bandbreite_lwl(BLP, L):
    return BLP / L


# ===========================================================================
# 24. DIGITALTECHNIK (Gatter, Zahlensysteme, Dualarithmetik, Schaltalgebra)
#     Ganzzahl-/Logikfunktionen ohne physikalische Einheit -> nicht in der Registry.
# ===========================================================================
def UND(*x):
    """AND (beliebig viele Eingaenge, 0/1, bool oder NumPy-Arrays)."""
    return np.logical_and.reduce([np.asarray(v, dtype=bool) for v in x]).astype(int)


def ODER(*x):
    return np.logical_or.reduce([np.asarray(v, dtype=bool) for v in x]).astype(int)


def NICHT(a):
    return np.logical_not(np.asarray(a, dtype=bool)).astype(int)


def NAND(*x):
    return NICHT(UND(*x))


def NOR(*x):
    return NICHT(ODER(*x))


def XOR(*x):
    """Antivalenz (bei mehr als 2 Eingaengen: ungerade Paritaet)."""
    return np.logical_xor.reduce([np.asarray(v, dtype=bool) for v in x]).astype(int)


def XNOR(*x):
    """Aequivalenz."""
    return NICHT(XOR(*x))


def halbaddierer(a, b):
    """-> (Summe S = a XOR b, Uebertrag C = a AND b)."""
    return XOR(a, b), UND(a, b)


def volladdierer(a, b, c_ein):
    """-> (S = a XOR b XOR c, C_aus = a b + c (a XOR b))."""
    return XOR(a, b, c_ein), ODER(UND(a, b), UND(c_ein, XOR(a, b)))


def wahrheitstabelle(f, n):
    """Liste von (Eingangstupel, Ausgang) fuer eine Funktion f mit n Eingaengen (MSB zuerst)."""
    zeilen = []
    for i in range(2 ** n):
        e = tuple((i >> (n - 1 - k)) & 1 for k in range(n))
        zeilen.append((e, int(f(*e))))
    return zeilen


def minterme(f, n):
    """Indizes der Zeilen mit f = 1 (Grundlage fuer DNF und KV-Tafel)."""
    return [i for i, (_, y) in enumerate(wahrheitstabelle(f, n)) if y]


def maxterme(f, n):
    """Indizes der Zeilen mit f = 0 (Grundlage fuer KNF)."""
    return [i for i, (_, y) in enumerate(wahrheitstabelle(f, n)) if not y]


def dnf(f, namen):
    """Disjunktive Normalform als Text, z. B. dnf(lambda a, b: a ^ b, 'AB') -> "/A B + A /B"."""
    n = len(namen)
    terme = [" ".join(v if (i >> (n - 1 - k)) & 1 else "/" + v for k, v in enumerate(namen)) for i in minterme(f, n)]
    return " + ".join(terme) if terme else "0"


def knf(f, namen):
    """Konjunktive Normalform als Text."""
    n = len(namen)
    terme = ["(" + " + ".join("/" + v if (i >> (n - 1 - k)) & 1 else v for k, v in enumerate(namen)) + ")"
             for i in maxterme(f, n)]
    return " ".join(terme) if terme else "1"


def aequivalent(f, g, n):
    """True, wenn zwei Schaltfunktionen fuer alle 2^n Belegungen gleich sind
    (prueft Gesetze der Schaltalgebra, z. B. De Morgan oder eine KV-Vereinfachung)."""
    return all(int(f(*e)) == int(g(*e)) for e, _ in wahrheitstabelle(f, n))


_ZIFFERN = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def dezimal_zu_basis(x, basis=2, nachkomma=0):
    """Dezimalzahl -> Zahl zur Basis 2...36 als Text (Restwertmethode; Nachkommastellen per Multiplikation)."""
    vz = "-" if x < 0 else ""
    x = abs(x)
    ganz, frac = int(x), x - int(x)
    s = ""
    while True:
        ganz, rest = divmod(ganz, basis)
        s = _ZIFFERN[rest] + s
        if ganz == 0:
            break
    if nachkomma:
        s += "."
        for _ in range(nachkomma):
            frac *= basis
            s += _ZIFFERN[int(frac)]
            frac -= int(frac)
    return vz + s


def basis_zu_dezimal(s, basis=2):
    """Text zur Basis 2...36 (optional mit '.') -> Dezimalzahl: sum d_i * basis^i."""
    s = s.strip().upper().replace(" ", "").replace("_", "")
    vz = -1 if s.startswith("-") else 1
    s = s.lstrip("+-")
    ganz, _, frac = s.partition(".")
    wert = sum(_ZIFFERN.index(z) * basis ** i for i, z in enumerate(reversed(ganz)))
    wert += sum(_ZIFFERN.index(z) * basis ** -(i + 1) for i, z in enumerate(frac))
    return vz * wert


def dezimal_zu_dual(x, nachkomma=0):
    return dezimal_zu_basis(x, 2, nachkomma)


def dual_zu_dezimal(s):
    return basis_zu_dezimal(s, 2)


def dezimal_zu_hex(x):
    return dezimal_zu_basis(x, 16)


def hex_zu_dezimal(s):
    return basis_zu_dezimal(s, 16)


def hex_zu_dual(s):
    """Jede Hexziffer -> 4 Bit (Tetrade)."""
    return " ".join(format(_ZIFFERN.index(z), "04b") for z in s.strip().upper())


def dual_zu_hex(s):
    """Von rechts in Vierergruppen zusammenfassen."""
    s = s.replace(" ", "")
    s = s.zfill((len(s) + 3) // 4 * 4)
    return "".join(_ZIFFERN[int(s[i:i + 4], 2)] for i in range(0, len(s), 4))


def zweierkomplement(x, bits=8):
    """Ganzzahl (auch negativ) -> Zweierkomplement-Darstellung mit 'bits' Stellen (invertieren + 1)."""
    if not -(1 << (bits - 1)) <= x < (1 << (bits - 1)):
        raise ValueError(f"{x} passt nicht in {bits} Bit (Zweierkomplement)")
    return format(x & ((1 << bits) - 1), f"0{bits}b")


def zweierkomplement_wert(s):
    """Zweierkomplement-Bitmuster -> vorzeichenbehaftete Ganzzahl (MSB = Vorzeichen)."""
    s = s.replace(" ", "")
    w = int(s, 2)
    return w - (1 << len(s)) if s[0] == "1" else w


def dual_addieren(a, b):
    return dezimal_zu_dual(int(a, 2) + int(b, 2))


def dual_subtrahieren(a, b, bits=None):
    """a - b durch Addition des Zweierkomplements von b; Ueberlauf-Bit wird verworfen.
    Ergebnis als Zweierkomplement mit 'bits' Stellen (Standard: genug fuer a und b plus Vorzeichen)."""
    bits = bits or max(len(a), len(b)) + 1
    summe = int(a, 2) + int(zweierkomplement(-int(b, 2), bits), 2)
    return format(summe & ((1 << bits) - 1), f"0{bits}b")


def dual_multiplizieren(a, b):
    """Schieben und Addieren: P = sum (a * b_i) << i."""
    a_w, p = int(a, 2), 0
    for i, bit in enumerate(reversed(b)):
        if bit == "1":
            p += a_w << i
    return dezimal_zu_dual(p)


def dual_dividieren(a, b):
    """Schriftliche Division (Schieben und Subtrahieren) -> (Quotient, Rest) als Dualzahlen."""
    d, q, r = int(b, 2), 0, 0
    if d == 0:
        raise ZeroDivisionError("Division durch 0")
    for bit in a:
        r = (r << 1) | int(bit)
        q <<= 1
        if r >= d:
            r -= d
            q |= 1
    return dezimal_zu_dual(q), dezimal_zu_dual(r)


# ===========================================================================
# 25. MATHEMATIK (Prozent, Dreisatz, Runden, Trigonometrie, Analysis, Vektoren)
# ===========================================================================
def prozentwert(G, p):
    """W = G p / 100."""
    return G * p / 100.0


def grundwert(W, p):
    """G = W 100 / p."""
    return W * 100.0 / p


def prozentsatz(W, G):
    """p = W / G 100."""
    return W / G * 100.0


def prozentuale_aenderung(alt, neu):
    """(neu - alt) / alt 100."""
    return (neu - alt) / alt * 100.0


def dreisatz(a1, b1, a2):
    """Proportional (je mehr, desto mehr): b2 = b1 a2 / a1."""
    return b1 * a2 / a1


def dreisatz_antiproportional(a1, b1, a2):
    """Antiproportional (je mehr, desto weniger): a1 b1 = a2 b2 -> b2 = a1 b1 / a2."""
    return a1 * b1 / a2


def runden(x, stellen=0):
    """Kaufmaennisch runden (5 -> auf), anders als Python round() (Bankers Rounding)."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-stellen)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def runden_auf_raster(x, raster):
    """Auf ein Vielfaches von 'raster' runden (z. B. 0,05 oder Normwerte im festen Raster)."""
    return runden(x / raster) * raster


def bruch(zaehler, nenner=1):
    """Exakter Bruch (gekuerzt) - +, -, *, / rechnen exakt: bruch(1, 3) + bruch(1, 6) == bruch(1, 2)."""
    from fractions import Fraction
    return Fraction(zaehler, nenner)


def gemischte_zahl(b):
    """Unechter Bruch -> (ganze Zahl, Restbruch), z. B. 7/3 -> (2, 1/3)."""
    from fractions import Fraction
    b = Fraction(b)
    ganz = int(b)
    return ganz, b - ganz


def nte_wurzel(a, n):
    """a^(1/n) (fuer ungerade n auch negative a)."""
    if a < 0 and n % 2 == 1:
        return -((-a) ** (1.0 / n))
    return a ** (1.0 / n)


def logarithmus(x, basis=10.0):
    """log_b(x) = ln(x) / ln(b) (Basiswechsel)."""
    return np.log(x) / np.log(basis)


def exponentialfunktion(a, b, x):
    """f(x) = a e^(b x) (b < 0: Abklingen, z. B. U(t) = U0 e^(-t/tau))."""
    return a * np.exp(b * x)


def halbwertszeit(b):
    """T_1/2 = ln(2) / |b| fuer f = a e^(-b t) (Verdopplungszeit bei Wachstum)."""
    return np.log(2.0) / abs(b)


def zinseszins(K0, r, t):
    """K(t) = K0 (1 + r)^t (allgemeines exponentielles Wachstum mit Rate r pro Periode)."""
    return K0 * (1.0 + r) ** t


def pythagoras_hypotenuse(a, b):
    """c = sqrt(a^2 + b^2) (auch Scheinwiderstand/-leistung, Zeigerbetrag)."""
    return np.hypot(a, b)


def pythagoras_kathete(c, a):
    """b = sqrt(c^2 - a^2)."""
    return np.sqrt(c ** 2 - a ** 2)


def winkelfunktionen(gegenkathete, ankathete):
    """-> dict mit sin, cos, tan, cot und Winkel (rad) eines rechtwinkligen Dreiecks."""
    c = np.hypot(gegenkathete, ankathete)
    return dict(sin=gegenkathete / c, cos=ankathete / c, tan=gegenkathete / ankathete,
                cot=ankathete / gegenkathete, winkel=np.arctan2(gegenkathete, ankathete))


def sinussatz_seite(a, alpha, beta):
    """b = a sin(beta) / sin(alpha) (Winkel in rad)."""
    return a * np.sin(beta) / np.sin(alpha)


def kosinussatz(a, b, gamma):
    """c = sqrt(a^2 + b^2 - 2 a b cos(gamma)) (gamma in rad, gegenueber c)."""
    return np.sqrt(a ** 2 + b ** 2 - 2.0 * a * b * np.cos(gamma))


def kosinussatz_winkel(a, b, c):
    """gamma = acos((a^2 + b^2 - c^2) / (2 a b))."""
    return np.arccos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b))


def ableitung(f, x, h=1e-6):
    """Numerische Ableitung f'(x) (zentraler Differenzenquotient)."""
    s = h * max(abs(x), 1.0)
    return (f(x + s) - f(x - s)) / (2.0 * s)


def ableitung2(f, x, h=1e-4):
    """Numerische zweite Ableitung f''(x)."""
    s = h * max(abs(x), 1.0)
    return (f(x + s) - 2.0 * f(x) + f(x - s)) / s ** 2


def integral(f, a, b, n=1000):
    """Bestimmtes Integral (Simpson-Regel, n gerade) = F(b) - F(a)."""
    n += n % 2
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / (3.0 * n) * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())


def mittelwert_integral(f, a, b, n=1000):
    """Linearer Mittelwert 1/(b-a) int f dx (Effektivwert: sqrt(mittelwert_integral(f^2)))."""
    return integral(f, a, b, n) / (b - a)


def flaeche_zwischen(f, g, a, b, n=1000):
    """int_a^b |f(x) - g(x)| dx."""
    return integral(lambda x: np.abs(f(x) - g(x)), a, b, n)


def vektor_betrag(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def einheitsvektor(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def skalarprodukt(a, b):
    return float(np.dot(a, b))


def kreuzprodukt(a, b):
    return np.cross(a, b)


def winkel_vektoren(a, b):
    """Winkel zwischen zwei Vektoren in rad."""
    return float(np.arccos(np.clip(skalarprodukt(a, b) / (vektor_betrag(a) * vektor_betrag(b)), -1.0, 1.0)))


def umstellen(fn: Callable, gesucht: str, ergebnis: float, **bekannt) -> float:
    """Formel numerisch nach einer Eingangsgroesse aufloesen ("Formel umstellen").
    Beispiel: umstellen(ef.resonanzfrequenz, "C", 1000.0, L=10e-3) -> C fuer f0 = 1 kHz.
    Sucht ueber 30 Dekaden (positive Werte) einen Vorzeichenwechsel und bisektiert dann."""
    def g(x):
        return float(np.real(fn(**{gesucht: x}, **bekannt))) - ergebnis

    xs = np.logspace(-15, 15, 601)
    gs = np.array([g(x) for x in xs])
    idx = np.where(np.sign(gs[:-1]) * np.sign(gs[1:]) <= 0)[0]
    if len(idx) == 0:
        raise ValueError(f"Keine (positive) Loesung fuer {gesucht} gefunden")
    lo, hi = xs[idx[0]], xs[idx[0] + 1]
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if np.sign(g(mid)) == np.sign(g(lo)):
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


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
