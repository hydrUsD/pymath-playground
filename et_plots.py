#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
et_plots.py - matplotlib-Visualisierungen zur Formelsammlung et_formeln.py

Aufruf
------
    python et_plots.py --liste                    # verfuegbare Plots
    python et_plots.py rc_lade_entlade bode       # bestimmte Plots anzeigen
    python et_plots.py --alle --speichern plots   # alle als PNG in Ordner 'plots' speichern
    python et_plots.py --alle                     # alle nacheinander anzeigen

Jede Plot-Funktion gibt die ``matplotlib.figure.Figure`` zurueck und kann auch
direkt aus eigenem Code aufgerufen werden.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
import numpy as np

import et_formeln as ef

PI = math.pi
plt = None  # wird nach Backend-Wahl gesetzt (_pyplot)


def _pyplot(headless: bool):
    global plt
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt = _plt
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 110,
                         "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------
# 1. Schaltvorgaenge
# ---------------------------------------------------------------------------
def rc_lade_entlade():
    """RC-Glied: Auf- und Entladung mit Zeitkonstanten-Markierungen."""
    R, C, U0 = 1e3, 1e-6, 5.0
    tau = ef.zeitkonstante_rc(R, C)
    t = np.linspace(0, 6 * tau, 600)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(t * 1e3, ef.rc_ladespannung(t, U0, R, C), label="$u_C(t)$ Laden")
    a1.plot(t * 1e3, ef.rc_entladespannung(t, U0, R, C), label="$u_C(t)$ Entladen")
    for k, anteil in ((1, 0.632), (2, 0.865), (3, 0.950), (5, 0.993)):
        a1.plot(k * tau * 1e3, U0 * anteil, "ko", ms=4)
        a1.annotate(f"{k}τ: {anteil*100:.1f} %", (k * tau * 1e3, U0 * anteil), textcoords="offset points", xytext=(6, -12))
    a1.set(xlabel="t in ms", ylabel="u in V", title=f"RC: R = 1 kΩ, C = 1 µF, τ = {tau*1e3:.1f} ms")
    a1.legend()
    a2.plot(t * 1e3, ef.rc_ladestrom(t, U0, R, C) * 1e3, color="tab:red")
    a2.set(xlabel="t in ms", ylabel="i in mA", title="Ladestrom $i(t) = U_0/R\\,e^{-t/\\tau}$")
    fig.tight_layout()
    return fig


def rlc_sprungantwort():
    """Reihen-RLC: Sprungantwort fuer verschiedene Daempfungsgrade."""
    L, C = 1e-3, 1e-6
    Rkrit = ef.widerstand_aperiodisch(L, C)
    t = np.linspace(0, 1.2e-3, 3000)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for zeta in (0.1, 0.3, 0.7, 1.0, 2.0):
        R = zeta * Rkrit
        ax.plot(t * 1e3, ef.rlc_sprung_uc(t, R, L, C, 1.0), label=f"ζ = {zeta} (R = {R:.1f} Ω)")
    ax.axhline(1, color="k", lw=0.8)
    ax.set(xlabel="t in ms", ylabel="$u_C / U_0$", title="Reihen-RLC-Sprungantwort")
    ax.legend()
    ov = ef.ueberschwingweite(0.3)
    ax.annotate(f"Überschwingen (ζ=0,3): {ov*100:.1f} %", (0.1, 1 + ov), xytext=(0.35, 1.5),
                arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Filter / Frequenzgang
# ---------------------------------------------------------------------------
def bode():
    """Bode-Diagramm: RC-Tiefpass/Hochpass, Butterworth 1..5, Tiefpass 2. Ordnung mit variabler Guete."""
    f = np.logspace(1, 5, 800)
    fc = 1e3
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    ax[0, 0].semilogx(f, ef.betrag_db(ef.tiefpass_1(f, fc)), label="Tiefpass 1. Ord.")
    ax[0, 0].semilogx(f, ef.betrag_db(ef.hochpass_1(f, fc)), label="Hochpass 1. Ord.")
    ax[1, 0].semilogx(f, ef.phase_grad(ef.tiefpass_1(f, fc)))
    ax[1, 0].semilogx(f, ef.phase_grad(ef.hochpass_1(f, fc)))
    for n in range(1, 6):
        ax[0, 1].semilogx(f, ef.betrag_db(ef.butterworth_betrag(f, fc, n)), label=f"Butterworth n={n}")
    for Q in (0.5, 0.7071, 1, 3, 10):
        H = ef.tiefpass_2(f, fc, Q)
        ax[1, 1].semilogx(f, ef.phase_grad(H), label=f"Q={Q}")
    ax[0, 0].set(ylim=(-60, 5), ylabel="|H| in dB", title="RC-Filter 1. Ordnung (fc = 1 kHz)")
    ax[0, 1].set(ylim=(-100, 5), title="Butterworth-Tiefpass")
    ax[1, 0].set(xlabel="f in Hz", ylabel="φ in °")
    ax[1, 1].set(xlabel="f in Hz", title="Phase Tiefpass 2. Ordnung")
    for a in ax[0]:
        a.axhline(-3.01, color="gray", ls="--", lw=0.8)
        a.legend(fontsize=8)
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def resonanz_reihenkreis():
    """Reihenschwingkreis: |Z|, Strom und Phase fuer verschiedene Guete (verschiedene R)."""
    L, C = 1e-3, 1e-6
    f0 = ef.resonanzfrequenz(L, C)
    f = np.linspace(0.3 * f0, 2.0 * f0, 4000)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for R in (2.0, 5.0, 10.0, 30.0):
        Z = ef.z_reihe_rlc(f, R, L, C)
        Q = ef.guete_reihe(R, L, C)
        lab = f"R={R:g} Ω (Q={Q:.1f})"
        ax[0].semilogy(f, np.abs(Z), label=lab)
        ax[1].plot(f, 1.0 / np.abs(Z), label=lab)
        ax[2].plot(f, np.degrees(np.angle(Z)), label=lab)
    for a in ax:
        a.axvline(f0, color="k", ls=":", lw=0.8)
    ax[0].set(xlabel="f in Hz", ylabel="|Z| in Ω", title=f"Impedanz, f0 = {f0:.0f} Hz")
    ax[1].set(xlabel="f in Hz", ylabel="I in A bei U = 1 V", title="Strom (Resonanzkurve)")
    ax[2].set(xlabel="f in Hz", ylabel="φ in °", title="Phase")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def skintiefe():
    """Skintiefe ueber der Frequenz fuer Cu, Al und Eisen."""
    f = np.logspace(1, 9, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, rho, mur in (("Kupfer", 1.72e-8, 1.0), ("Aluminium", 2.82e-8, 1.0), ("Eisen (µr=200)", 1.0e-7, 200.0)):
        ax.loglog(f, ef.skintiefe(f, rho, mur) * 1e3, label=name)
    ax.set(xlabel="f in Hz", ylabel="δ in mm", title="Skintiefe δ = √(ρ / (π f µ0 µr))")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Wechsel- und Drehstrom
# ---------------------------------------------------------------------------
def wechselstromleistung():
    """u(t), i(t), p(t) mit Wirk- und Blindanteil."""
    f, U, I, phi = 50.0, 230.0, 10.0, PI / 4
    t = np.linspace(0, 0.04, 2000)
    u = math.sqrt(2) * U * np.sin(2 * PI * f * t)
    i = math.sqrt(2) * I * np.sin(2 * PI * f * t - phi)
    p = u * i
    P = ef.wirkleistung(U, I, phi)
    Q = ef.blindleistung(U, I, phi)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(t * 1e3, u, label="u(t) in V")
    a1.plot(t * 1e3, i * 20, label="i(t) · 20 in A")
    a1.legend(loc="upper right")
    a1.set(title=f"U = 230 V, I = 10 A, φ = 45°")
    a2.plot(t * 1e3, p, color="tab:green", label="p(t) = u·i")
    a2.axhline(P, color="k", ls="--", label=f"P = {P:.0f} W")
    a2.fill_between(t * 1e3, p, P, alpha=0.15, color="tab:green")
    a2.set(xlabel="t in ms", ylabel="p in W")
    a2.legend(title=f"Q = {Q:.0f} var, S = {ef.scheinleistung(U, I):.0f} VA", loc="upper right")
    fig.tight_layout()
    return fig


def leistungsdreieck_kompensation():
    """Leistungsdreieck vor/nach Kompensation und Kondensatorbedarf ueber cos(phi2)."""
    P, c1 = 10e3, 0.7
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for c2, ls, col in ((c1, "-", "tab:red"), (0.95, "--", "tab:green")):
        Q = P * math.tan(math.acos(c2))
        a1.plot([0, P, P, 0], [0, 0, Q, 0], ls, color=col, label=f"cos φ = {c2}: Q = {Q/1e3:.2f} kvar, S = {math.hypot(P, Q)/1e3:.2f} kVA")
    a1.set(xlabel="P in W", ylabel="Q in var", title="Leistungsdreieck", aspect="equal")
    a1.legend(fontsize=8)
    c2 = np.linspace(0.75, 0.999, 200)
    a2.plot(c2, ef.kompensation_kapazitaet(P, c1, c2, 50, 230) * 1e6)
    a2.set(xlabel="Ziel-cos φ", ylabel="C in µF (einphasig, 230 V, 50 Hz)",
           title="Kondensatorbedarf für P = 10 kW von cos φ = 0,7")
    fig.tight_layout()
    return fig


def drehstrom():
    """Drei Strangspannungen, verkettete Spannung und Summe."""
    f, Uph = 50.0, 230.0
    t = np.linspace(0, 0.04, 2000)
    ph = [math.sqrt(2) * Uph * np.sin(2 * PI * f * t - k * 2 * PI / 3) for k in range(3)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for k, (y, c) in enumerate(zip(ph, ("tab:orange", "tab:green", "tab:purple")), 1):
        a1.plot(t * 1e3, y, color=c, label=f"L{k}")
    a1.plot(t * 1e3, ph[0] + ph[1] + ph[2], "k--", label="Summe = 0")
    a2.plot(t * 1e3, ph[0] - ph[1], label=f"$u_{{12}}$, Û = {math.sqrt(2)*ef.spannung_verkettet(Uph):.0f} V")
    a2.plot(t * 1e3, ph[0], color="tab:orange", alpha=0.5, label="$u_{1}$")
    a1.set(ylabel="u in V", title="Strangspannungen 3 × 230 V")
    a2.set(xlabel="t in ms", ylabel="u in V", title="Verkettete Spannung (U_L = √3 · U_ph = 400 V)")
    a1.legend(ncol=4, loc="upper right")
    a2.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Gleichstrom / Quelle
# ---------------------------------------------------------------------------
def leistungsanpassung():
    """Leistung und Wirkungsgrad ueber Ra/Ri."""
    Uq, Ri = 10.0, 5.0
    Ra = np.linspace(0.05, 6 * Ri, 500)
    I = Uq / (Ri + Ra)
    P = I ** 2 * Ra
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(Ra / Ri, P / ef.leistung_max_anpassung(Uq, Ri), label="P / Pmax")
    ax.plot(Ra / Ri, ef.wirkungsgrad_quelle(Ri, Ra), label="Wirkungsgrad η")
    ax.axvline(1.0, color="k", ls=":")
    ax.set(xlabel="Ra / Ri", title="Leistungsanpassung: Pmax bei Ra = Ri, dort η = 50 %")
    ax.legend()
    fig.tight_layout()
    return fig


def diodenkennlinie():
    """Shockley-Kennlinie linear und logarithmisch, Temperaturabhaengigkeit."""
    U = np.linspace(0, 0.8, 500)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for T in (250, 300, 350, 400):
        I = ef.diode_strom(U, 1e-12, 1.0, T)
        a1.plot(U, np.minimum(I * 1e3, 50), label=f"T = {T} K (UT = {ef.thermospannung(T)*1e3:.1f} mV)")
        a2.semilogy(U, I, label=f"{T} K")
    a1.set(xlabel="U in V", ylabel="I in mA", title="Diodenkennlinie (Is = 1 pA, n = 1)", ylim=(0, 50))
    a2.set(xlabel="U in V", ylabel="I in A", title="logarithmisch: 59,5 mV pro Dekade bei 300 K")
    a1.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Maschinen, Leitungen, Einheiten
# ---------------------------------------------------------------------------
def kloss_kennlinie():
    """Asynchronmaschine: M(n) nach Kloss fuer verschiedene Kippschluepfe."""
    f, p = 50.0, 2
    ns = ef.synchrondrehzahl(f, p) * 60
    s = np.linspace(1e-4, 1.0, 1000)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for sk in (0.1, 0.2, 0.4):
        ax.plot(ns * (1 - s), ef.kloss_moment(100.0, s, sk), label=f"$s_k$ = {sk}")
    ax.axvline(ns, color="k", ls=":")
    ax.set(xlabel="n in 1/min", ylabel="M in % von $M_k$", title=f"Kloss'sche Formel, ns = {ns:.0f} 1/min")
    ax.legend()
    fig.tight_layout()
    return fig


def leitung_reflexion():
    """Reflexionsfaktor, VSWR und Rueckflussdaempfung ueber ZL/Z0."""
    zl = np.logspace(-1.3, 1.3, 400) * 50.0
    r = ef.reflexionsfaktor(zl, 50.0)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].semilogx(zl / 50, r)
    ax[0].set(xlabel="ZL / Z0", ylabel="r", title="Reflexionsfaktor")
    ax[1].loglog(zl / 50, ef.stehwellenverhaeltnis(r))
    ax[1].set(xlabel="ZL / Z0", ylabel="VSWR", title="Stehwellenverhältnis")
    with np.errstate(divide="ignore"):
        ax[2].semilogx(zl / 50, np.minimum(ef.rueckflussdaempfung_db(r), 60))
    ax[2].set(xlabel="ZL / Z0", ylabel="a_R in dB", title="Rückflussdämpfung")
    fig.tight_layout()
    return fig


def db_umrechnung():
    """dB gegen lineares Verhaeltnis (Leistung vs. Spannung) und dBm/Watt."""
    v = np.logspace(-3, 3, 400)
    P = np.logspace(-9, 3, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.semilogx(v, ef.db_leistung(v), label="Leistung 10·lg(v)")
    a1.semilogx(v, ef.db_spannung(v), label="Spannung/Strom 20·lg(v)")
    for val, txt in ((2, "×2: 3 dB / 6 dB"), (10, "×10: 10 dB / 20 dB")):
        a1.annotate(txt, (val, ef.db_spannung(val)), textcoords="offset points", xytext=(6, 4), fontsize=8)
    a1.set(xlabel="Verhältnis v", ylabel="dB", title="Dezibel")
    a1.legend()
    a2.semilogx(P, ef.watt_zu_dbm(P))
    a2.set(xlabel="P in W", ylabel="dBm", title="Leistung in dBm (0 dBm = 1 mW)")
    fig.tight_layout()
    return fig


def trafo_wirkungsgrad():
    """Transformator: Wirkungsgrad ueber Belastung (Pcu ~ Last^2, Pfe konstant)."""
    Sn, Pfe, Pcun, cos_phi = 10e3, 100.0, 200.0, 0.9
    x = np.linspace(0.02, 1.3, 400)
    P2 = x * Sn * cos_phi
    eta = ef.trafo_wirkungsgrad(P2, Pfe, Pcun * x ** 2)
    x_opt = math.sqrt(Pfe / Pcun)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x * 100, eta * 100)
    ax.axvline(x_opt * 100, color="k", ls=":", label=f"Optimum bei {x_opt*100:.0f} % Last (Pfe = Pcu)")
    ax.set(xlabel="Last in % von Sn", ylabel="η in %", title="Trafo-Wirkungsgrad (10 kVA, cos φ = 0,9)", ylim=(90, 100))
    ax.legend()
    fig.tight_layout()
    return fig


PLOTS = {
    "rc_lade_entlade": rc_lade_entlade,
    "rlc_sprungantwort": rlc_sprungantwort,
    "bode": bode,
    "resonanz_reihenkreis": resonanz_reihenkreis,
    "skintiefe": skintiefe,
    "wechselstromleistung": wechselstromleistung,
    "leistungsdreieck_kompensation": leistungsdreieck_kompensation,
    "drehstrom": drehstrom,
    "leistungsanpassung": leistungsanpassung,
    "diodenkennlinie": diodenkennlinie,
    "kloss_kennlinie": kloss_kennlinie,
    "leitung_reflexion": leitung_reflexion,
    "db_umrechnung": db_umrechnung,
    "trafo_wirkungsgrad": trafo_wirkungsgrad,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Elektrotechnik-Plots (matplotlib)")
    ap.add_argument("plots", nargs="*", help="Namen der Plots (siehe --liste)")
    ap.add_argument("--liste", action="store_true", help="verfuegbare Plots anzeigen")
    ap.add_argument("--alle", action="store_true", help="alle Plots erzeugen")
    ap.add_argument("--speichern", metavar="ORDNER", help="PNG-Dateien in ORDNER speichern (kein Fenster)")
    a = ap.parse_args(argv)
    if a.liste or not (a.plots or a.alle):
        for n, fn in PLOTS.items():
            print(f"  {n:32s} {fn.__doc__.strip().splitlines()[0]}")
        return 0
    _pyplot(headless=bool(a.speichern))
    namen = list(PLOTS) if a.alle else a.plots
    unbekannt = [n for n in namen if n not in PLOTS]
    if unbekannt:
        print("Unbekannte Plots:", ", ".join(unbekannt), file=sys.stderr)
        return 2
    if a.speichern:
        os.makedirs(a.speichern, exist_ok=True)
    for n in namen:
        fig = PLOTS[n]()
        if a.speichern:
            fig.savefig(os.path.join(a.speichern, f"{n}.png"), dpi=130)
            plt.close(fig)
            print("gespeichert:", os.path.join(a.speichern, f"{n}.png"))
    if not a.speichern:
        plt.show()
    return 0


# Beim Import aus eigenem Code: pyplot mit Standard-Backend initialisieren
if plt is None and __name__ != "__main__":
    _pyplot(headless=False)

if __name__ == "__main__":
    sys.exit(main())
