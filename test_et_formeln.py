#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifikation der Formelsammlung (nur Standardbibliothek + NumPy, Start: python test_et_formeln.py)

Drei unabhaengige Pruefebenen:
  A) Dimensionspruefung   - jede registrierte Formel muss dimensionshomogen sein
  B) Referenzwerte        - von Hand/Lehrbuch bekannte Ergebnisse (Literalwerte, nicht aus dem Modul kopiert)
  C) Gegenrechnung        - physikalische Identitaeten und numerische Integration (RK4, Summation, Sweep)
"""
import math
import unittest

import numpy as np

import et_formeln as ef

PI = math.pi


class Basis(unittest.TestCase):
    def nah(self, ist, soll, rel=1e-9, abs_=0.0):
        np.testing.assert_allclose(ist, soll, rtol=rel, atol=abs_)


# ---------------------------------------------------------------------------
# A) Dimensionspruefung
# ---------------------------------------------------------------------------
class TestDimensionen(Basis):
    def test_parser(self):
        self.assertEqual(list(ef.dim_vektor("V/A")), list(ef.dim_vektor("ohm")))
        self.assertEqual(list(ef.dim_vektor("J/kg/K")), [0, 2, -2, 0, -1])
        self.assertEqual(list(ef.dim_vektor("m^-3")), [0, -3, 0, 0, 0])
        self.assertEqual(list(ef.dim_vektor("V*s")), list(ef.dim_vektor("Wb")))
        self.assertEqual(list(ef.dim_vektor("H/m")), list(ef.dim_vektor("ohm*s/m")))
        self.assertEqual(list(ef.dim_vektor("F/m")), list(ef.dim_vektor("C/V/m")))
        self.assertEqual(list(ef.dim_vektor("T")), list(ef.dim_vektor("V*s/m^2")))
        self.assertEqual(list(ef.dim_vektor("N")), list(ef.dim_vektor("A*T*m")))

    def test_konstanten_dimensionen(self):
        # c^2 * mu0 * eps0 = 1  (Zahlenwert und Dimension)
        self.nah(ef.C0 ** 2 * ef.MU0 * ef.EPS0, 1.0, rel=1e-9)
        d = 2 * ef.dim_vektor("m/s") + ef.dim_vektor("H/m") + ef.dim_vektor("F/m")
        self.assertTrue(np.all(d == 0))
        # h*f = E: J*s * Hz = J ; kB*T = J
        self.assertTrue(np.all(ef.dim_vektor("J*s") + ef.dim_vektor("Hz") == ef.dim_vektor("J")))
        self.assertTrue(np.all(ef.dim_vektor("J/K") + ef.dim_vektor("K") == ef.dim_vektor("J")))

    def test_alle_formeln_dimensionshomogen(self):
        self.assertTrue(ef.pruefe_alle_dimensionen())

    def test_pruefer_erkennt_fehler(self):
        """Negativtest: absichtlich falsche Formeln MUESSEN durchfallen."""
        lam = [1.7, 2.3, 3.1, 1.3, 2.9]

        @ef.formel("falsch", "U = R + I", {"R": "ohm", "I": "A"}, "V", "Test", dict(R=1.0, I=2.0))
        def falsch_summe(R, I):
            return R + I

        @ef.formel("falsch2", "P = U*I^2", {"U": "V", "I": "A"}, "W", "Test", dict(U=1.0, I=2.0))
        def falsch_potenz(U, I):
            return U * I ** 2

        @ef.formel("falsch3", "f0 = 1/(2 pi L C)", {"L": "H", "C": "F"}, "Hz", "Test", dict(L=1e-3, C=1e-6))
        def falsch_thomson(L, C):
            return 1.0 / (2 * PI * L * C)   # Wurzel vergessen

        try:
            for fn in (falsch_summe, falsch_potenz, falsch_thomson):
                ok, _ = ef.pruefe_dimension(fn, lam)
                self.assertFalse(ok, fn.__name__)
        finally:
            for n in ("falsch_summe", "falsch_potenz", "falsch_thomson"):
                ef.REGISTRY.pop(n, None)

    def test_registry_konsistent(self):
        import inspect
        for name, fn in ef.REGISTRY.items():
            m = fn.meta
            params = set(inspect.signature(fn).parameters)
            self.assertEqual(set(m["ein"]), params, name)
            self.assertEqual(set(m["bsp"]), {p for p in params if p in m["bsp"]}, name)
            fn(**{k: np.asarray(v) for k, v in m["bsp"].items()})   # Beispiel rechenbar


# ---------------------------------------------------------------------------
# B) Referenzwerte
# ---------------------------------------------------------------------------
class TestGleichstrom(Basis):
    def test_ohm_und_leistung(self):
        self.nah(ef.spannung_ohm(100, 0.5), 50.0)
        self.nah(ef.strom_ohm(12, 6), 2.0)
        self.nah(ef.widerstand_ohm(12, 2), 6.0)
        self.nah(ef.leistung_ui(230, 10), 2300.0)
        self.nah(ef.leistung_i2r(2, 10), 40.0)
        self.nah(ef.leistung_u2r(12, 6), 24.0)
        self.nah(ef.energie(1000, 3600), 3.6e6)
        self.nah(ef.ladung(2, 1800), 3600.0)

    def test_widerstandsnetze(self):
        self.nah(ef.widerstand_reihe(10, 20, 30), 60.0)
        self.nah(ef.widerstand_parallel(100, 100), 50.0)
        self.nah(ef.widerstand_parallel(10, 20, 30), 60.0 / 11.0)
        self.nah(ef.widerstand_parallel2(100, 300), 75.0)
        self.nah(ef.spannungsteiler(12, 1000, 2000), 8.0)
        self.nah(ef.stromteiler(3, 1, 2), 2.0)
        self.nah(ef.wheatstone_rx(100, 200, 50), 25.0)

    def test_stern_dreieck(self):
        self.nah(ef.stern_zu_dreieck(10, 10, 10), (30, 30, 30))
        self.nah(ef.dreieck_zu_stern(30, 30, 30), (10, 10, 10))
        # Rundlauf mit ungleichen Widerstaenden
        r = (10.0, 20.0, 30.0)
        self.nah(ef.dreieck_zu_stern(*ef.stern_zu_dreieck(*r)), r)
        # Gleiche Klemmenwiderstaende: R_ab (Stern) = R1+R2 muss = R12 || (R23+R31) (Dreieck)
        R12, R23, R31 = ef.stern_zu_dreieck(*r)
        self.nah(r[0] + r[1], ef.widerstand_parallel(R12, R23 + R31))

    def test_leiter(self):
        self.nah(ef.widerstand_leiter(0.0172e-6, 100, 1.5e-6), 1.146666666, rel=1e-7)
        self.nah(ef.widerstand_temperatur(1.0, 0.00393, 100), 1.393)
        self.nah(ef.spannungsfall_gleichstrom(16, 0.0172e-6, 30, 2.5e-6), 6.6048, rel=1e-9)   # 2 * 3,3024 V (Hin+Rueck)
        self.nah(ef.erwaermung_dT(2000, 60, 1.0, 4182), 28.694, rel=1e-4)

    def test_quelle(self):
        self.nah(ef.klemmenspannung(12.6, 0.05, 20), 11.6)
        self.nah(ef.leistung_max_anpassung(10, 5), 5.0)
        self.nah(ef.wirkungsgrad_quelle(5, 5), 0.5)
        # Anpassung: P(Ra) ist bei Ra = Ri maximal und = Pmax
        Ra = np.linspace(0.1, 50, 5000)
        P = (10.0 / (5.0 + Ra)) ** 2 * Ra
        self.assertAlmostEqual(Ra[np.argmax(P)], 5.0, delta=0.02)
        self.nah(P.max(), ef.leistung_max_anpassung(10, 5), rel=1e-5)

    def test_knotenpotential(self):
        phi = ef.knotenpotentiale([[3, -1], [-1, 2]], [1, 2])
        self.nah(phi, [0.8, 1.4])     # 3*0.8-1.4=1.0 ; -0.8+2.8=2.0
        self.nah(np.array([[3, -1], [-1, 2]]) @ phi, [1, 2])


class TestBauelementeFelder(Basis):
    def test_kapazitaet_induktivitaet(self):
        self.nah(ef.kapazitaet_platten(1, 1e-4, 1e-3), 8.8541878128e-13)
        self.nah(ef.kapazitaet_koax(1, 1, 1e-3, math.e * 1e-3), 2 * PI * 8.8541878128e-12)
        self.nah(ef.kapazitaet_kugel(1, 0.1), 1.1126500560e-11, rel=1e-8)   # 4 pi eps0 * 0.1
        self.nah(ef.induktivitaet_spule(1, 100, 1e-4, 0.1), 1.25663706212e-5)
        self.nah(ef.induktivitaet_al(250e-9, 40), 400e-6)
        self.nah(ef.kapazitaet_reihe(10e-6, 40e-6), 8e-6)
        self.nah(ef.kapazitaet_reihe2(10e-6, 40e-6), 8e-6)
        self.nah(ef.kapazitaet_parallel(1e-6, 2e-6, 3e-6), 6e-6)
        self.nah(ef.induktivitaet_reihe(1e-3, 2e-3), 3e-3)
        self.nah(ef.induktivitaet_parallel(3e-3, 6e-3), 2e-3)
        self.nah(ef.induktivitaet_reihe_gekoppelt(1e-3, 4e-3, 0.5), 7e-3)   # 1+4+2*0.5*2
        self.nah(ef.gegeninduktivitaet(0.8, 1e-3, 4e-3), 1.6e-3)

    def test_ringkern_gleich_spule_bei_gleicher_laenge(self):
        rm = 0.02
        self.nah(ef.induktivitaet_ringkern(1000, 100, 1e-4, rm), ef.induktivitaet_spule(1000, 100, 1e-4, 2 * PI * rm))

    def test_energie_und_dynamik(self):
        self.nah(ef.energie_kondensator(100e-6, 10), 5e-3)
        self.nah(ef.energie_spule(10e-3, 2), 20e-3)
        self.nah(ef.ladung_kondensator(100e-6, 10), 1e-3)
        self.nah(ef.strom_kondensator(100e-6, 5, 1e-3), 0.5)
        self.nah(ef.spannung_spule(10e-3, 2, 1e-3), 20.0)
        self.nah(ef.zeitkonstante_rc(1000, 1e-6), 1e-3)
        self.nah(ef.zeitkonstante_rl(10e-3, 10), 1e-3)

    def test_elektrisches_feld(self):
        self.nah(ef.coulomb_kraft(1, 1, 1), 8.987551792e9, rel=1e-8)
        self.nah(ef.coulomb_kraft(1e-6, 2e-6, 0.1), 1.797510358, rel=1e-8)
        self.nah(ef.feldstaerke_punktladung(1e-9, 0.05), 3595.02, rel=1e-5)
        self.nah(ef.potential_punktladung(1e-9, 0.05), 179.751, rel=1e-5)
        self.nah(ef.feldstaerke_platten(100, 1e-3), 1e5)
        self.nah(ef.verschiebungsdichte(1, 1e5), 8.8541878128e-7)
        # E = -grad(phi): numerische Ableitung des Potentials
        r, h = 0.05, 1e-7
        E_num = -(ef.potential_punktladung(1e-9, r + h) - ef.potential_punktladung(1e-9, r - h)) / (2 * h)
        self.nah(E_num, ef.feldstaerke_punktladung(1e-9, r), rel=1e-6)
        # F = q*E
        self.nah(ef.coulomb_kraft(1e-9, 2e-9, 0.05), 2e-9 * ef.feldstaerke_punktladung(1e-9, 0.05))
        # Plattenkondensator: Q = C U, D = Q/A
        C = ef.kapazitaet_platten(4, 1e-2, 1e-3)
        Q = ef.ladung_kondensator(C, 100)
        self.nah(ef.verschiebungsdichte(4, ef.feldstaerke_platten(100, 1e-3)), Q / 1e-2)
        # Energie: W = 1/2 C U^2 = w * Volumen
        self.nah(ef.energie_kondensator(C, 100), ef.energiedichte_e(4, 1e5) * 1e-2 * 1e-3)

    def test_magnetisches_feld(self):
        self.nah(ef.h_feld_leiter(10, 0.05), 31.8309886, rel=1e-8)
        self.nah(ef.flussdichte(1, ef.h_feld_leiter(10, 0.05)), 4.0e-5, rel=1e-8)   # 2e-7 * I / r
        self.nah(ef.h_feld_spule(500, 2, 0.25), 4000.0)
        self.nah(ef.fluss(1.5, 2e-4), 3e-4)
        self.nah(ef.energiedichte_b(1.0, 1.0), 397887.36, rel=1e-6)
        self.nah(ef.lorentzkraft(1.602176634e-19, 1e6, 1.0), 1.602176634e-13)
        self.nah(ef.kraft_leiter(0.5, 10, 0.2), 1.0)
        self.nah(ef.kraft_leiter(0.5, 10, 0.2, PI / 6), 0.5, rel=1e-12)
        self.nah(ef.induktionsspannung_bewegt(0.8, 0.5, 10), 4.0)
        self.nah(ef.induktionsspannung(200, 1e-3, 0.01), 20.0)

    def test_magnetischer_kreis(self):
        Rm = ef.magn_widerstand(0.2, 1000, 4e-4)
        self.nah(Rm, 0.2 / (1.25663706212e-6 * 1000 * 4e-4))
        Theta = ef.durchflutung(500, 2)
        phi = ef.magn_fluss_kreis(Theta, Rm)
        # L = N^2 / Rm  == induktivitaet_spule
        self.nah(500 ** 2 / Rm, ef.induktivitaet_spule(1000, 500, 4e-4, 0.2))
        # Energie im Feld = 1/2 Theta Phi = 1/2 L I^2
        self.nah(0.5 * Theta * phi, ef.energie_spule(500 ** 2 / Rm, 2.0))

    def test_hall_wellen_skin(self):
        self.nah(ef.hall_spannung(0.01, 0.5, 1e22, 1e-4), 0.005 / (1e22 * 1.602176634e-19 * 1e-4), rel=1e-12)
        self.nah(ef.hall_spannung(0.01, 0.5, 1e22, 1e-4), 3.1208e-2, rel=1e-4)
        self.nah(ef.skintiefe(1e6, 1.72e-8), 6.6e-5, rel=1e-2)                # Cu 1 MHz: 66 um
        self.nah(ef.skintiefe(50, 1.68e-8), 9.226e-3, rel=1e-3)                # Cu 50 Hz: ~9,2 mm
        self.nah(ef.wellenwiderstand_medium(), 376.730313668, rel=1e-8)
        self.nah(ef.wellenwiderstand_medium(4.0), 376.730313668 / 2, rel=1e-8)
        self.nah(ef.phasengeschwindigkeit(4.0), ef.C0 / 2)
        self.nah(ef.wellenlaenge(1e9), 0.299792458)
        self.nah(ef.poynting(377, 1), 377.0)
        # Poynting = Energiedichte * c  (ebene Welle im Vakuum): w_ges = eps0 E^2, S = w c
        E = 100.0
        H = E / ef.wellenwiderstand_medium()
        self.nah(ef.poynting(E, H), 2 * ef.energiedichte_e(1, E) * ef.C0, rel=1e-9)


class TestWechselstrom(Basis):
    def test_grundgroessen(self):
        self.nah(ef.kreisfrequenz(50), 314.1592653589793)
        self.nah(ef.frequenz_aus_T(0.02), 50.0)
        self.nah(ef.effektivwert_sinus(325.2691193), 230.0, rel=1e-8)
        self.nah(ef.scheitelwert_sinus(230), 325.2691193, rel=1e-8)
        self.nah(ef.gleichrichtwert_sinus(10), 6.366197724, rel=1e-8)
        self.nah(ef.effektivwert_dreieck(10), 5.773502692, rel=1e-8)
        self.nah(ef.phase_aus_zeit(50, 0.005), PI / 2)
        # Formfaktor Sinus = Ueff / Gleichrichtwert = pi / (2 sqrt 2) = 1.1107
        self.nah(ef.effektivwert_sinus(1) / ef.gleichrichtwert_sinus(1), 1.110720735, rel=1e-8)

    def test_effektivwert_numerisch(self):
        t = np.linspace(0, 0.02, 200001)
        self.nah(ef.effektivwert_numerisch(325.27 * np.sin(2 * PI * 50 * t), t), 325.27 / math.sqrt(2), rel=1e-6)
        # Dreieck 0 -> 10 -> 0 -> -10 -> 0
        td = np.linspace(0, 1, 400001)
        dreieck = 10 * (2 / PI) * np.arcsin(np.sin(2 * PI * td))
        self.nah(ef.effektivwert_numerisch(dreieck, td), 10 / math.sqrt(3), rel=1e-6)
        # Sinus mit Gleichanteil: Ueff^2 = U0^2 + Uhat^2/2
        ys = 3 + 4 * np.sin(2 * PI * td)
        self.nah(ef.effektivwert_numerisch(ys, td), math.sqrt(9 + 8), rel=1e-6)

    def test_blindwiderstaende_impedanzen(self):
        self.nah(ef.blindwiderstand_l(50, 1.0), 314.1592654)
        self.nah(ef.blindwiderstand_c(50, 10e-6), 318.3098862, rel=1e-8)
        self.nah(ef.z_spule(50, 0.1), 31.41592654j)
        self.nah(ef.z_kondensator(50, 10e-6), -318.3098862j, rel=1e-8)
        self.nah(ef.betrag_impedanz(3, 4), 5.0)
        self.nah(ef.phasenwinkel_impedanz(3, 4), math.atan2(4, 3))
        self.nah(abs(ef.z_reihe_rlc(1000, 10, 1e-3, 1e-6)), abs(10 + 1j * (2 * PI * 1000 * 1e-3 - 1 / (2 * PI * 1000 * 1e-6))))

    def test_leistungen(self):
        self.nah(ef.scheinleistung(230, 10), 2300.0)
        self.nah(ef.wirkleistung(230, 10, PI / 3), 1150.0)
        self.nah(ef.blindleistung(230, 10, PI / 3), 1991.858428, rel=1e-8)
        self.nah(ef.leistungsfaktor(1150, 2300), 0.5)
        self.nah(ef.phasenwinkel_pq(1000, 1000), PI / 4)
        self.nah(ef.strom_einphasig(2300, 230, 0.8), 12.5)
        # Leistungsdreieck: S^2 = P^2 + Q^2
        P, Q = ef.wirkleistung(230, 10, 0.7), ef.blindleistung(230, 10, 0.7)
        self.nah(math.hypot(P, Q), 2300.0)

    def test_momentanleistung(self):
        """Numerisch: Mittelwert von u(t)*i(t) = U I cos(phi)."""
        f, U, I, phi = 50.0, 230.0, 10.0, 0.6
        t = np.linspace(0, 0.02, 400001)
        u = math.sqrt(2) * U * np.sin(2 * PI * f * t)
        i = math.sqrt(2) * I * np.sin(2 * PI * f * t - phi)
        p = u * i
        P_num = np.sum((p[1:] + p[:-1]) * np.diff(t)) / 2 / 0.02
        self.nah(P_num, ef.wirkleistung(U, I, phi), rel=1e-6)

    def test_kompensation(self):
        # P = 10 kW, cos 0,70 -> 0,95 bei 230 V, 50 Hz  (Handrechnung: dQ = 10k*(1,0203-0,3287) = 6,916 kvar)
        dQ = ef.kompensation_blindleistung(10000, 0.7, 0.95)
        self.nah(dQ, 10000 * (math.tan(math.acos(0.7)) - math.tan(math.acos(0.95))))
        self.assertAlmostEqual(dQ, 6915.8, delta=1.0)
        C = ef.kompensation_kapazitaet(10000, 0.7, 0.95, 50, 230)
        self.assertAlmostEqual(C * 1e6, 416.2, delta=0.5)
        # Rueckrechnung: Qc = omega C U^2
        self.nah(2 * PI * 50 * C * 230 ** 2, dQ)
        # nach Kompensation stimmt cos phi
        Q_neu = 10000 * math.tan(math.acos(0.7)) - 2 * PI * 50 * C * 230 ** 2
        self.nah(math.cos(math.atan2(Q_neu, 10000)), 0.95)


class TestSchwingkreise(Basis):
    L, C, R = 1e-3, 1e-6, 10.0

    def test_resonanz(self):
        self.nah(ef.resonanzfrequenz(1e-3, 1e-6), 5032.921211, rel=1e-8)
        self.nah(ef.guete_reihe(10, 1e-3, 1e-6), math.sqrt(1000) / 10)
        self.nah(ef.guete_parallel(1000, 1e-3, 1e-6), 1000 * math.sqrt(1e-3))
        self.nah(ef.bandbreite(5000, 10), 500.0)
        self.nah(ef.daempfungsgrad_rlc(10, 1e-3, 1e-6), 0.5 / ef.guete_reihe(10, 1e-3, 1e-6))
        self.nah(ef.widerstand_aperiodisch(1e-3, 1e-6), 2 * math.sqrt(1000))

    def test_grenzfrequenzen_identitaeten(self):
        f0 = ef.resonanzfrequenz(self.L, self.C)
        Q = ef.guete_reihe(self.R, self.L, self.C)
        f1, f2 = ef.grenzfrequenzen_kreis(f0, Q)
        self.nah(f2 - f1, f0 / Q)                 # Bandbreite exakt f0/Q
        self.nah(f1 * f2, f0 ** 2)                # geometrisches Mittel = f0
        # An den -3 dB Grenzen ist |Z| = sqrt(2) R  (Reihenkreis)
        self.nah(abs(ef.z_reihe_rlc(f1, self.R, self.L, self.C)), math.sqrt(2) * self.R, rel=1e-9)
        self.nah(abs(ef.z_reihe_rlc(f2, self.R, self.L, self.C)), math.sqrt(2) * self.R, rel=1e-9)
        # Bei f0 ist Z = R (reell)
        self.nah(ef.z_reihe_rlc(f0, self.R, self.L, self.C), self.R + 0j, abs_=1e-9)

    def test_numerischer_sweep_reihe_und_parallel(self):
        f = np.linspace(1000, 12000, 2_200_001)
        f0 = ef.resonanzfrequenz(self.L, self.C)
        i_max = f[np.argmin(np.abs(ef.z_reihe_rlc(f, self.R, self.L, self.C)))]
        self.nah(i_max, f0, rel=1e-5)
        # Parallelkreis: Impedanzmaximum bei f0, Wert = R
        zp = np.abs(ef.z_parallel_rlc(f, 1000.0, self.L, self.C))
        self.nah(f[np.argmax(zp)], f0, rel=1e-5)
        self.nah(zp.max(), 1000.0, rel=1e-6)

    def test_gedaempfte_eigenfrequenz(self):
        w0 = 1 / math.sqrt(self.L * self.C)
        d = self.R / (2 * self.L)
        self.nah(ef.eigenkreisfrequenz_gedaempft(self.R, self.L, self.C), math.sqrt(w0 ** 2 - d ** 2))
        self.assertEqual(ef.eigenkreisfrequenz_gedaempft(1000.0, self.L, self.C), 0.0)   # ueberkritisch


class TestSchaltvorgaenge(Basis):
    def test_rc_rl(self):
        R, C = 1000.0, 1e-6
        tau = R * C
        self.nah(ef.rc_ladespannung(tau, 5, R, C), 5 * (1 - math.exp(-1)))          # 63,2 %
        self.nah(ef.rc_ladespannung(5 * tau, 1, R, C), 0.993262053, rel=1e-8)
        self.nah(ef.rc_entladespannung(tau, 5, R, C), 5 / math.e)
        self.nah(ef.rc_ladestrom(0, 5, R, C), 5e-3)
        self.nah(ef.rc_zeit_bis_anteil(R, C, 0.5), tau * math.log(2))
        self.nah(ef.rc_zeit_bis_anteil(R, C, 0.632120559), tau, rel=1e-8)
        self.nah(ef.rl_einschaltstrom(1e-3, 10, 10, 10e-3), 1 - math.exp(-1))
        self.nah(ef.rl_freilaufstrom(1e-3, 1, 10, 10e-3), 1 / math.e)

    def test_rlc_gegen_rk4(self):
        """Analytische Loesung (alle 3 Faelle) gegen numerische Integration der DGL."""
        L, C, U0 = 1e-3, 1e-6, 1.0
        Rkrit = ef.widerstand_aperiodisch(L, C)
        for R in (5.0, 0.6 * Rkrit, Rkrit, 3 * Rkrit):
            def f(z):
                uc, i = z
                return np.array([i / C, (U0 - R * i - uc) / L])
            dt, n = 2e-8, 10000
            z, t_ = np.array([0.0, 0.0]), np.arange(n + 1) * dt
            verlauf = [z[0]]
            for _ in range(n):
                k1 = f(z); k2 = f(z + dt / 2 * k1); k3 = f(z + dt / 2 * k2); k4 = f(z + dt * k3)
                z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                verlauf.append(z[0])
            self.nah(ef.rlc_sprung_uc(t_, R, L, C, U0), np.array(verlauf), rel=0, abs_=2e-6)
            # Randbedingungen
            self.nah(ef.rlc_sprung_uc(0.0, R, L, C, U0), 0.0, abs_=1e-12)
            self.nah(ef.rlc_sprung_uc(1.0, R, L, C, U0), U0, abs_=1e-9)

    def test_ueberschwingen(self):
        self.nah(ef.ueberschwingweite(0.5), 0.163033534, rel=1e-6)
        self.nah(ef.ueberschwingweite(1 / math.sqrt(2)), 0.043213918, rel=1e-6)
        self.assertEqual(float(ef.ueberschwingweite(1.0)), 0.0)
        self.assertEqual(float(ef.ueberschwingweite(2.0)), 0.0)
        # Gegenprobe: Maximum der Sprungantwort
        L, C, R = 1e-3, 1e-6, 20.0                       # zeta = 0,3162
        z = ef.daempfungsgrad_rlc(R, L, C)
        t = np.linspace(0, 2e-3, 200001)
        self.nah(ef.rlc_sprung_uc(t, R, L, C).max() - 1.0, ef.ueberschwingweite(z), rel=1e-6)


class TestFilter(Basis):
    def test_grenzfrequenz_bode(self):
        fc = ef.grenzfrequenz_rc(1000, 159.155e-9)
        self.nah(fc, 1000.0, rel=1e-5)
        self.nah(ef.grenzfrequenz_rl(100, 15.9155e-3), 1000.0, rel=1e-5)
        H = ef.tiefpass_1(1000, 1000)
        self.nah(abs(H), 1 / math.sqrt(2))
        self.nah(ef.betrag_db(H), -3.010299957, rel=1e-8)
        self.nah(ef.phase_grad(H), -45.0)
        Hh = ef.hochpass_1(1000, 1000)
        self.nah(abs(Hh), 1 / math.sqrt(2))
        self.nah(ef.phase_grad(Hh), 45.0)
        # Asymptotik: 1 Dekade oberhalb fc -> ca. -20 dB ; 2 Dekaden -> -40 dB
        self.nah(ef.betrag_db(ef.tiefpass_1(1e4, 1e3)), -20.0432, rel=1e-4)
        self.nah(ef.betrag_db(ef.tiefpass_1(1e5, 1e3)), -40.0004, rel=1e-5)
        # TP + HP: |Htp|^2 + |Hhp|^2 = 1
        f = np.logspace(0, 6, 50)
        self.nah(abs(ef.tiefpass_1(f, 1e3)) ** 2 + abs(ef.hochpass_1(f, 1e3)) ** 2, 1.0)

    def test_tiefpass_2(self):
        self.nah(abs(ef.tiefpass_2(1000, 1000, 2.0)), 2.0)                # |H(f0)| = Q
        self.nah(ef.phase_grad(ef.tiefpass_2(1000, 1000, 2.0)), -90.0)
                # Butterworth 2. Ordnung: Q = 1/sqrt 2 -> |H(f0)|^2 = 1/2 ... |H| = Q = 0,7071
        self.nah(abs(ef.tiefpass_2(1000, 1000, 1 / math.sqrt(2))), 1 / math.sqrt(2))
        # ... und identisch zur Butterworth-Formel n=2 fuer alle f
        f = np.logspace(1, 5, 60)
        self.nah(abs(ef.tiefpass_2(f, 1e3, 1 / math.sqrt(2))), ef.butterworth_betrag(f, 1e3, 2), rel=1e-12)

    def test_butterworth(self):
        for n in (1, 2, 3, 5, 8):
            self.nah(ef.butterworth_betrag(1000, 1000, n), 1 / math.sqrt(2))
        self.nah(ef.betrag_db(ef.butterworth_betrag(1e4, 1e3, 3)), -60.0, abs_=5e-3)     # 20 n dB/Dek
        self.nah(ef.steilheit_db_dekade(3), 60.0)
        n = ef.butterworth_ordnung(1000, 2000, 1.0, 40.0)
        self.nah(n, math.log10(9999 / (10 ** 0.1 - 1)) / (2 * math.log10(2)))
        self.assertEqual(math.ceil(n), 8)
        # Kontrolle: mit n=8 werden die Vorgaben eingehalten, mit n=7 nicht
        self.assertLessEqual(ef.betrag_db(ef.butterworth_betrag(1000, 1000 / (10 ** 0.1 - 1) ** (1 / 16), 8)), -0.99)
        fc8 = 1000 / (10 ** 0.1 - 1) ** (1 / 16)
        self.assertLess(ef.betrag_db(ef.butterworth_betrag(2000, fc8, 8)), -40.0)
        fc7 = 1000 / (10 ** 0.1 - 1) ** (1 / 14)
        self.assertGreater(ef.betrag_db(ef.butterworth_betrag(2000, fc7, 7)), -40.0)


class TestVerstaerkerHalbleiter(Basis):
    def test_opv(self):
        self.nah(ef.opv_invertierend(10e3, 1e3), -10.0)
        self.nah(ef.opv_nichtinvertierend(10e3, 1e3), 11.0)
        self.nah(ef.opv_grenzfrequenz(1e6, 100), 1e4)
        self.nah(ef.opv_grenzfrequenz(1e6, -100), 1e4)
        self.nah(ef.opv_slew_fmax(1e6, 10), 15915.494309, rel=1e-8)

    def test_rauschen(self):
        self.nah(ef.rauschspannung_thermisch(300, 1000, 1), 4.0704e-9, rel=1e-3)      # 4,07 nV/sqrt(Hz)
        self.nah(ef.rauschstrom_schrot(1e-3, 1), 1.7900e-11, rel=1e-3)                # 17,9 pA/sqrt(Hz)
        # Bandbreite: Rauschen ~ sqrt(B)
        self.nah(ef.rauschspannung_thermisch(300, 1000, 100), 10 * ef.rauschspannung_thermisch(300, 1000, 1))

    def test_halbleiter(self):
        self.nah(ef.thermospannung(300), 0.025852, rel=1e-4)                            # 25,85 mV
        self.nah(ef.diode_strom(0.0, 1e-12), 0.0, abs_=1e-30)
        # 1 Dekade Strom je UT*ln(10) = 59,5 mV bei 300 K (bei Is << I)
        dU = ef.thermospannung(300) * math.log(10)
        self.nah(ef.diode_strom(0.6 + dU, 1e-12) / ef.diode_strom(0.6, 1e-12), 10.0, rel=1e-6)
        self.assertAlmostEqual(dU * 1e3, 59.53, delta=0.05)
        self.nah(ef.diode_strom(-1.0, 1e-12), -1e-12, rel=1e-6)                         # Sperrsaettigung
        self.nah(ef.led_vorwiderstand(5, 2, 0.02), 150.0)
        self.nah(ef.brummspannung(0.1, 100, 1e-3), 1.0)
        self.nah(ef.mosfet_id_saettigung(2e-3, 3, 1), 4e-3)
        self.assertEqual(float(ef.mosfet_id_saettigung(2e-3, 0.5, 1)), 0.0)           # gesperrt
        self.nah(ef.bjt_ic(100, 20e-6), 2e-3)
        self.nah(ef.bjt_gm(1e-3, 300), 0.038680, rel=1e-3)                              # 38,7 mS


class TestLeitungen(Basis):
    def test_koax(self):
        self.nah(ef.wellenwiderstand_koax(1.0, math.e, 1.0), 376.730313668 / (2 * PI), rel=1e-8)   # 59,958 Ohm
        self.nah(ef.wellenwiderstand_koax(2.25, 3.5, 1.0), 50.08, rel=1e-3)                      # RG-58-artig
        # Zusammenhang: Z0 = 1/(v C') mit L'C' = 1/v^2 -> Z0 = sqrt(L'/C')
        eps_r, D, d = 2.25, 3.5e-3, 1e-3
        C_pl = ef.kapazitaet_koax(eps_r, 1.0, d / 2, D / 2)
        v = ef.phasengeschwindigkeit(eps_r)
        self.nah(1 / (v * C_pl), ef.wellenwiderstand_koax(eps_r, D, d), rel=1e-9)

    def test_reflexion(self):
        r = ef.reflexionsfaktor(100, 50)
        self.nah(r, 1 / 3)
        self.nah(ef.stehwellenverhaeltnis(r), 2.0)
        self.nah(ef.rueckflussdaempfung_db(r), 9.5424251, rel=1e-7)
        self.nah(ef.reflexionsfaktor(50, 50), 0.0)
        self.nah(ef.reflexionsfaktor(0, 50), -1.0)
        self.assertTrue(np.isinf(ef.stehwellenverhaeltnis(1.0)))
        # VSWR-Identitaet: s = ZL/Z0 (ZL > Z0, reell)
        self.nah(ef.stehwellenverhaeltnis(ef.reflexionsfaktor(150, 50)), 3.0)

    def test_freiraum_und_transformation(self):
        # Faustformel: 32,44 + 20 lg(d/km) + 20 lg(f/MHz)
        self.nah(ef.freiraumdaempfung_db(1000, 1e9), 92.44, abs_=0.02)
        self.nah(ef.freiraumdaempfung_db(2000, 1e9) - ef.freiraumdaempfung_db(1000, 1e9), 6.0206, rel=1e-4)
        # lambda/4-Transformator: Zin = Z0^2/ZL
        self.nah(ef.leitung_eingangsimpedanz(100.0, 50.0, PI / 2), 25.0 + 0j, abs_=1e-9)
        # lambda/2: Zin = ZL
        self.nah(ef.leitung_eingangsimpedanz(100.0, 50.0, PI), 100.0 + 0j, abs_=1e-9)
        # angepasst: Zin = Z0 unabhaengig von der Laenge
        self.nah(ef.leitung_eingangsimpedanz(50.0, 50.0, 0.7), 50.0 + 0j, abs_=1e-9)


class TestDrehstromTrafoMaschinen(Basis):
    def test_drehstrom(self):
        self.nah(ef.spannung_verkettet(230.94010767), 400.0, rel=1e-8)
        self.nah(ef.spannung_strang(400), 230.94010767, rel=1e-8)
        self.nah(ef.scheinleistung_drehstrom(400, 10), 6928.203230, rel=1e-8)
        self.nah(ef.wirkleistung_drehstrom(400, 10, 0.8), 5542.562584, rel=1e-8)
        self.nah(ef.blindleistung_drehstrom(400, 10, math.acos(0.8)), 4156.921938, rel=1e-8)
        self.nah(ef.strom_drehstrom(5542.562584, 400, 0.8), 10.0, rel=1e-8)
        self.nah(ef.leistung_dreieck_aus_stern(1000), 3000.0)
        self.nah(ef.spannungsfall_drehstrom(100, 0.1, 0.05, math.acos(0.8)), 19.0526, rel=1e-4)
        # Summenleistung dreier 1~-Lasten = 3 * Uph * Iph * cos = sqrt(3) UL IL cos
        Uph, I = 230.0, 10.0
        self.nah(3 * ef.wirkleistung(Uph, I, 0.6), ef.wirkleistung_drehstrom(ef.spannung_verkettet(Uph), I, math.cos(0.6)))
        # Dreiphasensumme der Momentanwerte = 0 ; verkettete Spannung numerisch
        t = np.linspace(0, 0.02, 2001)
        ph = [math.sqrt(2) * 230 * np.sin(2 * PI * 50 * t - k * 2 * PI / 3) for k in range(3)]
        self.nah(ph[0] + ph[1] + ph[2], 0.0, abs_=1e-9)
        self.nah(ef.effektivwert_numerisch(ph[0] - ph[1], t), 230 * math.sqrt(3), rel=1e-6)

    def test_drehstrom_kompensation(self):
        Cd = ef.kompensation_drehstrom_dreieck(10000, 50, 400)
        Cs = ef.kompensation_drehstrom_stern(10000, 50, 400)
        self.nah(Cd * 1e6, 66.315, rel=1e-4)
        self.nah(Cs, 3 * Cd)                          # Stern braucht die 3-fache Kapazitaet
        self.nah(3 * 2 * PI * 50 * Cd * 400 ** 2, 10000.0)

    def test_trafo(self):
        self.nah(ef.trafo_u2(230, 1000, 100), 23.0)
        self.nah(ef.trafo_i2(1, 1000, 100), 10.0)
        self.nah(ef.trafo_z_primaer(10, 1000, 100), 1000.0)
        self.nah(ef.trafo_hauptgleichung(50, 500, 1.5, 1e-3), 166.6, rel=1e-3)
        self.nah(ef.trafo_wirkungsgrad(10000, 100, 200), 10000 / 10300)
        self.nah(ef.trafo_kurzschlussstrom(100, 0.04), 2500.0)
        # Leistungserhaltung: U1 I1 = U2 I2 ; Z' = U1/I1
        U1, N1, N2, I1 = 230.0, 1000.0, 100.0, 1.0
        self.nah(U1 * I1, ef.trafo_u2(U1, N1, N2) * ef.trafo_i2(I1, N1, N2))
        self.nah(U1 / I1, ef.trafo_z_primaer(ef.trafo_u2(U1, N1, N2) / ef.trafo_i2(I1, N1, N2), N1, N2))
        # Hauptgleichung aus Induktionsgesetz numerisch: Uhat = N * omega * B_hat * A
        f, N, B, A = 50.0, 500.0, 1.5, 1e-3
        self.nah(ef.scheitelwert_sinus(ef.trafo_hauptgleichung(f, N, B, A)), N * 2 * PI * f * B * A)

    def test_asm(self):
        self.nah(ef.synchrondrehzahl(50, 2) * 60, 1500.0)
        self.nah(ef.schlupf(1500 / 60, 1440 / 60), 0.04)
        self.nah(ef.laeuferfrequenz(0.04, 50), 2.0)
        self.nah(ef.drehmoment(7500, 24), 49.7359197, rel=1e-7)
        self.nah(ef.leistung_mechanisch(24, ef.drehmoment(7500, 24)), 7500.0)
        self.nah(ef.kloss_moment(100, 0.2, 0.2), 100.0)                            # M(sk) = Mk
        self.nah(ef.kloss_moment(100, 0.1, 0.2), 80.0)                             # 2/(0,5+2) = 0,8
        self.nah(ef.kloss_moment(100, 0.4, 0.2), 80.0)                             # symmetrisch um sk
        self.nah(ef.kloss_moment(100, 1e-9, 0.2), 0.0, abs_=1e-6)
        self.nah(ef.wirkungsgrad(7500, 8500), 0.882352941, rel=1e-8)
        self.nah(ef.motor_strom_drehstrom(7500, 400, 0.9, 0.85), 14.1518, rel=1e-4)

    def test_gleichstrommaschine(self):
        k, w, Ra, Ia = 1.0, 100.0, 0.5, 10.0
        U = ef.dcm_ankerspannung(k, w, Ra, Ia)
        self.nah(U, 105.0)
        self.nah(ef.dcm_drehmoment(k, Ia), 10.0)
        self.nah(ef.dcm_omega(U, Ra, Ia, k), 100.0)
        # Energiebilanz: innere Leistung E*Ia = M*omega ; Klemmen: U*Ia = E*Ia + Ra*Ia^2
        self.nah(k * w * Ia, ef.dcm_drehmoment(k, Ia) * w)
        self.nah(U * Ia, k * w * Ia + Ra * Ia ** 2)
        self.nah(ef.hochlaufzeit(0.05, 150, 15), 0.5)
        # Hochlaufzeit numerisch: J dw/dt = M  -> t = J w / M
        J, M, dt, w_ = 0.05, 15.0, 1e-5, 0.0
        for _ in range(50000):
            w_ += M / J * dt
        self.nah(w_, 150.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Einheitenumrechnungen
# ---------------------------------------------------------------------------
class TestEinheiten(Basis):
    def test_praefixe(self):
        self.nah(ef.si_praefix(4700, "n", "µ"), 4.7)
        self.nah(ef.si_praefix(1.5, "M", "k"), 1500.0)
        self.nah(ef.si_praefix(2.2, "", "m"), 2200.0)
        self.nah(ef.si_praefix(1, "G", "p"), 1e21)
        for w, soll in ((0.00047, (470.0, "µ")), (2.2e-11, (22.0, "p")), (1.5e6, (1.5, "M")), (999.0, (999.0, "")),
                        (1000.0, (1.0, "k")), (-4700.0, (-4.7, "k"))):
            m, p = ef.auto_praefix(w)
            self.nah(m, soll[0], rel=1e-9)
            self.assertEqual(p, soll[1])
        self.assertEqual(ef.formatiere_si(4700, "Ω"), "4.7 kΩ")
        self.assertEqual(ef.formatiere_si(1e-6, "F"), "1 µF")
        self.assertEqual(ef.auto_praefix(0.0), (0.0, ""))

    def test_db(self):
        self.nah(ef.db_leistung(2), 3.010299957, rel=1e-8)
        self.nah(ef.db_leistung(10), 10.0)
        self.nah(ef.db_spannung(10), 20.0)
        self.nah(ef.db_spannung(2), 6.020599913, rel=1e-8)
        self.nah(ef.von_db_leistung(3.0103), 2.0, rel=1e-4)
        self.nah(ef.von_db_spannung(20), 10.0)
        self.nah(ef.von_db_spannung(ef.db_spannung(0.123)), 0.123)
        self.nah(ef.von_db_leistung(ef.db_leistung(123.4)), 123.4)
        self.nah(ef.watt_zu_dbm(1e-3), 0.0, abs_=1e-12)
        self.nah(ef.watt_zu_dbm(1.0), 30.0)
        self.nah(ef.dbm_zu_watt(30), 1.0)
        self.nah(ef.dbm_zu_watt(-30), 1e-6)
        self.nah(ef.volt_zu_dbv(0.1), -20.0)
        self.nah(ef.dbv_zu_volt(-20), 0.1)
        self.nah(ef.volt_zu_dbuv(1e-3), 60.0)
        self.nah(ef.dbuv_zu_volt(60), 1e-3)
        self.nah(ef.dbm_zu_dbuv(0.0, 50.0), 106.9897, rel=1e-6)                    # dBuV = dBm + 107 (50 Ohm)
        # Gegenprobe ueber P = U^2/R: 0 dBm an 50 Ohm -> U = sqrt(1 mW * 50 Ohm) = 0,2236 V
        U = math.sqrt(1e-3 * 50)
        self.nah(ef.volt_zu_dbuv(U), ef.dbm_zu_dbuv(0.0, 50.0), rel=1e-9)
        self.nah(ef.neper_zu_db(1), 8.685889638, rel=1e-8)
        self.nah(ef.db_zu_neper(ef.neper_zu_db(2.5)), 2.5)
        # 1 Np entspricht Spannungsverhaeltnis e
        self.nah(ef.db_spannung(math.e), ef.neper_zu_db(1))

    def test_energie_leistung(self):
        self.nah(ef.kwh_zu_joule(1), 3.6e6)
        self.nah(ef.joule_zu_kwh(3.6e6), 1.0)
        self.nah(ef.wh_zu_joule(1), 3600.0)
        self.nah(ef.ev_zu_joule(1), 1.602176634e-19)
        self.nah(ef.joule_zu_ev(ef.ev_zu_joule(5.0)), 5.0)
        self.nah(ef.cal_zu_joule(1), 4.184)
        self.nah(ef.ah_zu_coulomb(1), 3600.0)
        self.nah(ef.coulomb_zu_ah(7200), 2.0)
        self.nah(ef.ps_zu_watt(1), 735.49875)
        self.nah(ef.hp_zu_watt(1), 745.6998716, rel=1e-9)
        self.nah(ef.watt_zu_ps(735.49875), 1.0)
        # Akku 2 Ah bei 0,5 A -> 4 h
        self.nah(ef.batterie_laufzeit(ef.ah_zu_coulomb(2.0), 0.5) / 3600, 4.0)

    def test_temperatur(self):
        self.nah(ef.celsius_zu_kelvin(0), 273.15)
        self.nah(ef.celsius_zu_kelvin(25), 298.15)
        self.nah(ef.kelvin_zu_celsius(300), 26.85, rel=1e-12)
        self.nah(ef.celsius_zu_fahrenheit(100), 212.0)
        self.nah(ef.celsius_zu_fahrenheit(-40), -40.0)
        self.nah(ef.celsius_zu_fahrenheit(37), 98.6)
        self.nah(ef.fahrenheit_zu_celsius(212), 100.0)
        self.nah(ef.fahrenheit_zu_celsius(ef.celsius_zu_fahrenheit(21.5)), 21.5)

    def test_winkel_drehzahl(self):
        self.nah(ef.grad_zu_rad(180), PI)
        self.nah(ef.rad_zu_grad(PI / 2), 90.0)
        self.nah(ef.hz_zu_rads(50), 314.1592654)
        self.nah(ef.rads_zu_hz(2 * PI), 1.0)
        self.nah(ef.rpm_zu_rads(60), 2 * PI)
        self.nah(ef.rpm_zu_rads(1500), 50 * PI)
        self.nah(ef.rads_zu_rpm(2 * PI), 60.0)

    def test_laengen_awg(self):
        self.nah(ef.inch_zu_meter(1), 0.0254)
        self.nah(ef.mil_zu_meter(1000), 0.0254)
        self.nah(ef.cmil_zu_m2(1), 5.067074791e-10, rel=1e-8)
        # Tabellenwerte AWG (ASTM B258)
        for n, d_mm in ((0, 8.251), (10, 2.588), (20, 0.8128), (30, 0.2546), (36, 0.127), (40, 0.0799)):
            self.nah(ef.awg_durchmesser(n) * 1e3, d_mm, rel=2e-3)
        self.nah(ef.awg_durchmesser(-3) * 1e3, 11.684, rel=1e-3)                     # 4/0
        self.nah(ef.awg_querschnitt(10) * 1e6, 5.26, rel=2e-3)                        # 5,26 mm^2
        # 6 AWG-Stufen halbieren den Durchmesser (Faktor 2, Regel: 6 Stufen ~ 2x)
        self.nah(ef.awg_durchmesser(10) / ef.awg_durchmesser(16), 2.0, rel=1e-2)
        # 3 AWG-Stufen verdoppeln den Querschnitt (~ Faustregel)
        self.nah(ef.awg_querschnitt(10) / ef.awg_querschnitt(13), 2.0, rel=1e-2)
        self.nah(ef.durchmesser_aus_querschnitt(math.pi / 4 * 4e-6), 2e-3)

    def test_leitfaehigkeit_magnetik(self):
        self.nah(ef.rho_mm2_zu_si(0.0172), 1.72e-8)
        self.nah(ef.rho_si_zu_mm2(1.72e-8), 0.0172)
        self.nah(ef.kappa_mm2_zu_si(58.0), 5.8e7)
        self.nah(1 / ef.kappa_mm2_zu_si(58.0), 1.7241379e-8, rel=1e-6)
        self.nah(ef.sigma_zu_iacs(5.8e7), 100.0)
        self.nah(ef.sigma_zu_iacs(3.5e7), 60.34, rel=1e-3)                           # Aluminium ~ 61 %
        self.nah(ef.gauss_zu_tesla(10000), 1.0)
        self.nah(ef.tesla_zu_gauss(1.5), 15000.0)
        self.nah(ef.oersted_zu_a_pro_m(1), 79.57747155, rel=1e-8)
        self.nah(ef.a_pro_m_zu_oersted(ef.oersted_zu_a_pro_m(3.3)), 3.3)
        self.nah(ef.maxwell_zu_weber(1e8), 1.0)
        self.nah(ef.gilbert_zu_ampere(1), 0.7957747155, rel=1e-8)
        # Cu-Draht: A [mm^2] -> R aus beiden Schreibweisen identisch
        R1 = ef.widerstand_leiter(ef.rho_mm2_zu_si(0.0172), 100, 1.5e-6)
        R2 = 0.0172 * 100 / 1.5
        self.nah(R1, R2)

    def test_e_reihen_farbcode(self):
        self.nah(ef.naechster_e_wert(4600, "E12"), 4700.0)
        self.nah(ef.naechster_e_wert(1100, "E6"), 1000.0)
        self.nah(ef.naechster_e_wert(1400, "E6"), 1500.0)
        self.nah(ef.naechster_e_wert(5000, "E24"), 5100.0)
        self.nah(ef.naechster_e_wert(0.0033, "E12"), 0.0033)
        self.nah(ef.naechster_e_wert(9700, "E12"), 10000.0)                           # Dekadenuebergang
        self.assertEqual(len(ef.E_REIHEN["E12"]), 12)
        self.assertEqual(len(ef.E_REIHEN["E24"]), 24)
        # E-Reihen: jeweils Verhaeltnis benachbarter Werte ~ 10^(1/n) (Rundung erlaubt)
        for name, n in (("E6", 6), ("E12", 12), ("E24", 24)):
            reihe = ef.E_REIHEN[name] + (10.0,)
            for a, b in zip(reihe[:-1], reihe[1:]):
                self.assertAlmostEqual(math.log10(b / a), 1 / n, delta=0.025, msg=(name, a, b))   # IEC 60063 nutzt gerundete Werte
        self.assertEqual(ef.widerstand_farbcode("gelb", "violett", "rot", "gold"), (4700.0, 5.0))
        self.assertEqual(ef.widerstand_farbcode("braun", "schwarz", "schwarz", "rot", "braun"), (10000.0, 1.0))
        w, tol = ef.widerstand_farbcode("rot", "rot", "silber", "silber")
        self.nah(w, 0.22)
        self.assertEqual(tol, 10.0)
        self.assertEqual(ef.widerstand_farbcode("braun", "schwarz", "gold", "gold")[0], 1.0)   # 1 Ohm


if __name__ == "__main__":
    unittest.main(verbosity=1)
