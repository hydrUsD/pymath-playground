#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifikation der Ergaenzungen aus dem Abgleich mit Elektronik Kompendium (python test_ek_abdeckung.py)

  A) Abdeckung      - jede der 343 EK-Themenseiten ist zugeordnet, jede genannte Funktion existiert
  B) Referenzwerte  - Hand-/Datenblattwerte (Literale, nicht aus dem Modul kopiert)
  C) Gegenrechnung  - Umkehrfunktionen, Identitaeten, Grenzfaelle, Wahrheitstabellen
"""
import math
import unittest
from fractions import Fraction

import numpy as np

import ek_abdeckung as ea
import et_formeln as ef

PI = math.pi


class Basis(unittest.TestCase):
    def nah(self, ist, soll, rel=1e-9, abs_=0.0):
        np.testing.assert_allclose(ist, soll, rtol=rel, atol=abs_)


# ---------------------------------------------------------------------------
# A) Abdeckung
# ---------------------------------------------------------------------------
class TestAbdeckung(Basis):
    def test_alle_ek_seiten_zugeordnet(self):
        self.assertEqual(len(ea.alle_slugs()), 343)
        for gruppe in ea.ABDECKUNG.values():
            for slug, namen in gruppe.items():
                self.assertTrue(namen, slug)

    def test_alle_funktionen_existieren(self):
        self.assertEqual(ea.fehlende_funktionen(), {})

    def test_hinweise_nur_fuer_bekannte_seiten(self):
        self.assertLessEqual(set(ea.HINWEISE), set(ea.alle_slugs()))

    def test_neue_formeln_dimensionshomogen(self):
        self.assertTrue(ef.pruefe_alle_dimensionen())


# ---------------------------------------------------------------------------
# B/C) Nichtlineare Widerstaende
# ---------------------------------------------------------------------------
class TestNichtlineareWiderstaende(Basis):
    def test_ntc(self):
        # 10 kOhm / B 3950 bei 50 °C: 10k * exp(3950 (1/323,15 - 1/298,15)) = 3588 Ohm (Datenblatt ~3,6 k)
        self.nah(ef.ntc_widerstand(323.15, 10e3, 3950.0, 298.15), 3588.1, rel=1e-4)
        self.nah(ef.ntc_widerstand(298.15, 10e3, 3950.0, 298.15), 10e3)
        T = np.linspace(250.0, 400.0, 7)
        R = ef.ntc_widerstand(T, 10e3, 3950.0, 298.15)
        self.nah(ef.ntc_temperatur(R, 10e3, 3950.0, 298.15), T)                       # Umkehrung
        R85 = ef.ntc_widerstand(358.15, 10e3, 3950.0, 298.15)
        self.nah(ef.ntc_b_wert(298.15, 10e3, 358.15, R85), 3950.0)
        self.nah(ef.ntc_tk(3950.0, 298.15), -0.04443, rel=1e-3)                        # ~ -4,4 %/K
        # TK = dR/dT / R numerisch
        dRdT = ef.ableitung(lambda t: ef.ntc_widerstand(t, 10e3, 3950.0, 298.15), 298.15)
        self.nah(dRdT / 10e3, ef.ntc_tk(3950.0, 298.15), rel=1e-6)
        self.nah(ef.eigenerwaermung(10e-3, 1.5e-3), 6.667, rel=1e-3)

    def test_ptc(self):
        self.nah(ef.ptc_widerstand_linear(100.0, 0.00385, 373.15, 273.15), 138.5)      # Pt100 bei 100 °C
        self.nah(ef.temperaturkoeffizient(100.0, 138.5, 100.0), 0.00385)
        self.nah(ef.ptc_widerstand_exp(100.0, 0.15, 393.15, 393.15), 100.0)
        self.nah(ef.ptc_widerstand_exp(100.0, 0.15, 403.15, 393.15), 448.17, rel=1e-4)  # 100 e^1,5

    def test_vdr(self):
        I = ef.vdr_strom(np.array([200.0, 270.0, 350.0]), 270.0, 1e-3, 30.0)
        self.nah(I[1], 1e-3)
        self.nah(ef.vdr_spannung(I, 270.0, 1e-3, 30.0), [200.0, 270.0, 350.0])
        self.nah(ef.vdr_alpha(200.0, I[0], 350.0, I[2]), 30.0)
        self.nah(ef.vdr_strom(-270.0, 270.0, 1e-3, 30.0), -1e-3)                       # symmetrisch
        # Form R = R0 (U/U0)^n mit R0 = U_ref/I_ref, n = 1 - alpha
        self.nah(ef.vdr_widerstand(300.0, 270.0, 1e-3, 30.0), 270e3 * (300 / 270) ** (1 - 30))

    def test_ldr_feldplatte(self):
        self.nah(ef.ldr_widerstand(10.0, 10e3, 10.0, 0.7), 10e3)
        self.nah(ef.ldr_widerstand(1000.0, 10e3, 10.0, 0.5), 1e3)                      # 100x heller, gamma 0,5
        R = ef.ldr_widerstand(250.0, 10e3, 10.0, 0.7)
        self.nah(ef.ldr_beleuchtungsstaerke(R, 10e3, 10.0, 0.7), 250.0)
        self.nah(ef.ldr_gamma(10.0, 10e3, 250.0, R), 0.7)
        self.nah(ef.feldplatte_widerstand(0.0, 100.0, 7.7), 100.0)
        self.nah(ef.feldplatte_widerstand(0.1, 100.0, 7.7), 100.0 * (1 + 0.77 ** 2))


# ---------------------------------------------------------------------------
# Wechselstrom, Drehstrom, Trafo, Schwingkreise, Filter
# ---------------------------------------------------------------------------
class TestWechselDrehstrom(Basis):
    def test_wechselstrom(self):
        U, phi = ef.zeigeraddition(10.0, 0.0, 10.0, PI / 2)
        self.nah(U, 14.142136, rel=1e-6)
        self.nah(phi, PI / 4)
        self.nah(ef.augenblickswert(325.0, 50.0, 0.005, 0.0), 325.0)                   # Scheitel bei T/4
        self.nah(ef.effektivwert_puls(5.0, 0.25), 2.5)
        self.nah(ef.effektivwert_einweg(10.0), 5.0)
        self.nah(ef.gleichrichtwert_einweg(PI), 1.0)
        self.nah(ef.scheitelfaktor(np.sqrt(2) * 230, 230), np.sqrt(2))
        self.nah(ef.formfaktor(1 / np.sqrt(2), 2 / PI), 1.1107, rel=1e-4)              # Sinus
        self.nah(ef.blindleistung_kondensator(230.0, 50.0, 10e-6), 166.19, rel=1e-4)
        self.nah(ef.blindleistung_spule(230.0, 50.0, 0.5), 336.77, rel=1e-4)
        self.nah(ef.kapazitaet_aus_blindleistung(166.19, 50.0, 230.0), 10e-6, rel=1e-4)
        self.nah(ef.induktivitaet_aus_blindleistung(336.77, 50.0, 230.0), 0.5, rel=1e-4)
        self.nah(ef.scheinleistung_pq(3000.0, 4000.0), 5000.0)
        self.nah(ef.blindleistung_sp(5000.0, 3000.0), 4000.0)

    def test_parallelschaltung_r_x(self):
        R, X = 100.0, 50.0
        Z = 1 / (1 / R + 1 / (1j * X))
        self.nah(ef.z_parallel_betrag(R, X), abs(Z))
        self.nah(ef.phasenwinkel_parallel(R, X), np.angle(Z))
        self.nah(ef.phasenwinkel_parallel(R, -X), np.angle(1 / (1 / R + 1 / (-1j * X))))
        self.nah(np.cos(ef.phasenwinkel_parallel(R, X)), abs(Z) / R)                    # cos(phi) = Z/R
        self.nah(ef.strom_gesamt_parallel(3.0, 4.0), 5.0)

    def test_verlustfaktoren(self):
        self.nah(ef.verlustfaktor_spule(1000.0, 2.0, 10e-3) * ef.guete_spule(1000.0, 2.0, 10e-3), 1.0)
        self.nah(ef.guete_aus_verlustfaktor(0.01), 100.0)
        self.nah(ef.verlustfaktor_kondensator_reihe(1 / (2 * PI), 2.0, 0.5), 1.0)
        self.nah(ef.verlustfaktor_kondensator_parallel(1 / (2 * PI), 2.0, 0.5), 1.0)
        # P = U^2 omega C tan(delta) = Q_C tan(delta)
        self.nah(ef.verlustleistung_kondensator(230.0, 50.0, 10e-6, 0.01), 1.6619, rel=1e-4)

    def test_drehstrom_symmetrisch(self):
        self.nah(ef.leiterstrom_dreieck(10.0), 17.3205, rel=1e-5)
        self.nah(ef.strangstrom_dreieck(ef.leiterstrom_dreieck(10.0)), 10.0)
        self.nah(ef.neutralleiterstrom(10.0, 10.0, 10.0, 0.0, 0.0, 0.0), 0.0, abs_=1e-12)
        I1, I2, I3, IN, S = ef.drehstrom_stern_unsymmetrisch(230.0, 23.0, 23.0, 23.0)
        self.nah(abs(IN), 0.0, abs_=1e-12)
        self.nah(S, 3 * 230.0 ** 2 / 23.0)
        # Dreieck symmetrisch: I_L = sqrt(3) * U_L / Z, S = 3 U_L^2 / Z*
        I1, I2, I3, S = ef.drehstrom_dreieck_unsymmetrisch(400.0, 40 + 30j, 40 + 30j, 40 + 30j)
        self.nah([abs(I1), abs(I2), abs(I3)], [np.sqrt(3) * 8.0] * 3)
        self.nah(S, 3 * 400.0 ** 2 / (40 - 30j))
        self.nah(ef.wirkleistung_drehstrom_strang(400.0 / np.sqrt(3), 10.0, 0.8),
                 ef.wirkleistung_drehstrom(400.0, 10.0, 0.8))
        self.nah(ef.generatorfrequenz(2.0, 25.0), 50.0)
        self.nah(ef.phasenversatz(3.0), 2 * PI / 3)

    def test_drehstrom_unsymmetrisch(self):
        # ohmsche Last 10 A / 5 A / 5 A -> I_N = 5 A (nicht sqrt(150) = 12,2 A)
        self.nah(ef.neutralleiterstrom(10.0, 5.0, 5.0, 0.0, 0.0, 0.0), 5.0)
        I1, I2, I3, IN, S = ef.drehstrom_stern_unsymmetrisch(230.0, 23.0, 46.0, 46.0)
        self.nah(abs(IN), 5.0)
        self.nah([abs(I1), abs(I2), abs(I3)], [10.0, 5.0, 5.0])
        self.nah(S.real, 230.0 * 20.0)
        # Knotenregel im Dreieck: Summe der Leiterstroeme = 0
        I1, I2, I3, S = ef.drehstrom_dreieck_unsymmetrisch(400.0, 40.0, 40 + 30j, 80.0)
        self.nah(abs(I1 + I2 + I3), 0.0, abs_=1e-12)
        self.nah(S.real, 400.0 ** 2 / 40.0 + 400.0 ** 2 * 40 / 2500 + 400.0 ** 2 / 80.0)

    def test_trafo(self):
        self.nah(ef.trafo_fluss(230.0, 50.0, 500.0), 230.0 / (4.44 * 50 * 500), rel=1e-3)
        self.nah(ef.trafo_hauptgleichung(50.0, 500.0, ef.trafo_fluss(230.0, 50.0, 500.0), 1.0), 230.0)
        self.nah(ef.trafo_kupferverluste(1.0, 2.0, 10.0, 0.02), 4.0)
        self.nah(ef.trafo_uk(9.2, 230.0), 0.04)

    def test_schwingkreise(self):
        L, C, R = 1e-3, 1e-6, 10.0
        f0 = ef.resonanzfrequenz(L, C)
        self.nah(ef.resonanzkreisfrequenz(L, C), 2 * PI * f0)
        self.nah(ef.bandbreite_reihe(R, L), f0 / ef.guete_reihe(R, L, C))
        self.nah(ef.resonanzwiderstand_parallel(L, C, 1.0), 1000.0)
        self.nah(ef.spannungsueberhoehung(1.0, 50.0), 50.0)
        self.nah(ef.anzapfung_uebersetzung(100e-6, 25e-6), 2.0)
        self.nah(ef.anzapfung_widerstand(50.0, 100e-6, 25e-6), 200.0)
        self.nah(ef.eigenfrequenz_feder(100.0, 1.0), 1.59155, rel=1e-5)
        self.nah(ef.daempfungsgrad_aus_guete(10.0), 0.05)
        self.nah(ef.log_dekrement(1.0, np.exp(-1.0), 1.0), 1.0)
        # Energieerhaltung: L i^2/2 bei u = 0 gleich C u^2/2 bei i = 0
        self.nah(ef.energie_schwingkreis(L, 0.1, C, 0.0),
                 ef.energie_schwingkreis(L, 0.0, C, 0.1 * np.sqrt(L / C)))

    def test_filter(self):
        R, C = 1000.0, 159.155e-9
        fc = ef.grenzfrequenz_rc(R, C)
        self.nah(ef.rc_tiefpass_betrag(fc, R, C), 1 / np.sqrt(2), rel=1e-9)
        self.nah(ef.rc_hochpass_betrag(fc, R, C), 1 / np.sqrt(2), rel=1e-9)
        self.nah(ef.rc_tiefpass_phase(fc, R, C), -PI / 4)
        self.nah(ef.rc_hochpass_phase(fc, R, C), PI / 4)
        f = np.logspace(1, 5, 9)
        self.nah(ef.rc_tiefpass_betrag(f, R, C), np.abs(ef.tiefpass_1(f, fc)))
        self.nah(ef.rc_hochpass_phase(f, R, C), np.angle(ef.hochpass_1(f, fc)))
        L = 15.9e-3
        fl = ef.grenzfrequenz_rl(100.0, L)
        self.nah(ef.rl_tiefpass_betrag(fl, 100.0, L), 1 / np.sqrt(2))
        self.nah(ef.rl_hochpass_betrag(fl, 100.0, L), 1 / np.sqrt(2))
        self.nah(ef.siebfaktor_rc(f, R, C), 1 / ef.rc_tiefpass_betrag(f, R, C))
        self.nah(ef.lc_tiefpass(100.0, 0.1, 1e-3), 1 / (1 - (2 * PI * 100) ** 2 * 0.1e-3))
        self.nah(abs(ef.lc_tiefpass(100.0, 0.1, 1e-3)), 1 / ef.siebfaktor_lc(100.0, 0.1, 1e-3))
        self.nah(ef.mittenfrequenz(300.0, 3000.0), 948.68, rel=1e-5)
        # Bandpass + Bandsperre = 1, Sperre bei f0 = 0, Pass bei f0 = 1
        f = np.linspace(500.0, 2000.0, 7)
        self.nah(ef.bandpass_2(f, 1000.0, 5.0) + ef.bandsperre_2(f, 1000.0, 5.0), np.ones(7), rel=1e-12)
        self.nah(abs(ef.bandsperre_2(1000.0, 1000.0, 5.0)), 0.0, abs_=1e-15)
        self.nah(abs(ef.bandpass_2(1000.0, 1000.0, 5.0)), 1.0)
        self.nah(abs(ef.bandpass_2(1000.0 * (np.sqrt(1 + 1 / 100) + 0.1), 1000.0, 5.0)), 1 / np.sqrt(2))


# ---------------------------------------------------------------------------
# Halbleiter, Verstaerker, Kippschaltungen, Netzgeraete, Leistungselektronik
# ---------------------------------------------------------------------------
class TestElektronik(Basis):
    def test_dioden(self):
        self.nah(ef.sperrschichtkapazitaet(30e-12, 0.0, 0.7, 0.5), 30e-12)
        self.nah(ef.sperrschichtkapazitaet(30e-12, 2.1, 0.7, 0.5), 15e-12)               # (1 + 3)^0,5 = 2
        self.nah(ef.kapazitaetsverhaeltnis(0.0, 2.1, 0.7, 0.5), 2.0)
        self.nah(ef.zdiode_vorwiderstand(12.0, 5.1, 0.01, 0.02), 230.0)
        self.nah(ef.zdiode_strom(12.0, 5.1, 230.0, 0.02), 0.01)
        rmin, rmax = ef.zdiode_vorwiderstand_bereich(11.0, 14.0, 5.1, 0.005, 0.08, 0.0, 0.03)
        self.nah([rmin, rmax], [111.25, 168.571], rel=1e-5)
        self.assertLess(rmin, rmax)
        self.nah(ef.reihenstabilisierung_ua(5.6, 0.7), 4.9)
        self.nah(ef.brummspannung_effektiv(2 * np.sqrt(3)), 1.0)
        self.nah(ef.gleichspannung_geglaettet(16.3, 1.0), 15.8)

    def test_bjt(self):
        self.nah(ef.bjt_beta_aus_alpha(0.99), 99.0)
        self.nah(ef.bjt_alpha_aus_beta(ef.bjt_beta_aus_alpha(0.995)), 0.995)
        self.nah(ef.bjt_re(1e-3, 300.0), 25.852, rel=1e-4)                             # ~26 mV / 1 mA
        self.nah(ef.bjt_basisvorwiderstand(12.0, 0.7, 50e-6), 226e3)
        self.nah(ef.bjt_uce(12.0, 2e-3, 2.2e3, 0.0, 0.0), 7.6)
        # Spannungsteiler 47k/10k an 12 V, R_E 1k, B -> unendlich: I_E = (2,105 - 0,7) / 1k
        self.nah(ef.bjt_ie_spannungsteiler(12.0, 47e3, 10e3, 1e3, 1e12, 0.7), 1.40526e-3, rel=1e-5)
        self.assertLess(ef.bjt_ie_spannungsteiler(12.0, 47e3, 10e3, 1e3, 100.0, 0.7), 1.40526e-3)
        self.nah(ef.emitterschaltung_vu(4.7e3, 0.0, 26.0), -180.77, rel=1e-4)
        self.nah(ef.emitterschaltung_vu(4.7e3, 470.0, 13.0), -9.731, rel=1e-4)
        self.nah(ef.kollektorschaltung_vu(1e3, 0.0), 1.0)
        self.nah(ef.schalter_rb(5.0, 0.7, 0.1, 100.0, 3.0), 1433.33, rel=1e-5)
        self.nah(ef.koppelkondensator(20.0, 10e3), 795.77e-9, rel=1e-5)

    def test_fet_opto_hall(self):
        k, Ugs, Uth = 2e-3, 3.0, 1.0
        Usat = ef.fet_uds_sat(Ugs, Uth)
        self.nah(ef.fet_id_ohmsch(k, Ugs, Uth, Usat), ef.mosfet_id_saettigung(k, Ugs, Uth))   # stetig am Uebergang
        gm_num = ef.ableitung(lambda u: ef.mosfet_id_saettigung(k, u, Uth), Ugs)
        self.nah(ef.fet_gm(k, Ugs, Uth), gm_num, rel=1e-6)
        self.nah(ef.sourceschaltung_vu(4e-3, 4.7e3, 1e30), -18.8)
        self.nah(ef.drainschaltung_vu(4e-3, 1e3), 0.8)
        self.nah(ef.fet_restspannung(5.0, 0.02), 0.1)
        self.nah(ef.fet_schaltverluste(48.0, 5.0, 100e3, 20e-9, 30e-9), 0.6)
        self.nah(ef.fotostrom_quanten(1.0, 1.0, ef.C0 / 850e-9), 0.6856, rel=1e-3)   # 850 nm: 0,686 A/W ideal
        self.nah(ef.hall_flussdichte(ef.hall_spannung(0.01, 0.5, 1e22, 1e-4), 1e22, 1e-4, 0.01), 0.5)

    def test_verstaerker(self):
        self.nah(ef.verstaerkung_gegenkopplung(1e5, 0.1), 9.9990, rel=1e-5)             # ~ 1/k
        self.nah(ef.rueckkopplungsfaktor(9e3, 1e3), 0.1)
        self.nah(1 / ef.rueckkopplungsfaktor(9e3, 1e3), ef.opv_nichtinvertierend(9e3, 1e3))
        self.nah(ef.verstaerkung_kette(10.0, 20.0, 5.0), 1000.0)
        self.nah(ef.db_spannung(ef.verstaerkung_kette(10.0, 100.0)), 60.0)              # dB addieren sich
        self.nah(ef.leistungsverstaerkung_vu(10.0, 1e3, 10.0), 1e4)
        self.nah(ef.opv_differenz(10e3, 1e3, 1.0, 1.2), 2.0)
        self.nah(ef.opv_summierer(10e3, 10e3, 5e3, 0.5, 0.25), -1.0)
        self.nah(ef.opv_summierer_n(10e3, [10e3, 5e3], [0.5, 0.25]), -1.0)
        t = np.linspace(0.0, 2e-3, 201)
        self.nah(ef.opv_integrator(t, np.ones_like(t), 10e3, 100e-9, 0.0)[-1], -2.0)
        self.nah(ef.opv_differenzierer(t, 1000.0 * t, 10e3, 100e-9), -np.ones_like(t))
        f = 1 / (2 * PI * 10e3 * 100e-9)
        self.nah(ef.opv_integrator_betrag(f, 10e3, 100e-9), 1.0)
        self.nah(ef.opv_differenzierer_betrag(f, 10e3, 100e-9), 1.0)
        fg = 1 / (2 * PI * 10e3 * 15.9e-9)
        self.nah(ef.opv_tiefpass_betrag(fg, 10e3, 1e3, 15.9e-9), 10 / np.sqrt(2))
        self.nah(ef.opv_hochpass_betrag(1e9, 10e3, 1e3, 159e-9), 10.0, rel=1e-9)
        self.nah(ef.ausgangsleistung_sinus(10.0, 8.0), 6.25)
        self.nah(ef.wirkungsgrad_b_endstufe(12.0, 12.0), PI / 4)
        self.nah(ef.klirrfaktor(1.0, 0.03, 0.04), 0.05)
        self.nah(ef.klirrfaktor_gesamt(1.0, 0.03, 0.04), 0.05 / np.sqrt(1.0025))
        self.nah(ef.snr_db(1e-3, 1e-9), 60.0)
        self.nah(ef.rauschleistung(290.0, 1.0), 4.0039e-21, rel=1e-4)                  # -174 dBm/Hz
        self.nah(ef.watt_zu_dbm(ef.rauschleistung(290.0, 1.0)), -173.98, rel=1e-4)
        self.nah(ef.rauschtemperatur(2.0, 290.0), 290.0)                                # NF 3 dB -> 290 K
        self.nah(ef.friis([2.0, 10.0], [100.0]), 2.09)

    def test_kippschaltungen(self):
        th, tl, f = ef.ne555_astabil(1e3, 10e3, 100e-9)
        self.nah(f, 687.0, rel=1e-3)
        self.nah(th / (th + tl), 11 / 21)                                               # Tastgrad
        self.nah(ef.ne555_monostabil(100e3, 10e-6), 1.0986, rel=1e-4)                   # ~1,1 R C
        self.nah(ef.multivibrator_periode(47e3, 10e-6, 47e3, 10e-6), 0.6516, rel=1e-4)
        # exakt geht fuer U_CC >> U_BE gegen ln2 R C
        self.nah(ef.kippzeit_transistor(1.0, 1.0, 1e9, 0.7, 0.2), np.log(2.0), rel=1e-8)
        self.nah(ef.opv_astabil_periode(10e3, 100e-9, 10e3, 10e3), 2 * 1e-3 * np.log(3.0))
        self.nah(ef.opv_monoflop(10e3, 100e-9, 20e3, 10e3), 1e-3 * np.log(3.0))
        self.nah(ef.schmitt_invertierend(12.0, 10e3, 40e3, 0.0), (2.4, -2.4))
        self.nah(ef.schmitt_invertierend(12.0, 10e3, 40e3, 5.0), (6.4, 1.6))
        self.nah(ef.schmitt_nichtinvertierend(12.0, 10e3, 40e3), (3.0, -3.0))
        self.nah(ef.schaltschwellen(2.5, 0.5), (2.75, 2.25))
        self.nah(ef.saegezahn_frequenz(1e-3, 1e-6, 5.0), 200.0)
        self.nah(ef.saegezahn_anstieg(1e-3, 5e-3, 1e-6), 5.0)
        self.nah(ef.ujt_periode(10e3, 100e-9, 1 - np.exp(-1)), 1e-3)
        self.nah(ef.phasenschieber_frequenz(10e3, 10e-9), 649.75, rel=1e-4)
        self.nah(ef.wien_frequenz(10e3, 10e3, 10e-9, 10e-9), ef.grenzfrequenz_rc(10e3, 10e-9))
        self.nah(ef.rc_ladezeit(1.0, 1.0, 1.0, 0.0, 0.5), np.log(2.0))
        self.nah(ef.rc_ladezeit(1e3, 1e-6, 0.0, 10.0, 10.0 * np.exp(-1)), 1e-3)          # Entladen

    def test_netzgeraete(self):
        self.nah(ef.laengsregler_ua(1.25, 240.0, 720.0, 0.0), 5.0)
        self.nah(ef.laengsregler_verlust(12.0, 5.0, 0.5), 3.5)
        self.nah(ef.konstantstrom(1.25, 12.5), 0.1)
        self.nah(ef.tiefsetzsteller_ua(12.0, 0.4), 4.8)
        self.nah(ef.hochsetzsteller_ua(5.0, 0.5), 10.0)
        self.nah(ef.inverswandler_ua(12.0, 0.5), 12.0)
        self.nah(ef.tiefsetzsteller_l(12.0, 5.0, 100e3, 0.3), 97.222e-6, rel=1e-4)
        self.nah(ef.tiefsetzsteller_c(0.3, 100e3, 0.01), 37.5e-6)
        # Boost-L mit D = 1 - Ue/Ua identisch zur Form Ue D / (f dI)
        self.nah(ef.hochsetzsteller_l(5.0, 12.0, 100e3, 0.3), 5.0 * (7 / 12) / (100e3 * 0.3))
        self.nah(ef.kuehlkoerper_rth(398.15, 313.15, 10.0, 1.5, 0.5), 6.5)
        self.nah(ef.sperrschichttemperatur(313.15, 10.0, 8.5), 398.15)
        self.nah(ef.lastausregelung(5.05, 5.0), 0.01)

    def test_leistungselektronik(self):
        self.nah(ef.steuerwinkel(50.0, 5e-3), PI / 2)
        self.nah(ef.phasenanschnitt_effektivwert(230.0, 0.0), 230.0)
        self.nah(ef.phasenanschnitt_effektivwert(230.0, PI / 2), 230.0 / np.sqrt(2))
        self.nah(ef.phasenanschnitt_effektivwert(230.0, PI), 0.0, abs_=1e-12)
        # gegen numerische Integration des angeschnittenen Sinus
        a = 1.0
        wt = np.linspace(0.0, PI, 200001)
        u = np.where(wt >= a, np.sin(wt), 0.0)
        u_eff_num = np.sqrt(np.trapezoid(u ** 2, wt) / PI)
        self.nah(ef.phasenanschnitt_effektivwert(1 / np.sqrt(2), a), u_eff_num, rel=1e-5)    # Gitter-Sprungstelle
        self.nah(ef.gesteuert_b2(325.0, 0.0), 2 * 325.0 / PI)                          # = Gleichrichtwert
        self.nah(ef.gesteuert_m1(325.0, 0.0), 325.0 / PI)
        self.nah(ef.schwingungspaket_leistung(2000.0, 3.0, 10.0), 600.0)
        self.nah(ef.schwingungspaket_effektivwert(230.0, 1.0, 4.0), 115.0)
        self.nah(ef.thyristor_verlustleistung(0.9, 0.01, 10.0, 15.7), 11.46, rel=1e-3)
        self.nah(ef.zuendwiderstand(5.0, 1.5, 20e-3), 175.0)


# ---------------------------------------------------------------------------
# Quellen, Messtechnik, Felder
# ---------------------------------------------------------------------------
class TestQuellenMessen(Basis):
    def test_quellen(self):
        self.nah(ef.quellen_gruppenschaltung(1.5, 0.2, 4.0, 2.0), (6.0, 0.4))
        self.nah(ef.strom_gruppenschaltung(1.5, 0.2, 4.0, 2.0, 10.0), 6.0 / 10.4)
        self.nah(ef.innenwiderstand(12.6, 12.0, 10.0), 0.06)
        self.nah(ef.kurzschlussstrom(12.0, 0.5), 24.0)
        self.nah(ef.innenwiderstand_lk(12.0, 24.0), 0.5)
        self.nah(ef.nernst(1.10, 298.15, 2.0, 1.0), 1.10)
        self.nah(ef.nernst(1.10, 298.15, 2.0, 10.0), 1.10 - 0.02958, rel=1e-4)          # 0,059/z lg Q
        self.nah(ef.elektrolyse_masse(96485.33212, 63.546e-3, 2.0), 31.773e-3, rel=1e-4)   # 1 F -> 1/2 mol Cu
        self.nah(ef.fuellfaktor(4.0, 0.65, 7.5), 0.8205, rel=1e-4)
        self.nah(ef.solar_wirkungsgrad(320.0, 1000.0, 1.6), 0.2)
        self.nah(ef.solar_leistung_temperatur(320.0, -0.0035, 333.15, 298.15), 280.8)
        self.nah(ef.joule_zu_kwh(ef.solar_jahresertrag(10e3, 950 * 3600.0, 0.85)), 8075.0)
        self.nah(ef.ladezeit_akku(ef.ah_zu_coulomb(2.0), 0.2, 1.4), 14 * 3600.0)

    def test_messtechnik(self):
        self.nah(ef.vorwiderstand_messbereich(10.0, 0.1, 1e-3), 9900.0)
        self.nah(ef.shuntwiderstand(1.0, 1e-3, 100.0), 0.1001, rel=1e-4)
        # Messschaltungen: aus "gemessenen" Werten wahres R_x zurueckgewinnen
        Rx, RiV, RiA = 1000.0, 1e5, 1.0
        U = 10.0
        I_spr = U / Rx + U / RiV
        self.nah(ef.widerstand_spannungsrichtig(U, I_spr, RiV), Rx)
        I_str = U / (Rx + RiA)
        self.nah(ef.widerstand_stromrichtig(U, I_str, RiA), Rx)
        self.nah(ef.zaehler_leistung(30.0, 75.0 / 3.6e6, 3600.0), 400.0)               # 75 U/kWh, 30 U/h
        self.nah(ef.klassenfehler(1.5, 10.0), 0.15)
        self.nah(ef.klassenfehler_relativ(1.5, 10.0, 2.0), 0.075)
        self.nah(ef.gesamtfehler(0.03, 0.04), 0.05)
        x = [9.8, 10.1, 10.0, 10.3, 9.8]
        self.nah(ef.mittelwert(x), 10.0)
        self.nah(ef.standardabweichung(x), np.sqrt(0.18 / 4))
        self.nah(ef.standardfehler(x), np.sqrt(0.18 / 4) / np.sqrt(5))
        self.nah(ef.mittlerer_absoluter_fehler(x, 10.0), 0.16)
        self.nah(ef.fehlerfortpflanzung(ef.widerstand_ohm, [10.0, 0.5], [0.1, 0.005]), 20 * np.sqrt(2) * 0.01,
                 rel=1e-6)
        self.nah(ef.energiekosten(3500.0, 0.35), 1225.0)

    def test_gleichstromnetz(self):
        self.nah(ef.spannungsteiler_belastet(12.0, 1e3, 1e3, 1e30), ef.spannungsteiler(12.0, 1e3, 1e3))
        self.nah(ef.spannungsteiler_belastet(12.0, 1e3, 1e3, 1e3), 4.0)
        self.nah(ef.brueckenspannung(10.0, 1e3, 1e3, 2e3, 2e3), 0.0, abs_=1e-15)        # abgeglichen
        self.nah(ef.brueckenspannung(10.0, 1e3, 1.1e3, 1e3, 1e3), 0.238095, rel=1e-5)
        # Brueckenstrom ueber Ersatzquelle = Knotenpotentialverfahren
        R1, R2, R3, R4, RB, U = 1e3, 1.1e3, 1e3, 1e3, 500.0, 10.0
        G = np.array([[1 / R1 + 1 / R2 + 1 / RB, -1 / RB], [-1 / RB, 1 / R3 + 1 / R4 + 1 / RB]])
        phi = ef.knotenpotentiale(G, [U / R1, U / R3])
        self.nah(ef.bruecke_strom(U, R1, R2, R3, R4, RB), (phi[0] - phi[1]) / RB)
        self.nah(ef.bruecke_gesamtwiderstand(1e3, 1e3, 1e3, 1e3), 1e3)
        self.nah(ef.leitwert_leiter(5.8e7, 1.5e-6, 100.0), 1 / ef.widerstand_leiter(1 / 5.8e7, 100.0, 1.5e-6))
        self.nah(ef.arbeit_i2rt(2.0, 10.0, 60.0), 2400.0)

    def test_felder_wellen(self):
        self.nah(ef.b_feld_leiter(10.0, 0.05, 1.0), 40e-6, rel=1e-6)
        self.nah(ef.b_feld_leiter(10.0, 0.05, 1.0), ef.flussdichte(1.0, ef.h_feld_leiter(10.0, 0.05)))
        self.nah(ef.fluss_winkel(1.0, 1e-2, PI / 3), 5e-3)
        self.nah(ef.feldstaerke_flaechenladung(ef.flaechenladungsdichte(1e-8, 1e-2), 1.0), 112940.6, rel=1e-5)
        # Plattenkondensator: sigma = C U / A  ->  E = U / d
        sigma = ef.flaechenladungsdichte(ef.ladung_kondensator(ef.kapazitaet_platten(1.0, 1e-2, 1e-3), 100.0), 1e-2)
        self.nah(ef.feldstaerke_flaechenladung(sigma, 1.0), ef.feldstaerke_platten(100.0, 1e-3))
        self.nah(ef.verschiebungsdichte_punktladung(1e-9, 0.1),
                 ef.verschiebungsdichte(1.0, ef.feldstaerke_punktladung(1e-9, 0.1)))
        # Luftspalt dominiert: Phi ~ N I mu0 A / l_L
        self.nah(ef.magn_fluss_luftspalt(500.0, 1.0, 0.2, 1e12, 1e-3, 4e-4), 500 * ef.MU0 * 4e-4 / 1e-3)
        self.nah(ef.wellenlaenge_photon(ef.photonenenergie(5e14)), ef.C0 / 5e14)
        self.nah(ef.joule_zu_ev(ef.photonenenergie(ef.C0 / 1240e-9)), 1.0, rel=1e-3)   # 1240 nm <-> 1 eV
        self.nah(ef.snellius(1.0, PI / 6, 1.5), np.arcsin(1 / 3))
        self.assertTrue(np.isnan(ef.snellius(1.5, PI / 3, 1.0)))                          # Totalreflexion
        self.nah(ef.doppler(1000.0, 343.0, 0.0, 30.0), 1095.85, rel=1e-5)
        self.nah(ef.wellenzahl(2 * PI), 1.0)


# ---------------------------------------------------------------------------
# Mechanik, Waerme, Geometrie, Datenuebertragung
# ---------------------------------------------------------------------------
class TestPhysikGrundlagen(Basis):
    def test_mechanik(self):
        self.nah(ef.gewichtskraft(1.0), 9.80665)
        self.nah(ef.fallgeschwindigkeit(20.0), 19.8057, rel=1e-5)
        self.nah(ef.fallzeit(20.0), 2.0196, rel=1e-4)
        self.nah(ef.geschwindigkeit_beschleunigt(0.0, ef.G_ERDE, ef.fallzeit(20.0)), ef.fallgeschwindigkeit(20.0))
        self.nah(ef.weg_beschleunigt(0.0, 0.0, ef.G_ERDE, ef.fallzeit(20.0)), 20.0)
        self.nah(ef.geschwindigkeit_weg(5.0, 2.0, 24.0), 11.0)
        self.nah(ef.gravitationskraft(5.972e24, 1.0, 6.371e6), 9.82, rel=2e-3)          # g auf der Erde
        self.nah(ef.kraftzerlegung(100.0, PI / 6), (86.6025, 50.0), rel=1e-5)
        self.nah(ef.kraft_resultierende(30.0, 40.0), 50.0)
        self.nah(ef.kraefte_addieren((1, 2), (3, -2), (-4, 0)), (0.0, 0.0))
        self.nah(ef.schwerpunkt([1.0, 3.0], [0.0, 4.0]), 3.0)
        self.nah(ef.hebel_kraft(100.0, 0.2, 1.0), 20.0)
        self.nah(ef.drehmoment_kraft(100.0, 0.3, PI / 2), 30.0)
        self.nah(ef.hangabtriebskraft(10.0, PI / 6), 49.033, rel=1e-4)
        self.nah(ef.beschleunigung_schiefe_ebene(PI / 6, 0.0), ef.G_ERDE / 2)
        self.nah(ef.beschleunigung_schiefe_ebene(np.arctan(0.3), 0.3), 0.0, abs_=1e-14)   # Grenzwinkel
        self.nah(ef.leistung_drehmoment(50.0, 2 * PI * 24.0), ef.leistung_mechanisch(24.0, 50.0))
        self.nah(ef.rotationsenergie(0.5, 100.0), 2500.0)
        self.nah(ef.kinetische_energie(10.0, 5.0) + ef.hubarbeit(10.0, 0.0), 125.0)
        # Energieerhaltung freier Fall: m g h = 1/2 m v^2
        self.nah(ef.hubarbeit(3.0, 20.0), ef.kinetische_energie(3.0, ef.fallgeschwindigkeit(20.0)))
        self.nah(ef.gasdichte(101325.0, 287.05, 288.15), 1.225, rel=1e-3)               # Normatmosphaere
        self.nah(ef.dichte_mittel(1.0, 1e-3, 7.85, 1e-3), 4425.0)

    def test_waerme(self):
        self.nah(ef.waermemenge(1.0, 4182.0, 80.0), 334560.0)
        self.nah(ef.spez_waermekapazitaet(334560.0, 1.0, 80.0), 4182.0)
        self.nah(ef.waermemenge(1.0, 4182.0, 80.0) / (2000.0 * 1.0),
                 ef.heizzeit(1.0, 4182.0, 80.0, 1.0, 2000.0))
        self.nah(ef.erwaermung_dT(2000.0, ef.heizzeit(1.0, 4182.0, 80.0, 1.0, 2000.0), 1.0, 4182.0), 80.0)
        self.nah(ef.wirkungsgrad_carnot(300.0, 600.0), 0.5)
        self.nah(ef.waermestrom_leitung(0.04, 10.0, 20.0, 0.1), 80.0)
        self.nah(ef.waermestrom_rth(20.0, ef.waermewiderstand(0.1, 0.04, 10.0)), 80.0)
        # duennes Rohr -> ebene Wand
        self.nah(ef.waermestrom_zylinder(0.04, 1.0, 21.0, 20.0, 1.0, 1.001),
                 ef.waermestrom_leitung(0.04, 2 * PI * 1.0005, 1.0, 0.001), rel=1e-6)
        self.nah(ef.waermedurchgangskoeffizient(7.7, 0.24, 0.8, 25.0), 2.1282, rel=1e-4)
        self.nah(ef.waermestrom_strahlung(1.0, 1.0, 300.0, 0.0), 459.30, rel=1e-4)
        self.nah(ef.kuehlflaeche(10.0, 10.0, 40.0), 0.025)

    def test_geometrie(self):
        self.nah(ef.flaeche_n_eck(4.0, 2.0), 4.0)                                       # Quadrat
        self.nah(ef.flaeche_n_eck(6.0, 1.0), 2.598076, rel=1e-6)
        self.nah(ef.flaeche_n_eck(1e6, 2 * PI / 1e6), PI, rel=1e-9)                     # -> Kreis
        self.nah(ef.abstand_3d(0, 0, 0, 1, 2, 2), 3.0)
        self.nah(ef.volumen_kegel(1.0, 3.0), PI)
        self.nah(ef.volumen_kugel(1.0), 4.18879, rel=1e-5)
        self.nah(ef.ableitung(ef.volumen_kugel, 2.0), ef.oberflaeche_kugel(2.0), rel=1e-6)   # dV/dr = O
        self.nah(ef.oberflaeche_zylinder(1.0, 2.0), ef.mantel_zylinder(1.0, 2.0) + 2 * ef.flaeche_kreis(1.0))
        self.nah(ef.oberflaeche_kegel(3.0, 5.0), ef.mantel_kegel(3.0, 5.0) + ef.flaeche_kreis(3.0))
        self.nah(ef.oberflaeche_quader(1, 1, 1), 6.0)
        self.nah(ef.flaeche_trapez(4.0, 2.0, 3.0), 9.0)

    def test_datenuebertragung(self):
        self.nah(ef.shannon_kapazitaet(3100.0, ef.von_db_leistung(30.0)), 30898.0, rel=1e-4)
        self.nah(ef.nyquist_kapazitaet(3100.0, 4.0), 12400.0)
        self.nah(ef.baudrate(9600.0, 16.0), 2400.0)
        self.nah(ef.tdm_bitrate(32.0, 8.0, 125e-6), 2.048e6)                           # PCM30
        self.nah(ef.tdm_zeitschlitz(125e-6, 32.0), 3.90625e-6)
        self.nah(ef.p_fehlerfrei(1e-6, 1e6), np.exp(-1), rel=1e-6)
        self.nah(ef.entscheidungsgehalt(256.0), 8.0)
        self.nah(ef.redundanz_relativ(8.0, 7.0), 0.125)
        self.nah(ef.verfuegbarkeit(999.0, 1.0), 0.999)
        self.nah(ef.zuverlaessigkeit(1e-6, 1e6), np.exp(-1))
        self.nah(ef.carson_bandbreite(75e3, 15e3), 180e3)                               # UKW-Rundfunk
        self.nah(ef.numerische_apertur(1.48, 1.46), 0.24249, rel=1e-4)
        self.nah(ef.daempfung_lwl(1e-3, 0.5e-3, 10e3) * 1e3, 0.30103, rel=1e-5)          # dB/km
        self.nah(ef.dispersion_impulsverbreiterung(17e-6, 10e3, 1e-9), 170e-12)
        self.nah(ef.brechzahl(2e8), 1.49896, rel=1e-5)


# ---------------------------------------------------------------------------
# Digitaltechnik und Mathematik
# ---------------------------------------------------------------------------
class TestDigitalMathe(Basis):
    def test_gatter_und_addierer(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 1, 0, 1])
        self.assertEqual(list(ef.UND(a, b)), [0, 0, 0, 1])
        self.assertEqual(list(ef.ODER(a, b)), [0, 1, 1, 1])
        self.assertEqual(list(ef.NAND(a, b)), [1, 1, 1, 0])
        self.assertEqual(list(ef.NOR(a, b)), [1, 0, 0, 0])
        self.assertEqual(list(ef.XOR(a, b)), [0, 1, 1, 0])
        self.assertEqual(list(ef.XNOR(a, b)), [1, 0, 0, 1])
        self.assertEqual(list(ef.NICHT(a)), [1, 1, 0, 0])
        for x in range(2):
            for y in range(2):
                for c in range(2):
                    s, co = ef.volladdierer(x, y, c)
                    self.assertEqual(2 * co + s, x + y + c)
                s, co = ef.halbaddierer(x, y)
                self.assertEqual(2 * co + s, x + y)

    def test_schaltalgebra(self):
        eq = ef.aequivalent
        self.assertTrue(eq(lambda a, b: ef.NAND(a, b), lambda a, b: ef.ODER(ef.NICHT(a), ef.NICHT(b)), 2))  # De Morgan
        self.assertTrue(eq(lambda a, b: ef.NOR(a, b), lambda a, b: ef.UND(ef.NICHT(a), ef.NICHT(b)), 2))
        self.assertTrue(eq(lambda a, b, c: a & (b | c), lambda a, b, c: (a & b) | (a & c), 3))           # Distributiv
        self.assertTrue(eq(lambda a, b, c: a | (b & c), lambda a, b, c: (a | b) & (a | c), 3))
        self.assertTrue(eq(lambda a, b: a | (a & b), lambda a, b: a, 2))                                  # Absorption
        self.assertFalse(eq(lambda a, b: a | b, lambda a, b: a & b, 2))
        xor = lambda a, b: a ^ b                                                                          # noqa: E731
        self.assertEqual(ef.minterme(xor, 2), [1, 2])
        self.assertEqual(ef.maxterme(xor, 2), [0, 3])
        self.assertEqual(ef.dnf(xor, "AB"), "/A B + A /B")
        self.assertEqual(ef.knf(xor, "AB"), "(A + B) (/A + /B)")
        self.assertEqual(len(ef.wahrheitstabelle(lambda a, b, c: a ^ b ^ c, 3)), 8)

    def test_zahlensysteme(self):
        self.assertEqual(ef.dezimal_zu_dual(13), "1101")
        self.assertEqual(ef.dezimal_zu_dual(0.625, 3), "0.101")
        self.assertEqual(ef.dual_zu_dezimal("1101"), 13)
        self.assertEqual(ef.dual_zu_dezimal("0.101"), 0.625)
        self.assertEqual(ef.dezimal_zu_hex(419), "1A3")
        self.assertEqual(ef.hex_zu_dezimal("1A3"), 419)
        self.assertEqual(ef.hex_zu_dual("1A3"), "0001 1010 0011")
        self.assertEqual(ef.dual_zu_hex("110100011"), "1A3")
        self.assertEqual(ef.dezimal_zu_basis(255, 8), "377")
        for n in (0, 1, 7, 100, 4095):
            self.assertEqual(ef.basis_zu_dezimal(ef.dezimal_zu_basis(n, 16), 16), n)

    def test_dualarithmetik(self):
        self.assertEqual(ef.dual_addieren("1101", "1011"), "11000")
        self.assertEqual(ef.zweierkomplement(-5, 8), "11111011")
        self.assertEqual(ef.zweierkomplement_wert("11111011"), -5)
        self.assertEqual(ef.zweierkomplement_wert("01111111"), 127)
        self.assertEqual(ef.dual_subtrahieren("1101", "101"), "01000")                  # 13 - 5 = 8
        self.assertEqual(ef.zweierkomplement_wert(ef.dual_subtrahieren("101", "1101")), -8)
        self.assertEqual(ef.dual_multiplizieren("1101", "101"), "1000001")              # 13 * 5 = 65
        self.assertEqual(ef.dual_dividieren("1101", "11"), ("100", "1"))                # 13 / 3 = 4 Rest 1
        with self.assertRaises(ValueError):
            ef.zweierkomplement(128, 8)

    def test_grundrechnen(self):
        self.assertEqual(ef.runden(2.5), 3.0)                                           # nicht 2 (round)
        self.assertEqual(ef.runden(2.675, 2), 2.68)
        self.assertEqual(ef.runden(-2.5), -3.0)
        self.nah(ef.runden_auf_raster(3.37, 0.05), 3.35)
        self.nah(ef.prozentwert(200.0, 15.0), 30.0)
        self.nah(ef.grundwert(30.0, 15.0), 200.0)
        self.nah(ef.prozentsatz(30.0, 200.0), 15.0)
        self.nah(ef.prozentuale_aenderung(200.0, 230.0), 15.0)
        self.nah(ef.dreisatz(3.0, 12.0, 5.0), 20.0)
        self.nah(ef.dreisatz_antiproportional(4.0, 6.0, 3.0), 8.0)
        self.assertEqual(ef.bruch(1, 3) + ef.bruch(1, 6), Fraction(1, 2))
        self.assertEqual(ef.gemischte_zahl(Fraction(7, 3)), (2, Fraction(1, 3)))
        self.nah(ef.nte_wurzel(-27.0, 3), -3.0)
        self.nah(ef.nte_wurzel(16.0, 4), 2.0)
        self.nah(ef.logarithmus(1024.0, 2.0), 10.0)
        self.nah(ef.logarithmus(8.0 * 4.0, 2.0), ef.logarithmus(8.0, 2.0) + ef.logarithmus(4.0, 2.0))
        self.nah(ef.halbwertszeit(np.log(2.0)), 1.0)
        self.nah(ef.zinseszins(1000.0, 0.05, 2.0), 1102.5)
        for a, b in ((3.0, 4.0), (1.5, -2.0)):                                          # Binomische Formeln
            self.nah((a + b) ** 2, a ** 2 + 2 * a * b + b ** 2)
            self.nah((a + b) * (a - b), a ** 2 - b ** 2)

    def test_trigonometrie_analysis_vektoren(self):
        self.nah(ef.pythagoras_hypotenuse(3.0, 4.0), 5.0)
        self.nah(ef.pythagoras_kathete(5.0, 3.0), 4.0)
        w = ef.winkelfunktionen(3.0, 4.0)
        self.nah([w["sin"], w["cos"], w["tan"], w["cot"]], [0.6, 0.8, 0.75, 4 / 3])
        self.nah(w["sin"] ** 2 + w["cos"] ** 2, 1.0)
        self.nah(ef.kosinussatz(3.0, 4.0, PI / 2), 5.0)
        self.nah(ef.kosinussatz_winkel(3.0, 4.0, 5.0), PI / 2)
        self.nah(ef.sinussatz_seite(1.0, PI / 6, PI / 2), 2.0)
        self.nah(ef.ableitung(lambda x: x ** 3, 2.0), 12.0, rel=1e-6)
        self.nah(ef.ableitung(np.sin, 1.0), np.cos(1.0), rel=1e-6)
        self.nah(ef.ableitung2(np.exp, 1.0), np.e, rel=1e-5)
        self.nah(ef.integral(np.sin, 0.0, PI), 2.0, rel=1e-9)
        self.nah(np.sqrt(ef.mittelwert_integral(lambda t: np.sin(t) ** 2, 0.0, 2 * PI)), 1 / np.sqrt(2))  # Eff.wert
        self.nah(ef.flaeche_zwischen(lambda x: x, lambda x: x ** 2, 0.0, 1.0), 1 / 6, rel=1e-9)
        self.nah(ef.vektor_betrag([1, 2, 2]), 3.0)
        self.nah(ef.einheitsvektor([3, 0, 4]), [0.6, 0.0, 0.8])
        self.nah(ef.skalarprodukt([1, 2, 3], [4, 5, 6]), 32.0)
        self.nah(ef.kreuzprodukt([1, 0, 0], [0, 1, 0]), [0, 0, 1])
        self.nah(ef.winkel_vektoren([1, 0], [1, 1]), PI / 4)

    def test_umstellen(self):
        C = ef.umstellen(ef.resonanzfrequenz, "C", 1000.0, L=10e-3)
        self.nah(C, 2.533e-6, rel=1e-3)
        self.nah(ef.resonanzfrequenz(10e-3, C), 1000.0, rel=1e-9)
        R = ef.umstellen(ef.leistung_u2r, "R", 100.0, U=230.0)
        self.nah(R, 529.0, rel=1e-9)
        T = ef.umstellen(ef.ntc_widerstand, "T", 3588.1, R_N=10e3, B=3950.0, T_N=298.15)
        self.nah(T, 323.15, rel=1e-5)
        with self.assertRaises(ValueError):
            ef.umstellen(ef.leistung_u2r, "R", -1.0, U=230.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
