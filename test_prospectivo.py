"""Comprobaciones reproducibles de capital, excedente y riesgo."""
import unittest
import numpy as np
import pandas as pd
from prospectivo import (ParametrosFX, estadisticas_riesgo, optimizar_markowitz,
                         simular_fx, simular_portafolio, sortear_tasa)


class TestProspectivo(unittest.TestCase):
    def setUp(self):
        self.columnas = ['US1','US2','MX1','MX2']
        self.r = pd.DataFrame(np.zeros((30,4)), columns=self.columnas)
        self.pus = np.array([.5,.5])
        self.pmx = np.array([.5,.5])
        self.fx = np.full((3,13),20.)
        self.ceros = np.zeros(3)

    def simular(self, **kwargs):
        args = dict(retornos=self.r, pesos_us=self.pus, pesos_mx=self.pmx,
                    tickers_us=self.columnas[:2], tickers_mx=self.columnas[2:],
                    fx=self.fx, tasa_us=self.ceros, tasa_mx=self.ceros,
                    capital_mxn=1000., fraccion_us=.6, fraccion_riesgo=.7,
                    semilla=1, umbral_excedente=.015)
        args.update(kwargs)
        return simular_portafolio(**args)

    def test_capital_constante_sin_movimientos(self):
        x = self.simular()
        np.testing.assert_allclose(x['capital'], 1000.)
        np.testing.assert_allclose(x['hijo'], 0.)

    def test_divisa_solo_afecta_parte_us(self):
        fx = self.fx.copy(); fx[:, -1] = 22.
        x = self.simular(fx=fx)
        np.testing.assert_allclose(x['capital'][:, -1], 1060.)
        np.testing.assert_allclose(x['capital_sin_fx'][:, -1], 1000.)

    def test_excedente_se_transfiere_sin_crear_capital(self):
        r = self.r.copy(); r.iloc[:] = .03
        x = self.simular(retornos=r)
        y = self.simular(retornos=r, umbral_excedente=None)
        # La transferencia conserva el patrimonio del primer mes, pero
        # cambia la composición y por ello reduce el crecimiento posterior.
        np.testing.assert_allclose(x['capital'][:, 1], y['capital'][:, 1])
        self.assertTrue(np.all(x['capital'][:, -1] < y['capital'][:, -1]))
        self.assertTrue(np.all(x['hijo'][:, -1] > 0))
        self.assertTrue(np.all(x['transferencias'] > 0))

    def test_tasa_segura_anual(self):
        x = self.simular(fraccion_riesgo=0., tasa_us=np.full(3,.21),
                         tasa_mx=np.full(3,.10))
        np.testing.assert_allclose(x['capital'][:, -1], 600*1.21+400*1.10)

    def test_riesgo_y_tasa(self):
        r = estadisticas_riesgo(np.array([80.,90.,110.,120.]), 100.)
        self.assertAlmostEqual(r['prob_perdida'], .5)
        self.assertGreaterEqual(r['es95'], r['var95'])
        np.testing.assert_allclose(sortear_tasa(3,.04,.04,.04,1), .04)

    def test_markowitz_pesos_validos(self):
        rng = np.random.default_rng(5)
        datos = pd.DataFrame(rng.normal([.01,.02], [.03,.05], (30,2)), columns=['A','B'])
        tabla, elegidos, frames = optimizar_markowitz(datos, np.array([.02,.04,.03]),3,100,6)
        self.assertEqual(len(tabla),3)
        self.assertTrue(np.isclose(elegidos['sharpe'].sum(),1))
        self.assertTrue(np.isclose(elegidos['min_var'].sum(),1))
        self.assertEqual(frames[-1]['corrida'],3)

    def test_fx_dia_cero_y_positividad(self):
        p = ParametrosFX(0.,0.,0.,0.,0.,.01,.1,.8,6.,0.,0.,0.,0.,.1,'2026-01-01')
        fx = simular_fx(20.,p,5,12,7)
        np.testing.assert_allclose(fx[:,0],20.)
        self.assertTrue(np.all(fx > 0))


if __name__ == '__main__':
    unittest.main()
