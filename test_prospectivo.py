"""Comprobaciones reproducibles de capital, excedente y riesgo."""
import unittest
import numpy as np
import pandas as pd
from prospectivo import (ParametrosFX, estadisticas_riesgo, optimizar_asignacion_paises, optimizar_markowitz,
                         rendimientos_esperados, simular_fx, simular_portafolio,
                         seleccionar_activos_por_correlacion, sortear_tasa,
                         tabla_resumen_componentes)


class TestProspectivo(unittest.TestCase):
    def setUp(self):
        self.columnas = ['US1','US2','MX1','MX2']
        self.r = pd.DataFrame(np.zeros((30,4)), columns=self.columnas)
        self.pus = np.array([.5,.5])
        self.pmx = np.array([.5,.5])
        self.fx = np.full((3,253),20.)
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
        np.testing.assert_allclose(x['capital'][:, 21], y['capital'][:, 21])
        np.testing.assert_allclose(x['hijo'][:, :21], 0.)
        self.assertTrue(np.all(x['capital'][:, -1] < y['capital'][:, -1]))
        self.assertTrue(np.all(x['hijo'][:, -1] > 0))
        self.assertTrue(np.all(x['transferencias'] > 0))

    def test_desglose_concilia_en_cada_dia_y_escenario(self):
        r = self.r.copy(); r.iloc[:, :2] = .03; r.iloc[:, 2:] = -.01
        fx = self.fx.copy(); fx[:, 1:] = np.linspace(19., 23., 252)
        x = self.simular(retornos=r, fx=fx,
                         tasa_us=np.full(3, .04), tasa_mx=np.full(3, .08))
        np.testing.assert_allclose(x['eua'] + x['mexico'] + x['hijo'], x['capital'])
        np.testing.assert_allclose(x['eua_con_hija'] + x['mexico_con_hija'], x['capital'])
        np.testing.assert_allclose(x['hijo_eua'] + x['hijo_mexico'], x['hijo'])
        resumen = rendimientos_esperados(x, 1000., .6)
        self.assertAlmostEqual(resumen['global_final'], x['capital'][:, -1].mean())
        self.assertAlmostEqual(resumen['global_retorno'],
                               .6*resumen['eua_retorno'] + .4*resumen['mexico_retorno'])
        self.assertAlmostEqual(resumen['hija_retorno_aportaciones'],
                               resumen['hija_final']/resumen['hija_aportaciones']-1)

    def test_hija_sin_aportaciones_no_tiene_rendimiento(self):
        resumen = rendimientos_esperados(self.simular(), 1000., .6)
        self.assertIsNone(resumen['hija_retorno_aportaciones'])
        self.assertAlmostEqual(resumen['global_retorno'], 0.)

    def test_tasa_segura_anual(self):
        x = self.simular(fraccion_riesgo=0., tasa_us=np.full(3,.21),
                         tasa_mx=np.full(3,.10))
        np.testing.assert_allclose(x['capital'][:, -1], 600*1.21+400*1.10)
        np.testing.assert_allclose(x['seguro_eua_usd'][:, 0], 600/20)
        np.testing.assert_allclose(x['seguro_eua_usd'][:, -1], (600/20)*1.21)
        np.testing.assert_allclose(x['seguro_mexico_mxn'][:, 0], 400)
        np.testing.assert_allclose(x['seguro_mexico_mxn'][:, -1], 400*1.10)
        np.testing.assert_allclose(x['seguro_mexico_mxn'][:, 21], 400*1.10**(21/252))

    def test_tasa_usd_se_separa_del_efecto_cambiario(self):
        fx = self.fx.copy(); fx[:, -1] = 22.
        x = self.simular(fx=fx, fraccion_riesgo=0.,
                         tasa_us=np.full(3,.12), tasa_mx=self.ceros)
        inicial_usd = 600/20
        np.testing.assert_allclose(x['seguro_eua_usd'][:, -1], inicial_usd*1.12)
        np.testing.assert_allclose(x['capital'][:, -1], inicial_usd*1.12*22+400)
        self.assertAlmostEqual((x['seguro_eua_usd'][:, -1].mean()-inicial_usd)*20, 72.)

    def test_riesgo_y_tasa(self):
        r = estadisticas_riesgo(np.array([80.,90.,110.,120.]), 100.)
        self.assertAlmostEqual(r['prob_perdida'], .5)
        self.assertGreaterEqual(r['es95'], r['var95'])
        self.assertAlmostEqual(r['var95_pct'], r['var95']/100.)
        np.testing.assert_allclose(sortear_tasa(3,.04,.04,.04,1), .04)

    def test_markowitz_global_considera_conversion_del_dolar(self):
        fechas = pd.bdate_range('2024-01-02', periods=301)
        retornos = pd.DataFrame(0., index=fechas[1:], columns=self.columnas)
        cambios = .001 + .0002*np.sin(np.arange(300))
        fx = pd.Series(20*np.cumprod(np.r_[1., 1+cambios]), index=fechas)
        asignacion, frontera = optimizar_asignacion_paises(
            retornos, fx, self.pus, self.pmx, self.columnas[:2],
            self.columnas[2:], 1., 0., .05)
        self.assertGreater(asignacion['eua'], .99)
        self.assertAlmostEqual(asignacion['eua'] + asignacion['mexico'], 1.)
        self.assertEqual(len(frontera), 1001)

    def test_bootstrap_diario_produce_rutas_diferentes(self):
        r = self.r.copy()
        r.iloc[::2, :2] = .02
        r.iloc[1::2, :2] = -.01
        x = self.simular(retornos=r, umbral_excedente=None)
        self.assertFalse(np.allclose(x['capital'][0], x['capital'][1]))
        self.assertGreater(np.std(np.diff(x['capital'][0])), 0.)

    def test_seleccion_automatica_cumple_seis_correlaciones_negativas(self):
        fechas = pd.bdate_range('2024-01-02', periods=301)
        factor = .003 * np.sin(np.arange(300) * .17)
        r = np.column_stack([factor + .0001*i for i in range(3)]
                            + [-factor + .0001*i for i in range(2)])
        precios = pd.DataFrame(100*np.cumprod(np.vstack([np.ones(5), 1+r]), axis=0),
                               index=fechas, columns=list('ABCDE'))
        tres, dos, corr, info = seleccionar_activos_por_correlacion(precios, fechas[-1])
        self.assertTrue(info['regla_estricta'])
        self.assertEqual(info['pares_negativos'], 6)
        self.assertTrue((corr.loc[tres, dos] < 0).to_numpy().all())

    def test_tabla_componentes_concilia_traspasos(self):
        r = self.r.copy(); r.iloc[:] = .005
        fx = self.fx.copy(); fx[:, 1:] = np.linspace(20., 21., 252)
        x = self.simular(retornos=r, fx=fx)
        tabla = tabla_resumen_componentes(x, fx, 1000., .6, .7)
        detalle = tabla.iloc[1:]
        self.assertAlmostEqual(detalle['Inversión inicial (MXN)'].sum(), 1000.)
        self.assertAlmostEqual(detalle['Valor final esperado (MXN)'].sum(),
                               tabla.iloc[0]['Valor final esperado (MXN)'])
        self.assertAlmostEqual(detalle['Ganancia esperada (MXN)'].sum(),
                               tabla.iloc[0]['Ganancia esperada (MXN)'])
        np.testing.assert_allclose(x['transferencias_us_mxn'] +
                                   x['transferencias_mx_mxn'], x['transferencias'])

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
