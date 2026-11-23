"""Motor numérico del portafolio internacional prospectivo.

Las series y tasas se proporcionan desde la interfaz; este módulo no descarga datos.
El tipo de cambio es independiente del bootstrap de rendimientos bursátiles.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParametrosFX:
    media: float                 # media incondicional de retornos log diarios, en porcentaje
    ar1: float
    ar2: float
    ma1: float
    ma2: float
    omega: float                 # varianza en unidades de porcentaje al cuadrado
    alpha: float
    beta: float
    grados_libertad: float
    r1: float                    # último retorno, en porcentaje
    r2: float
    e1: float                    # último residuo, en porcentaje
    e2: float
    h: float                     # última varianza condicional
    fecha_ajuste: str


def preparar_rendimientos(precios: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """Usa sólo meses completos terminados antes del mes de valoración."""
    x = precios.copy().sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x = x.loc[x.index < pd.Timestamp(fecha_corte).replace(day=1)]
    mensual = x.resample('ME').last().dropna()
    r = mensual.pct_change(fill_method=None).dropna()
    if len(r) < 24:
        raise ValueError('Se necesitan al menos 24 rendimientos mensuales completos por activo.')
    if not np.isfinite(r.to_numpy()).all() or (r.to_numpy() <= -1).any():
        raise ValueError('El histórico contiene rendimientos inválidos.')
    return r


def preparar_rendimientos_diarios(precios: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """Rendimientos diarios conjuntos disponibles hasta la fecha de valoración."""
    x = precios.copy().sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x = x.loc[x.index <= pd.Timestamp(fecha_corte)].dropna()
    r = x.pct_change(fill_method=None).dropna()
    if len(r) < 252 or not np.isfinite(r.to_numpy()).all() or (r.to_numpy() <= -1).any():
        raise ValueError('Se necesitan al menos 252 rendimientos diarios válidos de todos los activos.')
    return r


def optimizar_asignacion_paises(
    retornos_diarios: pd.DataFrame, fx_historico: pd.Series,
    pesos_us: np.ndarray, pesos_mx: np.ndarray,
    tickers_us: list[str], tickers_mx: list[str],
    fraccion_riesgo: float, rf_us_anual: float, rf_mx_anual: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Markowitz de las dos carteras completas, ambas expresadas en MXN.

    Para estimar media y covarianza se usan retornos diarios históricos
    de cada cartera, con 70/30 (o el reparto elegido) y la tasa de referencia.
    La cartera de EUA se convierte diariamente por USD/MXN. La cuadrícula
    de pesos sólo define la compra inicial prospectiva; no impone rebalanceos.
    """
    if not 0 <= fraccion_riesgo <= 1 or min(rf_us_anual, rf_mx_anual) <= -1:
        raise ValueError('Revisa las tasas y el reparto entre activos y tasa segura.')
    if not (np.isclose(pesos_us.sum(), 1) and np.isclose(pesos_mx.sum(), 1)):
        raise ValueError('Los pesos de los activos por país deben sumar 100%.')
    fx = pd.to_numeric(fx_historico, errors='coerce').sort_index()
    fx.index = pd.to_datetime(fx.index).tz_localize(None)
    cambio = fx.pct_change(fill_method=None).rename('Cambio USD/MXN')
    conjunto = retornos_diarios.join(cambio, how='inner').dropna()
    if len(conjunto) < 252:
        raise ValueError('La optimización EUA/México requiere al menos 252 sesiones coincidentes de activos y USD/MXN.')
    r_us = conjunto[tickers_us].to_numpy(float) @ pesos_us
    r_mx = conjunto[tickers_mx].to_numpy(float) @ pesos_mx
    rf_us_dia = (1 + rf_us_anual) ** (1 / 252) - 1
    rf_mx_dia = (1 + rf_mx_anual) ** (1 / 252) - 1
    us_local = fraccion_riesgo * r_us + (1 - fraccion_riesgo) * rf_us_dia
    mx_local = fraccion_riesgo * r_mx + (1 - fraccion_riesgo) * rf_mx_dia
    r = np.column_stack(((1 + us_local) * (1 + conjunto['Cambio USD/MXN'].to_numpy(float)) - 1,
                         mx_local))
    media = r.mean(axis=0)
    cov = np.cov(r, rowvar=False)
    pesos_eua = np.linspace(0, 1, 1001)
    pesos = np.column_stack((pesos_eua, 1 - pesos_eua))
    mu = pesos @ media
    vol = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', pesos, cov, pesos), 0))
    sharpe = np.divide(mu - rf_mx_dia, vol, out=np.full(len(mu), -np.inf), where=vol > 0)
    i = int(np.argmax(sharpe))
    j = int(np.argmin(vol))
    if not np.isfinite(sharpe[i]):
        raise ValueError('No se obtuvo un Sharpe global definido.')
    frontera = pd.DataFrame({'Peso EUA': pesos_eua, 'Peso México': 1 - pesos_eua,
                             'Rendimiento histórico diario': mu,
                             'Volatilidad diaria': vol, 'Sharpe anualizado': sharpe * np.sqrt(252)})
    return {'eua': float(pesos_eua[i]), 'mexico': float(1 - pesos_eua[i]),
            'min_var_eua': float(pesos_eua[j]),
            'retorno_diario': float(mu[i]), 'vol_diaria': float(vol[i]),
            'sharpe_anual': float(sharpe[i] * np.sqrt(252)),
            'observaciones': int(len(conjunto))}, frontera


def optimizar_markowitz(
    retornos: pd.DataFrame,
    rf_anual_referencia: float | np.ndarray,
    corridas: int,
    portafolios_por_corrida: int,
    semilla: int,
    max_cuadros: int = 36,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict]]:
    """Busca máxima razón de Sharpe y mínima varianza en cada corrida.

    La tasa anual de referencia se transforma en tasa mensual efectiva. Los
    mismos pesos sorteados sirven para ambas búsquedas en cada corrida.
    """
    tasas = np.broadcast_to(np.asarray(rf_anual_referencia, float), (corridas,))
    if corridas < 1 or portafolios_por_corrida < 2 or np.any(tasas <= -1):
        raise ValueError('Revisa el número de corridas, portafolios y la tasa.')
    m = retornos.mean().to_numpy(float)
    cov = retornos.cov().to_numpy(float)
    n = len(m)
    rng = np.random.default_rng(semilla)
    filas, cuadros = [], []
    mejor_min_vol = np.inf
    pesos_min_global = None
    cada = max(1, corridas // max_cuadros)
    for k in range(1, corridas + 1):
        rf_mes = (1 + tasas[k - 1]) ** (1 / 12) - 1
        pesos = rng.dirichlet(np.ones(n), size=portafolios_por_corrida)
        mu = pesos @ m
        vol = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', pesos, cov, pesos), 0))
        sharpe = np.divide(mu - rf_mes, vol, out=np.full(len(mu), -np.inf), where=vol > 0)
        i_sharpe = int(np.argmax(sharpe))
        i_var = int(np.argmin(vol))
        if vol[i_var] < mejor_min_vol:
            mejor_min_vol = float(vol[i_var])
            pesos_min_global = pesos[i_var].copy()
        if not np.isfinite(sharpe[i_sharpe]):
            raise ValueError('No se obtuvo un portafolio con Sharpe definido.')
        fila = {'Corrida': k, 'Rf anual': float(tasas[k - 1]),
                'Sharpe máximo': float(sharpe[i_sharpe] * np.sqrt(12)),
                'Retorno máximo Sharpe': float(mu[i_sharpe]),
                'Volatilidad máximo Sharpe': float(vol[i_sharpe]),
                'Retorno mínima varianza': float(mu[i_var]),
                'Volatilidad mínima varianza': float(vol[i_var])}
        fila.update({f'Peso {t}': float(pesos[i_sharpe, j]) for j, t in enumerate(retornos.columns)})
        filas.append(fila)
        if k == 1 or k % cada == 0 or k == corridas:
            muestra = rng.choice(len(mu), min(1000, len(mu)), replace=False)
            cuadros.append({'corrida': k, 'vol': vol[muestra], 'mu': mu[muestra],
                            'mejor_vol': float(vol[i_sharpe]), 'mejor_mu': float(mu[i_sharpe]),
                            'min_vol': float(vol[i_var]), 'min_mu': float(mu[i_var]),
                            'rf_mes': rf_mes})
    tabla = pd.DataFrame(filas)
    ganador = tabla.loc[tabla['Sharpe máximo'].idxmax()]
    elegidos = {'sharpe': np.array([ganador[f'Peso {t}'] for t in retornos.columns]),
                'min_var': pesos_min_global}
    return tabla, elegidos, cuadros


def ajustar_fx_arma_garch(precios: pd.Series) -> ParametrosFX:
    """Reestima ARMA(2,2) + GARCH(1,1)-t hasta el último spot disponible.

    La estimación ARMA y la de volatilidad son secuenciales, de forma explícita.
    Un ajuste no convergente detiene el análisis; nunca usa el estado de 2026-10.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model

    s = pd.to_numeric(precios, errors='coerce').dropna().sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[~s.index.duplicated(keep='last')]
    if len(s) < 252 or (s <= 0).any():
        raise ValueError('El ajuste del dólar requiere al menos 252 precios diarios positivos.')
    r = (np.log(s).diff().dropna() * 100).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arma = ARIMA(r, order=(2, 0, 2), trend='c', enforce_stationarity=True,
                     enforce_invertibility=True).fit(method_kwargs={'maxiter': 300})
    if not arma.mle_retvals.get('converged', False):
        raise ValueError('El ARMA(2,2) no convergió. Amplía o revisa la serie USD/MXN.')
    e = pd.Series(np.asarray(arma.resid), index=r.index)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        garch = arch_model(e, mean='Zero', vol='GARCH', p=1, q=1,
                           dist='StudentsT', rescale=False).fit(disp='off')
    if garch.convergence_flag != 0:
        raise ValueError('El GARCH(1,1)-t no convergió. Revisa la serie USD/MXN.')
    a, g = arma.params, garch.params
    nu = float(g['nu'])
    if nu <= 2 or float(g['omega']) <= 0 or float(g['alpha[1]']) < 0 or float(g['beta[1]']) < 0:
        raise ValueError('El ajuste GARCH produjo parámetros inválidos.')
    h = float(garch.conditional_volatility.iloc[-1] ** 2)
    return ParametrosFX(float(a['const']), float(a['ar.L1']), float(a['ar.L2']),
                        float(a['ma.L1']), float(a['ma.L2']), float(g['omega']),
                        float(g['alpha[1]']), float(g['beta[1]']), nu,
                        float(r.iloc[-1]), float(r.iloc[-2]), float(e.iloc[-1]),
                        float(e.iloc[-2]), h, str(s.index[-1].date()))


def simular_fx(s0: float, p: ParametrosFX, n: int, dias: int, semilla: int) -> np.ndarray:
    """Simula USD/MXN diario. La t se estandariza a varianza unitaria."""
    if s0 <= 0 or n < 1 or dias < 1:
        raise ValueError('Spot, escenarios y días deben ser positivos.')
    rng = np.random.default_rng(semilla)
    out = np.empty((n, dias + 1), dtype=float)
    out[:, 0] = s0
    r1, r2 = np.full(n, p.r1), np.full(n, p.r2)
    e1, e2 = np.full(n, p.e1), np.full(n, p.e2)
    h = np.full(n, p.h)
    log_s = np.full(n, np.log(s0))
    for j in range(1, dias + 1):
        media = p.media + p.ar1 * (r1 - p.media) + p.ar2 * (r2 - p.media) + p.ma1 * e1 + p.ma2 * e2
        h = p.omega + p.alpha * e1**2 + p.beta * h
        z = rng.standard_t(p.grados_libertad, n) * np.sqrt((p.grados_libertad - 2) / p.grados_libertad)
        e = np.sqrt(h) * z
        r = media + e
        log_s += r / 100
        out[:, j] = np.exp(log_s)
        r2, r1, e2, e1 = r1, r, e1, e
    if not np.isfinite(out).all() or (out <= 0).any():
        raise ValueError('El modelo del dólar produjo precios inválidos; revisa el ajuste y el horizonte.')
    return out


def sortear_tasa(n: int, minimo: float, moda: float, maximo: float, semilla: int) -> np.ndarray:
    """Tasa anual efectiva, fija o triangular; los valores entran como decimales."""
    if not -1 < minimo <= moda <= maximo:
        raise ValueError('Las tasas requieren mínimo ≤ moda ≤ máximo y mínimo > −100%.')
    if minimo == maximo:
        return np.full(n, minimo)
    return np.random.default_rng(semilla).triangular(minimo, moda, maximo, n)


def simular_portafolio(
    retornos: pd.DataFrame, pesos_us: np.ndarray, pesos_mx: np.ndarray,
    tickers_us: list[str], tickers_mx: list[str], fx: np.ndarray,
    tasa_us: np.ndarray, tasa_mx: np.ndarray, capital_mxn: float,
    fraccion_us: float, fraccion_riesgo: float, semilla: int,
    umbral_excedente: float | None = 0.015,
) -> dict[str, np.ndarray]:
    """Compra inicial y conservación de posiciones durante 252 sesiones.

    El bootstrap conjunto conserva correlación histórica entre los diez activos.
    Las rutas cambiarias se generan aparte. Si hay excedente, se transfiere al
    cierre de cada bloque de 21 sesiones y devenga tasa segura desde el día siguiente.
    """
    n = len(fx)
    dias = 252
    if fx.shape[1] != dias + 1 or len(tasa_us) != n or len(tasa_mx) != n:
        raise ValueError('Se requieren 252 sesiones de FX y una tasa por escenario.')
    if capital_mxn <= 0 or not 0 <= fraccion_us <= 1 or not 0 <= fraccion_riesgo <= 1:
        raise ValueError('Capital y asignaciones iniciales inválidos.')
    if umbral_excedente is not None and not 0 <= umbral_excedente < 1:
        raise ValueError('El umbral de excedente debe estar entre 0% y 100%.')
    if set(tickers_us + tickers_mx) != set(retornos.columns):
        raise ValueError('Los activos simulados no coinciden con los optimizados.')
    if not np.isclose(pesos_us.sum(), 1) or not np.isclose(pesos_mx.sum(), 1):
        raise ValueError('Los pesos de cada cartera deben sumar 100%.')
    if (pesos_us < 0).any() or (pesos_mx < 0).any():
        raise ValueError('Los pesos no pueden ser negativos.')
    s0 = float(fx[0, 0])
    us_inicial_mxn = capital_mxn * fraccion_us
    mx_inicial_mxn = capital_mxn * (1 - fraccion_us)
    activos_us = np.broadcast_to(us_inicial_mxn * fraccion_riesgo * pesos_us / s0, (n, len(pesos_us))).copy()
    activos_mx = np.broadcast_to(mx_inicial_mxn * fraccion_riesgo * pesos_mx, (n, len(pesos_mx))).copy()
    seguro_us = np.full(n, us_inicial_mxn * (1 - fraccion_riesgo) / s0)
    seguro_mx = np.full(n, mx_inicial_mxn * (1 - fraccion_riesgo))
    hijo_us = np.zeros(n)
    hijo_mx = np.zeros(n)
    total = np.empty((n, dias + 1)); total_sin_fx = np.empty_like(total)
    cartera_us_mxn = np.empty_like(total)
    cartera_mx_mxn = np.empty_like(total)
    hijo_us_mxn = np.empty_like(total)
    hijo_mx_mxn = np.empty_like(total)
    pais_us_total = np.empty_like(total)
    pais_mx_total = np.empty_like(total)
    hijo_total = np.empty_like(total)
    seguro_us_usd = np.empty_like(total)
    seguro_mx_mxn = np.empty_like(total)
    transferencias = np.zeros((n, 12))
    def guardar(j: int) -> None:
        usd = activos_us.sum(axis=1) + seguro_us
        mxn = activos_mx.sum(axis=1) + seguro_mx
        seguro_us_usd[:, j] = seguro_us
        seguro_mx_mxn[:, j] = seguro_mx
        cartera_us_mxn[:, j] = usd * fx[:, j]
        cartera_mx_mxn[:, j] = mxn
        hijo_us_mxn[:, j] = hijo_us * fx[:, j]
        hijo_mx_mxn[:, j] = hijo_mx
        hijo_total[:, j] = hijo_us_mxn[:, j] + hijo_mx_mxn[:, j]
        pais_us_total[:, j] = cartera_us_mxn[:, j] + hijo_us_mxn[:, j]
        pais_mx_total[:, j] = cartera_mx_mxn[:, j] + hijo_mx_mxn[:, j]
        total[:, j] = cartera_us_mxn[:, j] + cartera_mx_mxn[:, j] + hijo_total[:, j]
        total_sin_fx[:, j] = (usd + hijo_us) * s0 + mxn + hijo_mx
    guardar(0)
    historico = retornos[tickers_us + tickers_mx].to_numpy(float)
    rng = np.random.default_rng(semilla)
    tasa_dia_us = (1 + tasa_us) ** (1 / dias) - 1
    tasa_dia_mx = (1 + tasa_mx) ** (1 / dias) - 1
    previo_us = activos_us.sum(axis=1)
    previo_mx = activos_mx.sum(axis=1)
    for j in range(1, dias + 1):
        # Un mismo día histórico para los diez activos conserva su dependencia.
        r = historico[rng.integers(0, len(historico), size=n)]
        activos_us *= 1 + r[:, :len(tickers_us)]
        activos_mx *= 1 + r[:, len(tickers_us):]
        seguro_us *= 1 + tasa_dia_us
        seguro_mx *= 1 + tasa_dia_mx
        hijo_us *= 1 + tasa_dia_us
        hijo_mx *= 1 + tasa_dia_mx
        if umbral_excedente is not None and j % 21 == 0:
            valor_us = activos_us.sum(axis=1)
            valor_mx = activos_mx.sum(axis=1)
            trans_us = np.maximum(valor_us - previo_us * (1 + umbral_excedente), 0)
            trans_mx = np.maximum(valor_mx - previo_mx * (1 + umbral_excedente), 0)
            activos_us *= np.divide(valor_us - trans_us, valor_us, out=np.ones(n), where=valor_us > 0)[:, None]
            activos_mx *= np.divide(valor_mx - trans_mx, valor_mx, out=np.ones(n), where=valor_mx > 0)[:, None]
            hijo_us += trans_us
            hijo_mx += trans_mx
            transferencias[:, j // 21 - 1] = trans_us * fx[:, j] + trans_mx
        if j % 21 == 0:
            previo_us = activos_us.sum(axis=1)
            previo_mx = activos_mx.sum(axis=1)
        guardar(j)
    return {'capital': total, 'capital_sin_fx': total_sin_fx,
            'eua': cartera_us_mxn, 'mexico': cartera_mx_mxn,
            'hijo': hijo_total, 'hijo_eua': hijo_us_mxn,
            'hijo_mexico': hijo_mx_mxn, 'eua_con_hija': pais_us_total,
            'mexico_con_hija': pais_mx_total,
            'seguro_eua_usd': seguro_us_usd,
            'seguro_mexico_mxn': seguro_mx_mxn,
            'transferencias': transferencias}


def estadisticas_riesgo(capital_final: np.ndarray, capital_inicial: float) -> dict[str, float]:
    x = np.asarray(capital_final, float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Los capitales finales deben ser finitos.')
    perdida = capital_inicial - x
    q95 = float(np.quantile(perdida, .95))
    return {'media': float(x.mean()), 'mediana': float(np.median(x)),
            'p05': float(np.quantile(x, .05)), 'p95': float(np.quantile(x, .95)),
            'prob_perdida': float(np.mean(x < capital_inicial)),
            'var95': max(0., q95),
            'var95_pct': max(0., q95 / capital_inicial),
            'es95': max(0., float(perdida[perdida >= q95].mean()))}


def rendimientos_esperados(simulacion: dict[str, np.ndarray], capital: float,
                           fraccion_us: float) -> dict[str, float | None]:
    """Retornos de país incluyen sus excedentes; los saldos graficados no se solapan.

    La hija inicia en cero. Su retorno es saldo final esperado / aportaciones
    esperadas - 1, sin anualizar; no se divide entre capital inicial cero.
    """
    if capital <= 0 or not 0 <= fraccion_us <= 1:
        raise ValueError('El resumen por país exige capital positivo y participación entre 0 y 1.')
    us_final = float(simulacion['eua_con_hija'][:, -1].mean())
    mx_final = float(simulacion['mexico_con_hija'][:, -1].mean())
    hija_final = float(simulacion['hijo'][:, -1].mean())
    aportaciones = simulacion['transferencias'].sum(axis=1)
    con_aportacion = aportaciones > 0
    aportaciones_esperadas = float(aportaciones.mean())
    retorno_hija = (hija_final / aportaciones_esperadas - 1
                    if aportaciones_esperadas > 0 else None)
    return {'eua_final': us_final, 'mexico_final': mx_final,
            'global_final': us_final + mx_final, 'hija_final': hija_final,
            'eua_retorno': us_final / (capital * fraccion_us) - 1 if fraccion_us else None,
            'mexico_retorno': mx_final / (capital * (1-fraccion_us)) - 1 if fraccion_us < 1 else None,
            'global_retorno': (us_final + mx_final) / capital - 1,
            'hija_retorno_aportaciones': retorno_hija,
            'hija_aportaciones': aportaciones_esperadas,
            'hija_escenarios_con_aportacion': float(con_aportacion.mean())}
