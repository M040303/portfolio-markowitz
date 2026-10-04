"""Evaluación fuera de muestra de la asignación 70/30 y el excedente."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def fechas_objetivo(inicio: date, fin: date) -> list[pd.Timestamp]:
    """Fechas mensuales ancladas al mismo día del mes, con cierre final exacto."""
    if inicio >= fin:
        raise ValueError("El inicio debe ser anterior al cierre.")
    fechas = [pd.Timestamp(inicio)]
    n = 1
    while True:
        candidata = pd.Timestamp(inicio) + pd.DateOffset(months=n)
        if candidata >= pd.Timestamp(fin):
            break
        fechas.append(candidata)
        n += 1
    fechas.append(pd.Timestamp(fin))
    return fechas


def preparar_intervalos(
    precios_diarios: pd.DataFrame,
    tasa_anual_diaria: pd.Series | None,
    inicio: date,
    fin: date,
    tasa_fija: float | None = None,
) -> pd.DataFrame:
    """Fija valuaciones en el primer día común de operación desde cada fecha objetivo.

    La tasa anual se observa al inicio del intervalo, nunca al cierre posterior.
    """
    if tasa_fija is None and tasa_anual_diaria is None:
        raise ValueError("Falta la serie de tasa o una tasa fija.")
    precios = precios_diarios.sort_index().dropna()
    if precios.empty:
        raise ValueError("No hay precios completos antes del cierre de evaluación.")
    fechas = fechas_objetivo(inicio, fin)
    if precios.index.max() < fechas[-1]:
        raise ValueError("Faltan precios para el cierre solicitado.")
    posiciones = precios.index.searchsorted(fechas, side="left")
    if any(pos >= len(precios) for pos in posiciones):
        raise ValueError("Falta un precio posterior a alguna fecha objetivo.")
    observadas = precios.index[posiciones]
    if len(set(observadas)) != len(observadas):
        raise ValueError("No hay datos suficientes para distinguir los meses.")
    if (observadas[-1] - fechas[-1]).days > 7:
        raise ValueError("El precio de cierre está demasiado lejos de la fecha objetivo.")
    muestra = precios.loc[observadas]
    rentabilidades = muestra.pct_change(fill_method=None).iloc[1:]
    tasa = None
    if tasa_fija is None:
        tasa = tasa_anual_diaria.sort_index().dropna()
        tasa.index = pd.to_datetime(tasa.index).tz_localize(None)
        tasa = tasa[tasa.index <= pd.Timestamp(fin)]
    filas = []
    for i, (fecha, row) in enumerate(rentabilidades.iterrows(), start=1):
        arranque = observadas[i - 1]
        if tasa_fija is None:
            # El cierre del mismo día puede no conocerse al momento de comprar.
            anteriores = tasa.loc[tasa.index < arranque]
            if anteriores.empty:
                raise ValueError(f"No hay una tasa conocida al {arranque.date()}.")
            anual = float(anteriores.iloc[-1])
        else:
            anual = float(tasa_fija)
        if anual <= -1:
            raise ValueError("La tasa anual debe ser mayor que -100%.")
        dias = (fecha - arranque).days
        rf_periodo = (1 + anual) ** (dias / 365.25) - 1
        filas.append(
            {
                "Inicio": arranque,
                "Cierre": fecha,
                "Días": dias,
                "Rf anual conocida": anual,
                "Rf del intervalo": rf_periodo,
                **{str(col): float(value) for col, value in row.items()},
            }
        )
    return pd.DataFrame(filas).set_index("Cierre")


def evaluar(
    intervalos: pd.DataFrame,
    tickers: tuple[str, ...],
    pesos: np.ndarray,
    capital: float,
    media_mensual: np.ndarray,
    rf_mensual_proyectada: float,
    umbral_mensual: float = 0.015,
) -> pd.DataFrame:
    """Aplica el umbral al rendimiento mensual de la parte riesgosa.

    Los pesos se restablecen al comienzo de cada intervalo; los excedentes
    trasladados no vuelven a la parte riesgosa.
    """
    if capital <= 0 or not 0 <= umbral_mensual < 1:
        raise ValueError("El capital debe ser positivo y el umbral debe estar entre 0 y 1.")
    if not np.isclose(np.sum(pesos), 1) or np.any(pesos < 0):
        raise ValueError("Los pesos del portafolio deben sumar uno y ser no negativos.")
    inicial_riesgoso = capital * 0.7
    inicial_seguro = capital * 0.3
    principal = referencia_riesgosa = inicial_riesgoso
    seguro = referencia_segura = inicial_seguro
    excedentes = transferido_total = 0.0
    rendimiento_esperado_riesgoso = float(pesos @ media_mensual)
    filas = []
    for periodo, (cierre, row) in enumerate(intervalos.iterrows(), start=1):
        rendimiento_riesgoso = float(row[list(tickers)].to_numpy(dtype=float) @ pesos)
        rendimiento_seguro = float(row["Rf del intervalo"])
        saldo_inicial = principal
        valor_antes = saldo_inicial * (1 + rendimiento_riesgoso)
        transferencia = max(0.0, saldo_inicial * (rendimiento_riesgoso - umbral_mensual))
        principal = valor_antes - transferencia
        seguro *= 1 + rendimiento_seguro
        excedentes = excedentes * (1 + rendimiento_seguro) + transferencia
        transferido_total += transferencia
        referencia_riesgosa *= 1 + rendimiento_riesgoso
        referencia_segura *= 1 + rendimiento_seguro
        meses = periodo  # horizonte de proyección: intervalos mensuales
        valor_esperado = (
            inicial_riesgoso * (1 + rendimiento_esperado_riesgoso) ** meses
            + inicial_seguro * (1 + rf_mensual_proyectada) ** meses
        )
        filas.append(
            {
                "Inicio": row["Inicio"],
                "Cierre": cierre,
                "Días": int(row["Días"]),
                "Rendimiento riesgoso": rendimiento_riesgoso,
                "Umbral mensual": umbral_mensual,
                "Rf anual conocida": row["Rf anual conocida"],
                "Rf intervalo": rendimiento_seguro,
                "Excedente trasladado": transferencia,
                "Excedente trasladado acumulado": transferido_total,
                "Principal riesgoso": principal,
                "Seguro inicial acumulado": seguro,
                "Cuenta de excedentes": excedentes,
                "Valor real con regla": principal + seguro + excedentes,
                "Valor real sin regla": referencia_riesgosa + referencia_segura,
                "Valor esperado 70/30": valor_esperado,
            }
        )
    return pd.DataFrame(filas).set_index("Cierre")
