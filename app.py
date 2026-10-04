from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

from backtest import evaluar, preparar_intervalos

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Simulador Markowitz",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_activos(
    tickers: tuple[str, ...], fecha_inicial: date, fecha_final: date
) -> pd.DataFrame:
    """Descarga precios ajustados y calcula parámetros mensuales."""
    descarga = yf.download(
        list(tickers),
        start=fecha_inicial.isoformat(),
        end=(fecha_final + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if descarga.empty:
        raise ValueError("Yahoo Finance no devolvió precios para los activos elegidos.")

    if isinstance(descarga.columns, pd.MultiIndex):
        if "Close" not in descarga.columns.get_level_values(0):
            raise ValueError("La descarga no contiene precios de cierre.")
        precios = descarga["Close"].copy()
    else:
        precios = descarga[["Close"]].copy()
        precios.columns = [tickers[0]]

    precios.columns = [str(col).upper() for col in precios.columns]
    faltantes = [ticker for ticker in tickers if ticker not in precios.columns]
    vacios = [ticker for ticker in tickers if ticker in precios and precios[ticker].dropna().empty]
    invalidos = sorted(set(faltantes + vacios))
    if invalidos:
        raise ValueError(f"No se obtuvieron datos válidos para: {', '.join(invalidos)}.")

    precios = precios[list(tickers)].dropna(how="all").ffill()
    return precios


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_rf_irx(fecha_inicial: date, fecha_final: date) -> pd.Series:
    """Descarga ^IRX: cotización anual porcentual transformada a decimal."""
    datos = yf.download(
        "^IRX",
        start=fecha_inicial.isoformat(),
        end=(fecha_final + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if datos.empty:
        raise ValueError("Yahoo Finance no devolvió información para ^IRX.")

    cierre = datos["Close"]
    if isinstance(cierre, pd.DataFrame):
        cierre = cierre.iloc[:, 0]

    anual = pd.to_numeric(cierre, errors="coerce").dropna() / 100
    anual = anual[anual > -1]
    if anual.empty:
        raise ValueError("No quedaron observaciones válidas de ^IRX.")

    return anual


def ejecutar_simulacion(
    tickers: tuple[str, ...],
    media: np.ndarray,
    covarianzas: np.ndarray,
    repeticiones: int,
    portafolios: int,
    semilla: int,
    rf_anual_historica: np.ndarray,
    rf_mensual_historica: np.ndarray,
    max_cuadros: int = 25,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Ejecuta Monte Carlo y conserva el máximo Sharpe de cada corrida."""
    rng = np.random.default_rng(semilla)
    resultados: list[dict[str, float]] = []
    cuadros: list[dict[str, object]] = []
    guardar_cada = max(1, repeticiones // max_cuadros)
    progreso = st.progress(0, text="Preparando simulaciones…")

    for repeticion in range(1, repeticiones + 1):
        indice_rf = int(rng.integers(0, len(rf_mensual_historica)))
        rf_mensual = float(rf_mensual_historica[indice_rf])
        rf_anual = float(rf_anual_historica[indice_rf])

        pesos = rng.dirichlet(np.ones(len(tickers)), size=portafolios)
        rendimientos = pesos @ media
        varianzas = np.einsum("ij,jk,ik->i", pesos, covarianzas, pesos)
        desviaciones = np.sqrt(np.clip(varianzas, 0, None))
        sharpes = np.divide(
            rendimientos - rf_mensual,
            desviaciones,
            out=np.full_like(rendimientos, -np.inf),
            where=desviaciones > 0,
        )
        mejor = int(np.argmax(sharpes))

        fila: dict[str, float] = {
            "Repetición": repeticion,
            "Rf anual": rf_anual,
            "Rf mensual": rf_mensual,
            "Sharpe máximo mensual": float(sharpes[mejor]),
            "Sharpe anualizado": float(sharpes[mejor] * np.sqrt(12)),
            "Rendimiento mensual": float(rendimientos[mejor]),
            "Rendimiento anualizado": float((1 + rendimientos[mejor]) ** 12 - 1),
            "Volatilidad mensual": float(desviaciones[mejor]),
            "Volatilidad anualizada": float(desviaciones[mejor] * np.sqrt(12)),
        }
        fila.update(
            {f"Peso {ticker}": float(pesos[mejor, i]) for i, ticker in enumerate(tickers)}
        )
        resultados.append(fila)

        if repeticion == 1 or repeticion % guardar_cada == 0:
            cantidad = min(1_500, portafolios)
            muestra = rng.choice(portafolios, size=cantidad, replace=False)
            if mejor not in muestra:
                muestra[0] = mejor
            cuadros.append(
                {
                    "repeticion": repeticion,
                    "rf_anual": rf_anual,
                    "riesgo": desviaciones[muestra] * np.sqrt(12),
                    "rendimiento": (1 + rendimientos[muestra]) ** 12 - 1,
                    "sharpe": sharpes[muestra] * np.sqrt(12),
                    "es_mejor": muestra == mejor,
                }
            )

        if repeticion == 1 or repeticion % max(1, repeticiones // 100) == 0:
            progreso.progress(
                repeticion / repeticiones,
                text=f"Corrida {repeticion:,} de {repeticiones:,}",
            )

    progreso.empty()
    return pd.DataFrame(resultados), cuadros


def crear_excel(
    resultados: pd.DataFrame,
    rendimientos: pd.DataFrame,
    precios: pd.DataFrame,
    tickers: tuple[str, ...],
    comparativa: pd.DataFrame,
) -> bytes:
    """Crea el libro descargable en memoria."""
    salida = BytesIO()
    columnas_pesos = [f"Peso {ticker}" for ticker in tickers]
    columnas_resumen = [
        "Rf anual",
        "Sharpe anualizado",
        "Rendimiento anualizado",
        "Volatilidad anualizada",
        *columnas_pesos,
    ]
    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        resultados.to_excel(writer, sheet_name="Maximos_Sharpe", index=False)
        resultados[columnas_resumen].describe().T.to_excel(writer, sheet_name="Resumen")
        precios.to_excel(writer, sheet_name="Precios_mensuales")
        rendimientos.to_excel(writer, sheet_name="Rendimientos")
        rendimientos.cov().to_excel(writer, sheet_name="Covarianzas")
        comparativa.to_excel(writer, sheet_name="Comparativa_mensual")
    return salida.getvalue()


def grafica_animada(cuadros: list[dict[str, object]]) -> go.Figure:
    """Construye una animación ligera de las nubes de portafolios."""
    filas = []
    for cuadro in cuadros:
        cantidad = len(cuadro["riesgo"])
        filas.append(
            pd.DataFrame(
                {
                    "Corrida": str(cuadro["repeticion"]),
                    "Volatilidad anualizada": cuadro["riesgo"],
                    "Rendimiento anualizado": cuadro["rendimiento"],
                    "Sharpe anualizado": cuadro["sharpe"],
                    "Tipo": np.where(cuadro["es_mejor"], "Máximo Sharpe", "Portafolio"),
                    "Rf anual": np.repeat(cuadro["rf_anual"], cantidad),
                }
            )
        )
    datos = pd.concat(filas, ignore_index=True)
    fig = px.scatter(
        datos,
        x="Volatilidad anualizada",
        y="Rendimiento anualizado",
        color="Sharpe anualizado",
        symbol="Tipo",
        animation_frame="Corrida",
        hover_data={"Rf anual": ":.2%"},
        color_continuous_scale="Viridis",
        range_x=[datos["Volatilidad anualizada"].min() * 0.95,
                 datos["Volatilidad anualizada"].max() * 1.05],
        range_y=[datos["Rendimiento anualizado"].min() - 0.02,
                 datos["Rendimiento anualizado"].max() + 0.02],
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.6})
    fig.update_layout(height=620, legend_title_text="Resultado")
    return fig


st.title("📈 Simulador de portafolios de Markowitz")
st.write(
    "Ingresa cinco activos de Yahoo Finance. La aplicación genera portafolios "
    "aleatorios, incorpora una tasa libre de riesgo y conserva el Sharpe máximo "
    "de cada corrida."
)

with st.sidebar:
    st.header("Configuración")
    with st.form("formulario"):
        st.caption("Símbolos tal como aparecen en Yahoo Finance")
        valores_default = ["MSFT", "TSLA", "INTC", "AMX", "AAPL"]
        entradas = [
            st.text_input(f"Activo {i + 1}", value=valor)
            for i, valor in enumerate(valores_default)
        ]
        fecha_inicial = st.date_input("Inicio del histórico", value=date(2020, 1, 1))
        fecha_corte = st.date_input("Corte para estimar pesos", value=date(2025, 9, 1))
        fecha_final = st.date_input("Cierre de evaluación", value=date(2026, 9, 1))
        capital = st.number_input("Capital inicial", min_value=1.0, value=100_000.0, step=1_000.0)
        umbral_pct = st.number_input(
            "Umbral mensual para trasladar el excedente (%)",
            min_value=0.0, max_value=99.0, value=1.5, step=0.1,
        )

        metodo_rf = st.selectbox(
            "Tasa libre de riesgo",
            ["Bootstrap histórico de ^IRX (USD)", "Tasa anual fija"],
        )
        tasa_fija_pct = st.number_input(
            "Tasa fija anual (%)",
            min_value=-5.0,
            max_value=50.0,
            value=4.0,
            step=0.1,
            disabled=metodo_rf != "Tasa anual fija",
        )

        repeticiones = st.number_input(
            "Número de corridas", min_value=10, max_value=10_000, value=1_000, step=10
        )
        portafolios = st.number_input(
            "Portafolios por corrida",
            min_value=100,
            max_value=25_000,
            value=5_000,
            step=100,
        )
        semilla = st.number_input(
            "Semilla aleatoria", min_value=0, max_value=2_147_483_647, value=19_200_837
        )
        aceptar_moneda = st.checkbox(
            "Confirmo que los cinco activos están expresados en una moneda comparable."
        )
        ejecutar = st.form_submit_button("Ejecutar análisis", type="primary", use_container_width=True)


if ejecutar:
    tickers = tuple(texto.strip().upper() for texto in entradas)
    errores = []
    if any(not ticker for ticker in tickers):
        errores.append("Debes completar los cinco tickers.")
    if len(set(tickers)) != 5:
        errores.append("Los cinco tickers deben ser diferentes.")
    if not fecha_inicial < fecha_corte < fecha_final <= date.today():
        errores.append("Usa inicio histórico < corte < cierre ≤ hoy.")
    if not aceptar_moneda:
        errores.append("Confirma que los activos utilizan una moneda comparable.")

    if errores:
        for error in errores:
            st.error(error)
    else:
        try:
            with st.spinner("Descargando precios y preparando datos mensuales…"):
                diarios = descargar_activos(
                    tickers, fecha_inicial,
                    min(fecha_final + timedelta(days=7), date.today()),
                )
                # El corte es exclusivo: ningún precio posterior entra al ajuste.
                pre_corte = diarios.loc[diarios.index < pd.Timestamp(fecha_corte)]
                precios = pre_corte.resample("ME").last().dropna()
                rendimientos = precios.pct_change(fill_method=None).dropna()
                if len(rendimientos) < 24:
                    raise ValueError("Se requieren al menos 24 rendimientos mensuales antes del corte.")
                media = rendimientos.mean().to_numpy(dtype=float)
                covarianzas = rendimientos.cov().to_numpy(dtype=float)

                if metodo_rf.startswith("Bootstrap"):
                    rf_diaria = descargar_rf_irx(fecha_inicial, fecha_final)
                    rf_pre_corte = rf_diaria.loc[rf_diaria.index < pd.Timestamp(fecha_corte)]
                    rf_anual_mensual = rf_pre_corte.resample("ME").last().dropna()
                    rf = pd.DataFrame({
                        "Rf anual": rf_anual_mensual,
                        "Rf mensual": (1 + rf_anual_mensual) ** (1 / 12) - 1,
                    })
                    rf_anual = rf["Rf anual"].to_numpy(dtype=float)
                    rf_mensual = rf["Rf mensual"].to_numpy(dtype=float)
                else:
                    anual = tasa_fija_pct / 100
                    mensual = (1 + anual) ** (1 / 12) - 1
                    rf = pd.DataFrame({"Rf anual": [anual], "Rf mensual": [mensual]})
                    rf_anual = np.array([anual])
                    rf_mensual = np.array([mensual])
                    rf_diaria = None
                if not len(rf_anual):
                    raise ValueError("No hay tasas libres de riesgo anteriores al corte.")

            resultados, cuadros = ejecutar_simulacion(
                tickers=tickers,
                media=media,
                covarianzas=covarianzas,
                repeticiones=int(repeticiones),
                portafolios=int(portafolios),
                semilla=int(semilla),
                rf_anual_historica=rf_anual,
                rf_mensual_historica=rf_mensual,
            )
            mejor = resultados.loc[resultados["Sharpe anualizado"].idxmax()]
            pesos_mejor = np.array([mejor[f"Peso {ticker}"] for ticker in tickers])
            intervalos = preparar_intervalos(
                diarios,
                rf_diaria,
                fecha_corte,
                fecha_final,
                tasa_fija=anual if metodo_rf == "Tasa anual fija" else None,
            )
            comparativa = evaluar(
                intervalos, tickers, pesos_mejor, float(capital), media,
                float(mejor["Rf mensual"]), float(umbral_pct) / 100,
            )
            st.session_state["analisis"] = {
                "tickers": tickers,
                "precios": precios,
                "rendimientos": rendimientos,
                "rf": rf,
                "resultados": resultados,
                "cuadros": cuadros,
                "comparativa": comparativa,
                "configuracion": {
                    "repeticiones": int(repeticiones),
                    "portafolios": int(portafolios),
                    "metodo_rf": metodo_rf,
                    "capital": float(capital),
                    "umbral_pct": float(umbral_pct),
                    "fecha_corte": fecha_corte,
                    "fecha_final": fecha_final,
                },
            }
        except Exception as exc:
            st.error(f"No fue posible completar el análisis: {exc}")


if "analisis" not in st.session_state:
    st.info("Configura los cinco activos en el panel lateral y presiona **Ejecutar análisis**.")
    st.warning(
        "Para usar ^IRX, lo recomendable es elegir activos cotizados en USD. Por ejemplo, "
        "usa AMX (ADR en dólares) en lugar de AMXL.MX (pesos mexicanos)."
    )
    st.stop()


analisis = st.session_state["analisis"]
tickers = analisis["tickers"]
precios = analisis["precios"]
rendimientos = analisis["rendimientos"]
rf = analisis["rf"]
resultados = analisis["resultados"]
cuadros = analisis["cuadros"]
configuracion = analisis["configuracion"]
comparativa = analisis["comparativa"]
columnas_pesos = [f"Peso {ticker}" for ticker in tickers]
mejor = resultados.loc[resultados["Sharpe anualizado"].idxmax()]

st.success(
    f"Análisis terminado: {configuracion['repeticiones'] * configuracion['portafolios']:,} "
    "portafolios evaluados."
)

tab_resumen, tab_datos, tab_simulacion, tab_animacion, tab_comparativa, tab_linea = st.tabs(
    ["Resumen", "Datos históricos", "Simulaciones", "Animación", "Esperado vs. real", "Línea de capital"]
)

with tab_resumen:
    st.subheader("Mejor portafolio observado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe anualizado", f"{mejor['Sharpe anualizado']:.3f}")
    c2.metric("Rendimiento anualizado", f"{mejor['Rendimiento anualizado']:.2%}")
    c3.metric("Volatilidad anualizada", f"{mejor['Volatilidad anualizada']:.2%}")
    c4.metric("Rf anual utilizada", f"{mejor['Rf anual']:.2%}")

    pesos_mejor = pd.DataFrame(
        {"Activo": tickers, "Peso": [mejor[col] for col in columnas_pesos]}
    )
    izquierda, derecha = st.columns([1, 1])
    with izquierda:
        fig_pesos = px.bar(
            pesos_mejor,
            x="Activo",
            y="Peso",
            text_auto=".1%",
            title="Composición del mejor portafolio",
        )
        fig_pesos.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_pesos, use_container_width=True)
    with derecha:
        st.dataframe(
            pesos_mejor.style.format({"Peso": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"El portafolio apareció en la corrida {int(mejor['Repetición']):,}.")

    excel = crear_excel(resultados, rendimientos, precios, tickers, comparativa)
    st.download_button(
        "Descargar resultados en Excel",
        data=excel,
        file_name="resultados_markowitz_maximo_sharpe.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

with tab_datos:
    st.subheader("Precios y rendimientos mensuales")
    precios_base_100 = precios.div(precios.iloc[0]).mul(100)
    fig_precios = px.line(
        precios_base_100,
        title="Evolución comparativa de precios (base 100)",
        labels={"value": "Índice base 100", "Date": "Fecha", "variable": "Activo"},
    )
    st.plotly_chart(fig_precios, use_container_width=True)

    st.write(f"Meses comunes utilizados: **{len(rendimientos):,}**")
    descriptivos = rendimientos.describe().T[["mean", "std", "min", "max"]]
    st.dataframe(descriptivos.style.format("{:.4%}"), use_container_width=True)

    correlaciones = rendimientos.corr()
    fig_corr = px.imshow(
        correlaciones,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matriz de correlaciones mensuales",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    if len(rf) > 1:
        fig_rf = px.line(
            rf,
            y="Rf anual",
            title="Histórico mensual de ^IRX",
            labels={"index": "Fecha", "Rf anual": "Tasa anual"},
        )
        fig_rf.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_rf, use_container_width=True)

with tab_simulacion:
    st.subheader("Resultados de los máximos de Sharpe")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            resultados,
            x="Sharpe anualizado",
            nbins=35,
            title="Distribución de máximos de Sharpe",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        pesos_promedio = resultados[columnas_pesos].mean().rename(
            index={f"Peso {ticker}": ticker for ticker in tickers}
        )
        fig_promedio = px.bar(
            x=pesos_promedio.index,
            y=pesos_promedio.values,
            labels={"x": "Activo", "y": "Peso promedio"},
            title="Peso promedio de los portafolios ganadores",
            text_auto=".1%",
        )
        fig_promedio.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_promedio, use_container_width=True)

    fig_ganadores = px.scatter(
        resultados,
        x="Volatilidad anualizada",
        y="Rendimiento anualizado",
        color="Rf anual",
        hover_data=["Repetición", "Sharpe anualizado"],
        title="Portafolios ganadores de cada corrida",
        color_continuous_scale="Turbo",
    )
    fig_ganadores.update_xaxes(tickformat=".1%")
    fig_ganadores.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_ganadores, use_container_width=True)

    st.dataframe(
        resultados.sort_values("Sharpe anualizado", ascending=False).head(100).style.format(
            {
                "Rf anual": "{:.2%}",
                "Rf mensual": "{:.3%}",
                "Rendimiento mensual": "{:.2%}",
                "Rendimiento anualizado": "{:.2%}",
                "Volatilidad mensual": "{:.2%}",
                "Volatilidad anualizada": "{:.2%}",
                **{col: "{:.2%}" for col in columnas_pesos},
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_animacion:
    st.subheader("Evolución de las nubes de portafolios")
    st.caption(
        "La animación utiliza una muestra de cada corrida seleccionada para mantener "
        "la aplicación fluida. Cada corrida completa sí participa en el resultado."
    )
    st.plotly_chart(grafica_animada(cuadros), use_container_width=True)

with tab_comparativa:
    st.subheader("Evaluación mensual fuera de muestra")
    st.write(
        f"Pesos y rendimientos estimados con datos anteriores al **{configuracion['fecha_corte']}**. "
        f"Valoraciones desde esa fecha hasta **{configuracion['fecha_final']}**; "
        "se usa el primer precio conjunto disponible desde cada fecha mensual. "
        "La tasa aplicada en cada intervalo es la conocida en su inicio."
    )
    ultimo = comparativa.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Esperado 70/30", f"{ultimo['Valor esperado 70/30']:,.2f}")
    c2.metric("Real con excedentes", f"{ultimo['Valor real con regla']:,.2f}")
    c3.metric("Real sin transferencias", f"{ultimo['Valor real sin regla']:,.2f}")
    st.write(
        f"**Excedente enviado a la tasa libre de riesgo:** "
        f"{ultimo['Excedente trasladado acumulado']:,.2f}. "
        f"**Saldo actual de esa cuenta:** {ultimo['Cuenta de excedentes']:,.2f}."
    )
    curva = comparativa[[
        "Valor esperado 70/30", "Valor real con regla", "Valor real sin regla"
    ]].copy()
    curva.loc[pd.Timestamp(comparativa.iloc[0]["Inicio"])] = configuracion["capital"]
    curva = curva.sort_index()
    fig_comparativa = px.line(
        curva, title="Capital esperado y observado mes a mes",
        labels={"index": "Fecha", "value": "Capital", "variable": "Trayectoria"},
    )
    st.plotly_chart(fig_comparativa, use_container_width=True)
    st.dataframe(
        comparativa.style.format({
            "Rendimiento riesgoso": "{:.2%}", "Umbral mensual": "{:.2%}",
            "Rf anual conocida": "{:.2%}", "Rf intervalo": "{:.2%}",
            "Excedente trasladado": "{:,.2f}",
            "Excedente trasladado acumulado": "{:,.2f}",
            "Principal riesgoso": "{:,.2f}", "Seguro inicial acumulado": "{:,.2f}",
            "Cuenta de excedentes": "{:,.2f}", "Valor real con regla": "{:,.2f}",
            "Valor real sin regla": "{:,.2f}", "Valor esperado 70/30": "{:,.2f}",
        }), use_container_width=True,
    )
    st.caption(
        "Regla: excedente = saldo riesgoso al inicio × max(rendimiento del intervalo − "
        f"{configuracion['umbral_pct']:.2f}%, 0). Se traslada al cierre y devenga la "
        "tasa libre de riesgo desde el intervalo siguiente. Los pesos de la parte "
        "riesgosa se restablecen cada mes. Sin impuestos ni costos de operación."
    )

with tab_linea:
    st.subheader("Línea de asignación de capital")
    st.write(
        "Recta que une la tasa libre de riesgo usada al elegir el portafolio "
        "con el portafolio riesgoso de máximo Sharpe. Cada coordenada "
        "está expresada en términos **mensuales**."
    )
    fraccion = np.linspace(0, 1.3, 80)
    rf_elegida = float(mejor["Rf mensual"])
    rendimiento_riesgoso = float(mejor["Rendimiento mensual"])
    volatilidad_riesgosa = float(mejor["Volatilidad mensual"])
    fig_linea = go.Figure()
    fig_linea.add_trace(go.Scatter(
        x=fraccion * volatilidad_riesgosa,
        y=rf_elegida + fraccion * (rendimiento_riesgoso - rf_elegida),
        mode="lines", name="Línea de asignación",
    ))
    for a, nombre in [(0, "Tasa libre de riesgo"), (0.7, "Asignación 70/30"),
                      (1, "Portafolio riesgoso")]:
        fig_linea.add_trace(go.Scatter(
            x=[a * volatilidad_riesgosa],
            y=[rf_elegida + a * (rendimiento_riesgoso - rf_elegida)],
            mode="markers+text", name=nombre, text=[nombre],
            textposition="top center", marker={"size": 11},
        ))
    fig_linea.update_layout(
        xaxis_title="Volatilidad mensual", yaxis_title="Rendimiento esperado mensual",
        height=520,
    )
    fig_linea.update_xaxes(tickformat=".1%")
    fig_linea.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_linea, use_container_width=True)
    st.latex(r"(\sigma_a,R_a)=(0,R_f)+a(\sigma_p,R_p-R_f)")
    st.caption(
        "Con cinco activos elegidos por la persona, esta es una línea de asignación "
        "de capital (CAL). Sería la CML en sentido estricto si el portafolio "
        "riesgoso representara al portafolio de mercado."
    )

st.divider()
st.caption(
    "Modelo educativo. Los rendimientos históricos y las simulaciones no garantizan "
    "resultados futuros ni constituyen asesoría financiera."
)
