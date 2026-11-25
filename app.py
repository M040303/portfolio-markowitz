"""Portafolio internacional prospectivo: México, EUA y USD/MXN.

Ejecutar con: streamlit run app.py
La evaluación histórica anterior sigue disponible en app_historico.py.
"""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from prospectivo import (
    ajustar_fx_arma_garch, bootstrap_rendimiento_global, estadisticas_riesgo, optimizar_asignacion_paises, optimizar_markowitz,
    preparar_rendimientos, preparar_rendimientos_diarios, rendimientos_esperados, simular_fx,
    seleccionar_activos_por_correlacion, simular_portafolio, sortear_tasa,
    tabla_resumen_componentes,
)

# Universos acotados de emisoras grandes; no son un ranking vivo por capitalización.
CANDIDATAS_US = ('MSFT','AAPL','NVDA','AMZN','GOOGL','META','BRK-B',
                 'JPM','V','XOM','WMT','JNJ','PG','COST')
CANDIDATAS_MX = ('GMEXICOB.MX','GFNORTEO.MX','AMXB.MX','FEMSAUBD.MX',
                 'CEMEXCPO.MX','WALMEX.MX','GAPB.MX','ASURB.MX',
                 'BIMBOA.MX','AC.MX','KIMBERA.MX','GRUMAB.MX',
                 'BBAJIOO.MX','CUERVO.MX')

st.set_page_config(page_title='Portafolio internacional | México y EUA', page_icon='🌎', layout='wide')
st.markdown('<style>.block-container{padding-top:1.4rem;padding-bottom:3rem}</style>', unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_precios(tickers: tuple[str, ...], inicio: date, fin: date,
                      estricto: bool = True) -> pd.DataFrame:
    datos = yf.download(list(tickers), start=inicio.isoformat(),
                        end=(fin + timedelta(days=1)).isoformat(), interval='1d',
                        auto_adjust=True, progress=False, threads=True)
    if datos.empty:
        raise ValueError('Yahoo Finance no devolvió precios de activos.')
    if isinstance(datos.columns, pd.MultiIndex):
        precios = datos['Close'].copy()
    else:
        precios = datos[['Close']].copy()
        precios.columns = [tickers[0]]
    precios.columns = [str(c).upper() for c in precios.columns]
    precios.index = pd.to_datetime(precios.index).tz_localize(None)
    faltan = [t for t in tickers if t not in precios.columns or precios[t].dropna().empty]
    if faltan and estricto:
        raise ValueError('Sin precios para: ' + ', '.join(faltan))
    return precios.reindex(columns=list(tickers)).sort_index().ffill(limit=5)


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_dolar(inicio: date, fin: date) -> pd.Series:
    datos = yf.download('MXN=X', start=inicio.isoformat(),
                        end=(fin + timedelta(days=1)).isoformat(), interval='1d',
                        auto_adjust=False, progress=False)
    if datos.empty:
        raise ValueError('Yahoo Finance no devolvió USD/MXN (MXN=X).')
    cierre = datos['Close']
    if isinstance(cierre, pd.DataFrame):
        cierre = cierre.iloc[:, 0]
    cierre = pd.to_numeric(cierre, errors='coerce').dropna()
    cierre.index = pd.to_datetime(cierre.index).tz_localize(None)
    return cierre[cierre > 0].sort_index()


@st.cache_data(show_spinner=False)
def calibrar_cache(precios: pd.Series):
    return ajustar_fx_arma_garch(precios)


def trazar_optimizacion(cuadros: list[dict], tabla: pd.DataFrame,
                       titulo: str, fraccion_riesgo: float) -> go.Figure:
    """Nube mensual, máximo actual, mínimos y máximos previos persistentes."""
    def trazas(c):
        anteriores = tabla.iloc[:int(c['corrida']) - 1]
        return [
            go.Scatter(x=c['vol'], y=c['mu'], mode='markers', name='Carteras de la corrida',
                       marker=dict(size=5, color='#8CA9BF', opacity=.28)),
            go.Scatter(x=anteriores['Volatilidad máximo Sharpe'],
                       y=anteriores['Retorno máximo Sharpe'], mode='markers',
                       name='Máximos Sharpe acumulados', marker=dict(size=8, color='#CF3D36')),
            go.Scatter(x=[c['mejor_vol']], y=[c['mejor_mu']], mode='markers',
                       name='Máximo Sharpe actual', marker=dict(size=17, symbol='star', color='#AE1D2A')),
            go.Scatter(x=[c['min_vol']], y=[c['min_mu']], mode='markers',
                       name='Mínima varianza actual', marker=dict(size=13, symbol='diamond', color='#008C82')),
            go.Scatter(x=[0, c['mejor_vol']], y=[c['rf_mes'], c['mejor_mu']],
                       mode='lines', name='Línea de asignación',
                       line=dict(color='#172D4E', width=2, dash='dash')),
            go.Scatter(x=[fraccion_riesgo * c['mejor_vol']],
                       y=[c['rf_mes'] + fraccion_riesgo * (c['mejor_mu'] - c['rf_mes'])],
                       mode='markers', name=f'Asignación {fraccion_riesgo:.0%}/{1-fraccion_riesgo:.0%}',
                       marker=dict(size=12, color='#E99B2E')),
        ]
    xs = np.r_[np.concatenate([c['vol'] for c in cuadros]), tabla['Volatilidad máximo Sharpe']]
    ys = np.r_[np.concatenate([c['mu'] for c in cuadros]), tabla['Retorno máximo Sharpe'],
               [c['rf_mes'] for c in cuadros]]
    x_max = max(float(np.max(xs)), .001)
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    y_margin = max((y_max-y_min)*.12, .004)
    frames = [go.Frame(name=str(c['corrida']), data=trazas(c),
                       layout=go.Layout(title=f'{titulo} · corrida {c["corrida"]:,}')) for c in cuadros]
    fig = go.Figure(data=trazas(cuadros[0]), frames=frames)
    fig.update_layout(title=f'{titulo} · corrida {cuadros[0]["corrida"]:,}', height=600,
                      xaxis=dict(title='Volatilidad mensual', range=[-.03*x_max, 1.12*x_max], tickformat='.1%'),
                      yaxis=dict(title='Rendimiento esperado mensual', range=[y_min-y_margin, y_max+y_margin], tickformat='.1%'),
                      legend=dict(orientation='h', y=1.13), margin=dict(t=110, b=80),
                      updatemenus=[dict(type='buttons', y=-.13, buttons=[
                          dict(label='▶ Reproducir', method='animate', args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
                          dict(label='⏸ Pausar', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
                      ])],
                      sliders=[dict(y=-.13, x=.22, len=.78, currentvalue=dict(prefix='Corrida: '),
                                    steps=[dict(label=str(c['corrida']), method='animate',
                                                args=[[str(c['corrida'])], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
                                           for c in cuadros])])
    return fig


def banda(resultados: np.ndarray, inicio: pd.Timestamp, nombre: str, eje: str) -> go.Figure:
    q = np.quantile(resultados, [.05, .5, .95], axis=0)
    fechas = pd.bdate_range(inicio, periods=resultados.shape[1]) if resultados.shape[1] == 253 else [inicio + pd.DateOffset(months=i) for i in range(resultados.shape[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas, y=q[2], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fechas, y=q[0], mode='lines', fill='tonexty',
                             fillcolor='rgba(41,105,163,.18)', line=dict(width=0), name='Banda 5–95%'))
    fig.add_trace(go.Scatter(x=fechas, y=q[1], mode='lines+markers',
                             line=dict(color='#174A77', width=3), name='Mediana'))
    fig.update_layout(title=nombre, xaxis_title='Sesión proyectada', yaxis_title=eje,
                      height=430, hovermode='x unified')
    return fig


def animacion_capital(simulacion: dict[str, np.ndarray], inicio: pd.Timestamp,
                     escenario: int | None = None) -> go.Figure:
    """Capital diario de una ruta o media; los componentes suman el total."""
    dias = simulacion['capital'].shape[1] - 1
    fechas = pd.bdate_range(inicio, periods=dias + 1)
    obtener = (lambda clave: simulacion[clave].mean(axis=0)) if escenario is None else (lambda clave: simulacion[clave][escenario])
    tipo = 'promedio' if escenario is None else f'escenario {escenario + 1}'
    series = {
        'EUA · sin hija': obtener('eua'),
        'México · sin hija': obtener('mexico'),
        'Hija · ambas monedas': obtener('hijo'),
        'Capital total': obtener('capital'),
    }
    colores = {'EUA · sin hija': '#2874A6', 'México · sin hija': '#16A085',
               'Hija · ambas monedas': '#D68910', 'Capital total': '#243447'}
    dominios = [(0, .24), (.26, .49), (.51, .74), (.76, 1)]

    def trazas(dia: int) -> list[go.BaseTraceType]:
        lineas = [go.Scatter(x=fechas[:dia+1], y=valores[:dia+1],
                            mode='lines+markers', name=nombre,
                            line=dict(color=colores[nombre], width=4 if nombre == 'Capital total' else 2.5),
                            marker=dict(size=8),
                            hovertemplate='%{x|%d %b %Y}<br>$%{y:,.0f} MXN<extra>'+nombre+'</extra>')
                  for nombre, valores in series.items()]
        indicadores = [go.Indicator(mode='number', value=float(valores[dia]),
                                   title=dict(text=nombre, font=dict(size=12)),
                                   number=dict(prefix='$', valueformat=',.0f',
                                               font=dict(size=21, color=colores[nombre])),
                                   domain=dict(x=list(dominios[k]), y=[.79, .98]))
                       for k, (nombre, valores) in enumerate(series.items())]
        return [*lineas, *indicadores]

    frames = [go.Frame(name=str(i), data=trazas(i),
                       layout=go.Layout(title=f'Proyección del capital · {tipo} · sesión {i} de {dias}'))
              for i in range(dias + 1)]
    fig = go.Figure(data=trazas(0), frames=frames)
    maximo = float(max(np.max(v) for v in series.values()))
    fig.update_layout(title=f'Proyección del capital · {tipo} · sesión 0 de {dias}', height=690,
                      xaxis=dict(title='Fecha proyectada (días hábiles)', domain=[0, 1],
                                 range=[fechas[0]-pd.Timedelta(days=8), fechas[-1]+pd.Timedelta(days=8)],
                                 tickformat='%b %Y'),
                      yaxis=dict(title='Capital (MXN)', domain=[0, .71],
                                 range=[-maximo*.04, maximo*1.12], tickformat='$,.0f'),
                      legend=dict(orientation='h', x=0, y=.76),
                      margin=dict(t=75, b=100, l=80), hovermode='x unified',
                      updatemenus=[dict(type='buttons', x=0, y=-.13, buttons=[
                          dict(label='▶ Reproducir', method='animate',
                               args=[None, dict(frame=dict(duration=90, redraw=True),
                                                transition=dict(duration=0), fromcurrent=True)]),
                          dict(label='⏸ Pausar', method='animate',
                               args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
                      ])],
                      sliders=[dict(x=.22, y=-.13, len=.78,
                                    currentvalue=dict(prefix='Sesión: '),
                                    steps=[dict(label=str(i) if i % 21 == 0 or i == dias else '', method='animate',
                                                args=[[str(i)], dict(mode='immediate',
                                                                     frame=dict(duration=0, redraw=True))])
                                           for i in range(dias + 1)])])
    return fig


def grafica_asignacion_global(frontera: pd.DataFrame, asignacion: dict) -> go.Figure:
    """Combinaciones Markowitz entre las dos carteras país en MXN."""
    x = frontera['Volatilidad diaria'] * np.sqrt(252)
    y = frontera['Rendimiento histórico diario'] * 252
    i = int(round(asignacion['eua'] * 1000))
    j = int(round(asignacion['min_var_eua'] * 1000))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Combinaciones EUA/México',
                             line=dict(color='#2874A6', width=3),
                             customdata=frontera['Peso EUA'],
                             hovertemplate='Volatilidad: %{x:.1%}<br>Rendimiento medio anualizado: %{y:.1%}<br>Peso EUA: %{customdata:.1%}<extra></extra>'))
    for k, nombre, color in [(i, 'Máximo Sharpe', '#D35400'),
                             (j, 'Mínima varianza', '#8E44AD')]:
        fig.add_trace(go.Scatter(x=[x.iloc[k]], y=[y.iloc[k]], mode='markers', name=nombre,
                                 marker=dict(size=15, color=color, line=dict(color='white', width=2)),
                                 hovertemplate=f'{nombre}<br>Volatilidad: %{{x:.1%}}<br>Rendimiento medio anualizado: %{{y:.1%}}<extra></extra>'))
    fig.update_layout(title='Markowitz global · dos carteras país en MXN',
                      xaxis=dict(title='Volatilidad histórica anualizada', tickformat='.0%'),
                      yaxis=dict(title='Rendimiento medio histórico anualizado', tickformat='.0%'),
                      height=440, hovermode='closest')
    return fig


def grafica_correlaciones(corr: pd.DataFrame, titulo: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index),
                               zmin=-1, zmax=1, zmid=0, colorscale='RdBu',
                               text=np.round(corr.to_numpy(), 2), texttemplate='%{text:.2f}',
                               hovertemplate='%{y} / %{x}: %{z:.2f}<extra></extra>',
                               colorbar=dict(title='ρ')))
    fig.update_layout(title=titulo, height=max(410, 32 * len(corr) + 120),
                      xaxis=dict(tickangle=-45), yaxis=dict(autorange='reversed'),
                      margin=dict(b=90))
    return fig


def tasa_controls(nombre: str, clave: str, defecto: float):
    modo = st.selectbox(f'{nombre}: método', ['Triangular', 'Fija'], key=f'{clave}_modo')
    if modo == 'Fija':
        fijo = st.number_input(f'{nombre}: tasa anual (%)', min_value=0., max_value=100.,
                               value=defecto, step=.25, key=f'{clave}_fija') / 100
        return fijo, fijo, fijo
    cols = st.columns(3)
    minimo = cols[0].number_input('Mín. %', min_value=0., max_value=100.,
                                   value=max(0., defecto-1.), step=.25, key=f'{clave}_min') / 100
    moda = cols[1].number_input('Moda %', min_value=0., max_value=100.,
                                 value=defecto, step=.25, key=f'{clave}_moda') / 100
    maximo = cols[2].number_input('Máx. %', min_value=0., max_value=100.,
                                   value=defecto+1., step=.25, key=f'{clave}_max') / 100
    return minimo, moda, maximo


def tabla_pesos(analisis: dict) -> pd.DataFrame:
    filas = []
    for pais, tickers, pesos, fraccion_pais in [
        ('EUA', analisis['us'], analisis['pesos_us']['sharpe'], analisis['fr_us']),
        ('México', analisis['mx'], analisis['pesos_mx']['sharpe'], 1-analisis['fr_us']),
    ]:
        for ticker, peso in zip(tickers, pesos):
            filas.append({'País': pais, 'Activo': ticker,
                          'Peso en cartera riesgosa': peso,
                          'Peso en capital del país': peso * analisis['fr_riesgo'],
                          'Peso en capital global': peso * analisis['fr_riesgo'] * fraccion_pais})
        filas.append({'País': pais, 'Activo': 'Tasa libre de riesgo',
                      'Peso en cartera riesgosa': np.nan,
                      'Peso en capital del país': 1-analisis['fr_riesgo'],
                      'Peso en capital global': (1-analisis['fr_riesgo']) * fraccion_pais})
    return pd.DataFrame(filas)


def tabla_rendimientos(analisis: dict) -> pd.DataFrame:
    e = analisis['esperados']
    capital = analisis['capital']
    return pd.DataFrame([
        {'Componente': 'EUA · incluyendo su parte de la hija',
         'Base (MXN)': capital * analisis['fr_us'], 'Final esperado (MXN)': e['eua_final'],
         'Rendimiento esperado': e['eua_retorno']},
        {'Componente': 'México · incluyendo su parte de la hija',
         'Base (MXN)': capital * (1-analisis['fr_us']), 'Final esperado (MXN)': e['mexico_final'],
         'Rendimiento esperado': e['mexico_retorno']},
        {'Componente': 'Global', 'Base (MXN)': capital,
         'Final esperado (MXN)': e['global_final'],
         'Rendimiento esperado': e['global_retorno']},
        {'Componente': 'Hija · sobre aportaciones (no anualizado)',
         'Base (MXN)': e['hija_aportaciones'], 'Final esperado (MXN)': e['hija_final'],
         'Rendimiento esperado': e['hija_retorno_aportaciones']},
    ])


def tabla_tasas_libres(analisis: dict) -> pd.DataFrame:
    """Separa intereses en moneda local del efecto de convertir USD a MXN."""
    sim = analisis['simulacion']
    capital = analisis['capital']
    fr_seguro = 1 - analisis['fr_riesgo']
    spot_inicial = analisis['spot']
    inicial_eua_mxn = capital * analisis['fr_us'] * fr_seguro
    inicial_eua_usd = inicial_eua_mxn / spot_inicial
    inicial_mx_mxn = capital * (1-analisis['fr_us']) * fr_seguro
    final_eua_usd = float(sim['seguro_eua_usd'][:, -1].mean())
    final_eua_mxn = float((sim['seguro_eua_usd'][:, -1] * analisis['fx'][:, -1]).mean())
    final_mx_mxn = float(sim['seguro_mexico_mxn'][:, -1].mean())
    return pd.DataFrame([
        {'País': 'EUA · USD', 'Capital inicial seguro (MXN)': inicial_eua_mxn,
         'Tasa anual media sorteada': float(analisis['rf_us'].mean()),
         'Tasa anual p05': float(np.quantile(analisis['rf_us'], .05)),
         'Tasa anual p95': float(np.quantile(analisis['rf_us'], .95)),
         'Intereses esperados a FX inicial (MXN)': (final_eua_usd-inicial_eua_usd)*spot_inicial,
         'Saldo final seguro (MXN)': final_eua_mxn,
         'Rendimiento en moneda local': final_eua_usd/inicial_eua_usd-1 if inicial_eua_usd else np.nan,
         'Rendimiento en MXN con FX': final_eua_mxn/inicial_eua_mxn-1 if inicial_eua_mxn else np.nan},
        {'País': 'México · MXN', 'Capital inicial seguro (MXN)': inicial_mx_mxn,
         'Tasa anual media sorteada': float(analisis['rf_mx'].mean()),
         'Tasa anual p05': float(np.quantile(analisis['rf_mx'], .05)),
         'Tasa anual p95': float(np.quantile(analisis['rf_mx'], .95)),
         'Intereses esperados a FX inicial (MXN)': final_mx_mxn-inicial_mx_mxn,
         'Saldo final seguro (MXN)': final_mx_mxn,
         'Rendimiento en moneda local': final_mx_mxn/inicial_mx_mxn-1 if inicial_mx_mxn else np.nan,
         'Rendimiento en MXN con FX': final_mx_mxn/inicial_mx_mxn-1 if inicial_mx_mxn else np.nan},
    ])


def excel_descargable(analisis: dict) -> bytes:
    b = BytesIO()
    simul = analisis['simulacion']
    n = len(simul['capital'])
    escenarios = pd.DataFrame({
        'Escenario': np.arange(1, n+1), 'Capital final MXN': simul['capital'][:, -1],
        'EUA sin hija MXN': simul['eua'][:, -1],
        'México sin hija MXN': simul['mexico'][:, -1],
        'EUA con hija MXN': simul['eua_con_hija'][:, -1],
        'México con hija MXN': simul['mexico_con_hija'][:, -1],
        'Seguro EUA final USD': simul['seguro_eua_usd'][:, -1],
        'Seguro México final MXN': simul['seguro_mexico_mxn'][:, -1],
        'Capital final sin cambio FX': simul['capital_sin_fx'][:, -1],
        'Cuenta hija final MXN': simul['hijo'][:, -1],
        'USD/MXN final': analisis['fx'][:, -1], 'Rf EUA anual': analisis['rf_us'],
        'Rf México anual': analisis['rf_mx']})
    resumen = pd.DataFrame([{k: v for k, v in analisis['riesgo'].items()
                             if k not in ('var95', 'var95_pct')}]).T.rename(columns={0:'Valor'})
    resumen.loc['VaR global 95% (%)', 'Valor'] = 100 * analisis['riesgo']['var95_pct']
    trayectorias = pd.DataFrame({
        'Sesión': np.arange(253), 'Fecha aproximada': pd.bdate_range(analisis['fecha_base'], periods=253), 'Capital p05': np.quantile(simul['capital'], .05, axis=0),
        'Capital mediano': np.median(simul['capital'], axis=0),
        'Capital p95': np.quantile(simul['capital'], .95, axis=0),
        'EUA esperado sin hija': simul['eua'].mean(axis=0),
        'México esperado sin hija': simul['mexico'].mean(axis=0),
        'Hija esperada': simul['hijo'].mean(axis=0),
        'Global esperado': simul['capital'].mean(axis=0),
        'USD/MXN p05': np.quantile(analisis['fx'], .05, axis=0),
        'USD/MXN mediano': np.median(analisis['fx'], axis=0),
        'USD/MXN p95': np.quantile(analisis['fx'], .95, axis=0)})
    parametros = pd.DataFrame([vars(analisis['parametros_fx'])]).T.rename(columns={0:'Valor'})
    with pd.ExcelWriter(b, engine='openpyxl') as writer:
        resumen.to_excel(writer, sheet_name='Resumen')
        tabla_rendimientos(analisis).to_excel(writer, sheet_name='Rendimientos_esperados', index=False)
        tabla_pesos(analisis).to_excel(writer, sheet_name='Pesos', index=False)
        tabla_tasas_libres(analisis).to_excel(writer, sheet_name='Tasas_libres', index=False)
        tabla_resumen_componentes(simul, analisis['fx'], analisis['capital'],
                                  analisis['fr_us'], analisis['fr_riesgo']).to_excel(
                                      writer, sheet_name='Resumen_portafolio', index=False)
        analisis['corr_us'].to_excel(writer, sheet_name='Correlaciones_EUA')
        analisis['corr_mx'].to_excel(writer, sheet_name='Correlaciones_MX')
        pd.DataFrame([
            {'País': pais, 'Grupo': grupo, 'Activo': ticker,
             'Pares cruzados negativos': analisis[f'info_{clave}']['pares_negativos'],
             'Regla de seis cumplida': analisis[f'info_{clave}']['regla_estricta']}
            for pais, clave in [('EUA', 'us'), ('México', 'mx')]
            for grupo, lista in [('Tres', analisis[f'grupo_{clave}_3']),
                                 ('Dos', analisis[f'grupo_{clave}_2'])]
            for ticker in lista]).to_excel(writer, sheet_name='Seleccion_activos', index=False)
        pd.DataFrame([analisis['asignacion_global']]).to_excel(writer, sheet_name='Asignacion_global', index=False)
        analisis['frontera_global'].to_excel(writer, sheet_name='Markowitz_global', index=False)
        escenarios.to_excel(writer, sheet_name='Escenarios', index=False)
        analisis['tabla_bootstrap'].to_excel(writer, sheet_name='Bootstrap_resumen', index=False)
        analisis['replicas_bootstrap'].to_excel(writer, sheet_name='Bootstrap_replicas', index=False)
        trayectorias.to_excel(writer, sheet_name='Bandas_diarias', index=False)
        analisis['ganadores_us'].to_excel(writer, sheet_name='Markowitz_EUA', index=False)
        analisis['ganadores_mx'].to_excel(writer, sheet_name='Markowitz_Mexico', index=False)
        parametros.to_excel(writer, sheet_name='Modelo_USDMXN')
        analisis['retornos'].to_excel(writer, sheet_name='Retornos_historicos')
    return b.getvalue()


st.title('🌎 Portafolio internacional prospectivo')
st.write('Dos carteras Markowitz, una en USD y otra en MXN. El dólar se proyecta por separado y el resultado conjunto se expresa en pesos.')
st.caption('Markowitz determina la asignación EUA/México; el reparto 70/30 dentro de cada país se hace una sola vez al inicio. Después las posiciones conservan sus unidades. No hay comparación con resultados futuros todavía no observados.')

with st.sidebar:
    st.header('Configurar escenario')
    modo_activos = st.radio('Selección de activos', ['Elegir manualmente', 'Elegir automáticamente'])
    ventana_corr = st.selectbox('Ventana de correlación diaria',
                                [252, 504, 756, 'Todo el histórico'], index=0,
                                help='La selección automática y las matrices usan las últimas sesiones coincidentes. 252 sesiones ≈ un año bursátil.')
    ventana_corr = None if ventana_corr == 'Todo el histórico' else ventana_corr
    with st.form('configuracion'):
        st.subheader('Activos')
        if modo_activos == 'Elegir manualmente':
            t_us = [st.text_input(f'EUA · activo {i+1}', value=t, key=f'us_{i}') for i,t in enumerate(['MSFT','AAPL','NVDA','JNJ','XOM'])]
            t_mx = [st.text_input(f'México · activo {i+1}', value=t, key=f'mx_{i}') for i,t in enumerate(['WALMEX.MX','GFNORTEO.MX','FEMSAUBD.MX','AMXB.MX','GMEXICOB.MX'])]
            pool_us, pool_mx = (), ()
        else:
            st.caption('Candidatas grandes de mercados EUA y México; no es un ranking actualizado por capitalización. Se excluyen símbolos sin datos suficientes. La regla de correlación usa rendimientos diarios, no precios.')
            pool_us = tuple(st.multiselect('Candidatas EUA', CANDIDATAS_US, default=CANDIDATAS_US))
            pool_mx = tuple(st.multiselect('Candidatas México', CANDIDATAS_MX, default=CANDIDATAS_MX))
            t_us, t_mx = [], []
        st.subheader('Horizonte y capital')
        inicio = st.date_input('Inicio del histórico', value=date(2018,1,1))
        corte = st.date_input('Fecha de valoración', value=date.today())
        completar_historico = st.checkbox('Ampliar histórico si faltan meses para Markowitz', value=True,
            help='Si el rango elegido no alcanza 24 rendimientos mensuales, descarga datos desde tres años antes de la valoración. La fecha de valoración no cambia.')
        capital = st.number_input('Capital inicial (MXN)', min_value=1_000., value=1_000_000., step=50_000.)
        st.caption('La asignación EUA/México se calculará con Markowitz sobre los dos portafolios completos, expresados en MXN.')
        fr_riesgo = st.slider('Activos con riesgo dentro de cada país (%)', 0, 100, 70, 5) / 100
        st.subheader('Tasas libres de riesgo')
        st.caption('Entradas editables, expresadas como tasas anuales efectivas. Son supuestos; verifica CETES y T-Bills antes de tomar decisiones.')
        rf_us_par = tasa_controls('T-Bills / EUA', 'rf_us', 4.)
        rf_mx_par = tasa_controls('CETES / México', 'rf_mx', 8.)
        st.caption('Con tasa triangular se sortea una tasa anual nueva para cada corrida Markowitz y otra para cada escenario prospectivo. En un escenario, esa tasa permanece fija durante las 252 sesiones. La semilla permite repetir los mismos sorteos.')
        st.subheader('Simulación')
        corridas = st.number_input('Corridas Markowitz por país', 10, 1_000, 1_000, step=10)
        carteras = st.number_input('Carteras por corrida', 100, 5_000, 5_000, step=100)
        escenarios = st.number_input('Escenarios prospectivos', 100, 20_000, 10_000, step=1_000)
        remuestreos = st.number_input('Remuestreos bootstrap', 100, 20_000, 10_000, step=1_000,
            help='Remuestrea los rendimientos finales de los escenarios para estimar la precisión de sus estadísticos.')
        semilla = st.number_input('Semilla', 0, 2_147_483_647, 20261119)
        aplicar_hijo = st.checkbox('Trasladar excedente mensual a cartera hija segura', value=True)
        umbral = st.number_input('Umbral mensual (%)', min_value=0., max_value=99., value=1.5, step=.1,
                                 disabled=not aplicar_hijo) / 100
        ejecutar = st.form_submit_button('Generar análisis', type='primary', width='stretch')

if ejecutar:
    st.session_state.pop('prospectivo', None)
    try:
        if modo_activos == 'Elegir manualmente':
            us = tuple(t.strip().upper() for t in t_us)
            mx = tuple(t.strip().upper() for t in t_mx)
            todos = us + mx
            if any(not t for t in todos) or len(set(todos)) != len(todos):
                raise ValueError('Completa diez símbolos distintos: cinco por país.')
        else:
            if len(pool_us) < 5 or len(pool_mx) < 5:
                raise ValueError('Elige al menos cinco candidatas por país para la selección automática.')
            todos = pool_us + pool_mx
        if not inicio < corte <= date.today():
            raise ValueError('El histórico debe iniciar antes de la valoración y ésta no puede ser futura.')
        meses_potenciales = (corte.year - inicio.year) * 12 + corte.month - inicio.month - 1
        inicio_efectivo = inicio
        if completar_historico and meses_potenciales < 24:
            inicio_efectivo = min(inicio, (pd.Timestamp(corte) - pd.DateOffset(years=3)).date())
        if not (rf_us_par[0] <= rf_us_par[1] <= rf_us_par[2] and rf_mx_par[0] <= rf_mx_par[1] <= rf_mx_par[2]):
            raise ValueError('En cada tasa debe cumplirse mínimo ≤ moda ≤ máximo.')
        with st.spinner('Descargando históricos y estimando el modelo USD/MXN…'):
            precios_pool = descargar_precios(todos, inicio_efectivo, corte,
                                             estricto=modo_activos == 'Elegir manualmente')
            if modo_activos == 'Elegir automáticamente':
                grupo_us_3, grupo_us_2, corr_us, info_us = seleccionar_activos_por_correlacion(
                    precios_pool[list(pool_us)], pd.Timestamp(corte), ventana_corr)
                grupo_mx_3, grupo_mx_2, corr_mx, info_mx = seleccionar_activos_por_correlacion(
                    precios_pool[list(pool_mx)], pd.Timestamp(corte), ventana_corr)
                us, mx = tuple(grupo_us_3 + grupo_us_2), tuple(grupo_mx_3 + grupo_mx_2)
                precios = precios_pool[list(us + mx)]
            else:
                precios = precios_pool
                grupo_us_3, grupo_us_2, corr_us, info_us = seleccionar_activos_por_correlacion(
                    precios[list(us)], pd.Timestamp(corte), ventana_corr)
                grupo_mx_3, grupo_mx_2, corr_mx, info_mx = seleccionar_activos_por_correlacion(
                    precios[list(mx)], pd.Timestamp(corte), ventana_corr)
            fx_historico = descargar_dolar(inicio_efectivo, corte)
            retornos = preparar_rendimientos(precios, pd.Timestamp(corte))
            retornos_diarios = preparar_rendimientos_diarios(precios, pd.Timestamp(corte))
            if len(fx_historico) < 252:
                raise ValueError('Se necesitan al menos 252 precios diarios del dólar.')
            p_fx = calibrar_cache(fx_historico)
            s0 = float(fx_historico.iloc[-1])
            fecha_base = fx_historico.index[-1]
        with st.spinner('Optimizando las dos carteras y simulando 12 meses…'):
            t_opt_us = sortear_tasa(int(corridas), *rf_us_par, int(semilla)+1)
            t_opt_mx = sortear_tasa(int(corridas), *rf_mx_par, int(semilla)+2)
            gan_us, pesos_us, cuadros_us = optimizar_markowitz(retornos[list(us)], t_opt_us, int(corridas), int(carteras), int(semilla)+3)
            gan_mx, pesos_mx, cuadros_mx = optimizar_markowitz(retornos[list(mx)], t_opt_mx, int(corridas), int(carteras), int(semilla)+4)
            asignacion_global, frontera_global = optimizar_asignacion_paises(
                retornos_diarios, fx_historico, pesos_us['sharpe'], pesos_mx['sharpe'],
                list(us), list(mx), fr_riesgo, rf_us_par[1], rf_mx_par[1])
            fr_us = asignacion_global['eua']
            fx_diario = simular_fx(s0, p_fx, int(escenarios), 252, int(semilla)+5)
            fx = fx_diario
            rf_us = sortear_tasa(int(escenarios), *rf_us_par, int(semilla)+6)
            rf_mx = sortear_tasa(int(escenarios), *rf_mx_par, int(semilla)+7)
            simul = simular_portafolio(retornos_diarios, pesos_us['sharpe'], pesos_mx['sharpe'],
                                      list(us), list(mx), fx, rf_us, rf_mx,
                                      float(capital), fr_us, fr_riesgo, int(semilla)+8,
                                      umbral if aplicar_hijo else None)
            riesgo = estadisticas_riesgo(simul['capital'][:, -1], float(capital))
            esperados = rendimientos_esperados(simul, float(capital), fr_us)
        with st.spinner('Calculando intervalos bootstrap del rendimiento global…'):
            tabla_bootstrap, replicas_bootstrap = bootstrap_rendimiento_global(
                simul['capital'][:, -1], float(capital), int(remuestreos), int(semilla)+9)
        st.session_state['prospectivo'] = dict(us=us, mx=mx, retornos=retornos,
            inicio_solicitado=inicio, inicio_efectivo=inicio_efectivo,
            modo_activos=modo_activos, grupo_us_3=grupo_us_3, grupo_us_2=grupo_us_2,
            ventana_corr=ventana_corr,
            grupo_mx_3=grupo_mx_3, grupo_mx_2=grupo_mx_2,
            corr_us=corr_us, corr_mx=corr_mx, info_us=info_us, info_mx=info_mx,
            ganadores_us=gan_us, ganadores_mx=gan_mx, pesos_us=pesos_us,
            pesos_mx=pesos_mx, cuadros_us=cuadros_us, cuadros_mx=cuadros_mx,
            asignacion_global=asignacion_global, frontera_global=frontera_global,
            parametros_fx=p_fx, spot=s0, fecha_base=fecha_base, fx=fx,
            rf_us=rf_us, rf_mx=rf_mx, simulacion=simul, riesgo=riesgo,
            esperados=esperados, tabla_bootstrap=tabla_bootstrap,
            replicas_bootstrap=replicas_bootstrap, remuestreos=int(remuestreos),
            capital=float(capital), fr_us=fr_us, fr_riesgo=fr_riesgo,
            corridas=int(corridas), carteras=int(carteras), escenarios=int(escenarios),
            umbral=umbral if aplicar_hijo else None)
    except Exception as exc:
        st.error(f'No se completó el análisis: {exc}')

if 'prospectivo' not in st.session_state:
    st.info('Configura los activos y presiona **Generar análisis**.')
    st.stop()

a = st.session_state['prospectivo']
r = a['riesgo']
e = a['esperados']
st.success(f'{a["escenarios"]:,} escenarios de inversión · {a["corridas"]:,} corridas × {a["carteras"]:,} carteras por país')
if a['inicio_efectivo'] < a['inicio_solicitado']:
    st.info(f'Para estimar Markowitz se amplió el histórico: del inicio elegido {a["inicio_solicitado"]:%d/%m/%Y} al {a["inicio_efectivo"]:%d/%m/%Y}. La fecha de valoración permanece igual.')
if a['parametros_fx'].alpha + a['parametros_fx'].beta >= .995:
    st.warning('El GARCH estimado tiene persistencia muy alta. A un año, las bandas pueden ser especialmente sensibles a los parámetros y a los choques extremos.')

resumen, markowitz, trayectorias, modelo, datos = st.tabs(['Decisión', 'Markowitz y línea de capital', 'Escenarios a 12 meses', 'Modelo del dólar', 'Datos y descarga'])
with resumen:
    st.subheader('Rendimientos esperados a 12 meses')
    t1,t2,t3,t4 = st.columns(4)
    t1.metric('EUA · inversión inicial en MXN',
              f'{e["eua_retorno"]:.2%}' if e['eua_retorno'] is not None else 'Sin asignación')
    t2.metric('México · inversión inicial en MXN',
              f'{e["mexico_retorno"]:.2%}' if e['mexico_retorno'] is not None else 'Sin asignación')
    t3.metric('Global · inversión inicial en MXN', f'{e["global_retorno"]:.2%}')
    t4.metric('VaR global 95% · 12 meses', f'{r["var95_pct"]:.2%}')
    st.caption(f'Asignación optimizada entre países: EUA {a["fr_us"]:.1%} · México {1-a["fr_us"]:.1%}. El VaR expresa la pérdida al cierre como porcentaje del capital inicial.')
    st.dataframe(tabla_rendimientos(a).style.format(
        {'Base (MXN)': '${:,.0f}', 'Final esperado (MXN)': '${:,.0f}',
         'Rendimiento esperado': '{:.2%}'}, na_rep='—'),
        hide_index=True, width='stretch')
    st.subheader('Tabla resumen del portafolio')
    resumen_componentes = tabla_resumen_componentes(a['simulacion'], a['fx'],
                                                      a['capital'], a['fr_us'], a['fr_riesgo'])
    st.dataframe(resumen_componentes.style.format({
        'Inversión inicial (MXN)': '${:,.0f}',
        'Aportaciones esperadas (MXN)': '${:,.0f}',
        'Valor final esperado (MXN)': '${:,.0f}',
        'Ganancia esperada (MXN)': '${:,.0f}',
        'Rendimiento': '{:.2%}', 'Riesgo (desv. % sobre base)': '{:.2%}',
        'VaR 95%': '{:.2%}'}, na_rep='—'), hide_index=True, width='stretch')
    st.caption('La ganancia de las carteras de acciones incluye los excedentes retirados y enviados a la hija; la ganancia de la hija excluye esas aportaciones. Así, las ganancias de los seis componentes suman la ganancia global. En las hijas, el rendimiento y el riesgo se calculan sobre las aportaciones esperadas, no sobre un capital inicial de cero. El riesgo es la dispersión entre resultados finales simulados, no volatilidad diaria.')
    st.caption('EUA y México incluyen los excedentes originados en cada país para calcular su rendimiento sobre la inversión inicial. La hija se muestra aparte: inicia en $0 y su rendimiento se calcula sobre sus aportaciones acumuladas, sin anualizar. Los valores esperados son promedios de los escenarios simulados; EUA ya incluye la conversión USD/MXN.')
    st.subheader('Rendimiento de la parte en tasa libre de riesgo')
    tasas_tabla = tabla_tasas_libres(a)
    lado_rf_us, lado_rf_mx = st.columns(2)
    for columna, fila in [(lado_rf_us, tasas_tabla.iloc[0]),
                          (lado_rf_mx, tasas_tabla.iloc[1])]:
        with columna:
            st.markdown(f'**{fila["País"]}**')
            st.metric('Tasa anual media sorteada', f'{fila["Tasa anual media sorteada"]:.2%}')
            st.metric('Interés esperado de la parte segura',
                      f'${fila["Intereses esperados a FX inicial (MXN)"]:,.0f} MXN')
            rendimiento_local = fila['Rendimiento en moneda local']
            st.caption('Rendimiento de la parte segura en su moneda: '
                       + (f'{rendimiento_local:.2%}' if pd.notna(rendimiento_local)
                          else 'sin capital asignado') + '.')
    with st.expander('Ver tasas sorteadas, saldos y efecto de convertir USD a MXN'):
        st.dataframe(tasas_tabla.style.format({
            'Capital inicial seguro (MXN)': '${:,.0f}',
            'Tasa anual media sorteada': '{:.2%}', 'Tasa anual p05': '{:.2%}',
            'Tasa anual p95': '{:.2%}',
            'Intereses esperados a FX inicial (MXN)': '${:,.0f}',
            'Saldo final seguro (MXN)': '${:,.0f}',
            'Rendimiento en moneda local': '{:.2%}',
            'Rendimiento en MXN con FX': '{:.2%}',
        }, na_rep='—'), hide_index=True, width='stretch')
    st.caption('Esta tabla corresponde al capital colocado inicialmente en la tasa libre de riesgo de cada país (30% por defecto). En EUA, el interés se gana en USD; la última columna también incorpora el movimiento USD/MXN. Los excedentes de la hija usan las mismas tasas, pero se muestran en la cuenta hija para evitar duplicar intereses.')
    st.caption('Tasa triangular: cada corrida Markowitz recibe su propio sorteo y cada escenario de 252 sesiones recibe otro. La tasa de un escenario permanece fija durante toda la ruta. Con tasa fija, todas las corridas y escenarios usan el valor indicado.')
    st.subheader('Riesgo global')
    c1,c2,c3 = st.columns(3)
    c1.metric('Capital mediano final', f'${r["mediana"]:,.0f} MXN')
    c2.metric('Prob. de terminar bajo el capital inicial', f'{r["prob_perdida"]:.1%}')
    c3.metric('ES 95% · pérdida media de cola', f'${r["es95"]:,.0f}')
    st.caption(f'Capital inicial ${a["capital"]:,.0f} MXN · EUA {a["fr_us"]:.0%} · México {1-a["fr_us"]:.0%} · dentro de cada país: activos {a["fr_riesgo"]:.0%} / tasa {1-a["fr_riesgo"]:.0%}.')
    st.plotly_chart(banda(a['simulacion']['capital'], a['fecha_base'], 'Capital total expresado en MXN', 'MXN'), width='stretch')
    st.write('**Rango final por escenarios**')
    st.dataframe(pd.DataFrame([{'Percentil 5':r['p05'], 'Mediana':r['mediana'], 'Media':r['media'], 'Percentil 95':r['p95']}]).style.format('${:,.0f}'), hide_index=True, width='stretch')
    st.caption('El VaR mostrado arriba mide el percentil 95 de la pérdida respecto del capital inicial, en porcentaje; el ES es la pérdida media del 5% peor, en MXN. Ambas cifras dependen del modelo y son pérdidas al cierre, no necesidades intermedias de liquidez.')
    st.subheader('Bootstrap del rendimiento global')
    st.caption(f'Se calcularon {a["escenarios"]:,} escenarios prospectivos y luego {a["remuestreos"]:,} remuestreos con reemplazo de sus rendimientos finales. La estimación describe los escenarios; el intervalo bootstrap del 95% muestra la precisión de cada estadístico dentro de este modelo, no un intervalo de rendimientos futuros.')
    st.dataframe(a['tabla_bootstrap'].style.format({
        'Estimación': '{:,.2f}', 'IC bootstrap 2.5%': '{:,.2f}',
        'IC bootstrap 97.5%': '{:,.2f}'}), hide_index=True, width='stretch')
    st.caption('Todas las cifras son porcentajes, excepto la varianza, expresada en puntos porcentuales al cuadrado (pp²). VaR y ES se muestran como pérdidas positivas cuando corresponde.')
    figura_bootstrap = go.Figure(go.Histogram(x=a['replicas_bootstrap']['Media'], nbinsx=50,
        marker_color='#245A81', name='Medias bootstrap'))
    figura_bootstrap.add_vline(x=float(a['tabla_bootstrap'].loc[
        a['tabla_bootstrap']['Métrica'] == 'Media', 'Estimación'].iloc[0]),
        line_dash='dash', line_color='#CC5A37')
    figura_bootstrap.update_layout(title='Precisión de la media del rendimiento global',
        xaxis_title='Media anual estimada (%)', yaxis_title='Remuestreos', showlegend=False)
    st.plotly_chart(figura_bootstrap, width='stretch')

with markowitz:
    st.subheader('Correlaciones para elegir activos')
    ventana_etiqueta = (f'{a["ventana_corr"]} sesiones' if a['ventana_corr']
                       else 'todo el histórico')
    st.caption(f'Se calculan con rendimientos diarios de precios ajustados sobre {ventana_etiqueta}. El grupo de tres y el de dos se comparan en seis pares cruzados; una correlación negativa histórica no garantiza protección futura.')
    for pais, corr, tres, dos, info in [
        ('EUA', a['corr_us'], a['grupo_us_3'], a['grupo_us_2'], a['info_us']),
        ('México', a['corr_mx'], a['grupo_mx_3'], a['grupo_mx_2'], a['info_mx'])]:
        st.markdown(f'**{pais}:** grupo de tres: {", ".join(tres)} · grupo de dos: {", ".join(dos)}')
        if info['regla_estricta']:
            st.success(f'{pais}: las seis correlaciones cruzadas son negativas ({info["observaciones"]} rendimientos diarios).')
        else:
            st.warning(f'{pais}: sólo {info["pares_negativos"]} de 6 correlaciones cruzadas son negativas; no existe una combinación que cumpla la regla estricta dentro de estas {info["candidatas_validas"]} candidatas válidas.')
        st.plotly_chart(grafica_correlaciones(corr, f'Matriz de correlación diaria · {pais}'), width='stretch')
    st.subheader('Asignación global entre EUA y México')
    g1, g2, g3 = st.columns(3)
    g1.metric('Peso inicial EUA', f'{a["fr_us"]:.1%}')
    g2.metric('Peso inicial México', f'{1-a["fr_us"]:.1%}')
    g3.metric('Sharpe histórico global anualizado', f'{a["asignacion_global"]["sharpe_anual"]:.2f}')
    st.plotly_chart(grafica_asignacion_global(a['frontera_global'], a['asignacion_global']), width='stretch')
    st.caption('Cada punto combina los dos portafolios completos, incluyendo su parte 70/30, con pesos EUA/México de 0% a 100% y sin posiciones cortas. Se usan sesiones históricas coincidentes y USD/MXN histórico para medir ambos en MXN; el máximo Sharpe se calcula contra la tasa mexicana central. Esta estimación usa ponderaciones constantes para medir la frontera; la simulación prospectiva hace una compra inicial y no rebalancea.')
    st.subheader('Optimización dentro de cada país')
    st.write('Los puntos rojos conservan el máximo Sharpe encontrado en corridas previas; el rombo identifica la mínima varianza de la corrida y el punto naranja muestra la asignación entre activos y tasa configurada. La tasa cambia entre corridas conforme a su distribución.')
    pesos_tabla = tabla_pesos(a)
    lado_us, lado_mx = st.columns(2)
    with lado_us:
        st.plotly_chart(trazar_optimizacion(a['cuadros_us'], a['ganadores_us'], 'EUA · USD', a['fr_riesgo']), width='stretch')
        mejor_us = a['ganadores_us'].loc[a['ganadores_us']['Sharpe máximo'].idxmax()]
        retorno_us = float(mejor_us['Retorno máximo Sharpe'])
        st.metric('Rendimiento esperado de la cartera riesgosa · EUA', f'{(1+retorno_us)**12-1:.2%}')
        st.metric('Tasa libre EUA · promedio sorteado', f'{a["rf_us"].mean():.2%}')
        st.caption(f'Estimación basada en rendimientos históricos: {retorno_us:.2%} mensual, en USD.')
        st.dataframe(pesos_tabla.loc[pesos_tabla['País']=='EUA'].style.format(
            {'Peso en cartera riesgosa':'{:.1%}', 'Peso en capital del país':'{:.1%}',
             'Peso en capital global':'{:.1%}'}, na_rep='—'), hide_index=True, width='stretch')
    with lado_mx:
        st.plotly_chart(trazar_optimizacion(a['cuadros_mx'], a['ganadores_mx'], 'México · MXN', a['fr_riesgo']), width='stretch')
        mejor_mx = a['ganadores_mx'].loc[a['ganadores_mx']['Sharpe máximo'].idxmax()]
        retorno_mx = float(mejor_mx['Retorno máximo Sharpe'])
        st.metric('Rendimiento esperado de la cartera riesgosa · México', f'{(1+retorno_mx)**12-1:.2%}')
        st.metric('Tasa libre México · promedio sorteado', f'{a["rf_mx"].mean():.2%}')
        st.caption(f'Estimación basada en rendimientos históricos: {retorno_mx:.2%} mensual, en MXN.')
        st.dataframe(pesos_tabla.loc[pesos_tabla['País']=='México'].style.format(
            {'Peso en cartera riesgosa':'{:.1%}', 'Peso en capital del país':'{:.1%}',
             'Peso en capital global':'{:.1%}'}, na_rep='—'), hide_index=True, width='stretch')
    st.caption('Los pesos de cada activo aparecen dentro de la cartera riesgosa, dentro del capital de su país y respecto al capital global inicial. La fila de tasa libre de riesgo completa el 100% de cada país.')
    st.caption('Cada cartera riesgosa se optimiza en su propia moneda. Las líneas son de asignación de capital (CAL); no se les denomina CML porque las cinco acciones seleccionadas no representan necesariamente al mercado completo.')

with trayectorias:
    st.subheader('Evolución diaria del capital invertido')
    st.write('Pulsa **Reproducir** o mueve el control de sesión. Cada escenario muestra una ruta diaria aleatoria del bootstrap; el promedio de muchas rutas se ve más suave. EUA y México excluyen la hija para que los tres saldos sumen el total.')
    vista = st.radio('Trayectoria de la animación', ['Escenario individual', 'Promedio de escenarios'], horizontal=True)
    escenario_elegido = (st.number_input('Número de escenario', min_value=1, max_value=a['escenarios'], value=1, step=1) - 1
                         if vista == 'Escenario individual' else None)
    st.plotly_chart(animacion_capital(a['simulacion'], a['fecha_base'], escenario_elegido), width='stretch')
    valores_tabla = (lambda clave: a['simulacion'][clave].mean(axis=0)) if escenario_elegido is None else (lambda clave: a['simulacion'][clave][escenario_elegido])
    tabla_mensual = pd.DataFrame({
        'Sesión': range(253),
        'Fecha aproximada': pd.bdate_range(a['fecha_base'], periods=253),
        'EUA sin hija (MXN)': valores_tabla('eua'),
        'México sin hija (MXN)': valores_tabla('mexico'),
        'Hija (MXN)': valores_tabla('hijo'),
        'Total (MXN)': valores_tabla('capital'),
    })
    with st.expander('Ver cifras de cada sesión'):
        st.dataframe(tabla_mensual.style.format({col:'${:,.0f}' for col in tabla_mensual if col not in ('Sesión', 'Fecha aproximada')}),
                     hide_index=True, width='stretch')
    st.plotly_chart(banda(a['fx'], a['fecha_base'], 'USD/MXN proyectado · percentiles 5, 50 y 95', 'MXN por USD'), width='stretch')
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(banda(a['simulacion']['capital_sin_fx'], a['fecha_base'], 'Capital con el dólar congelado', 'MXN'), width='stretch')
    with c2:
        st.plotly_chart(banda(a['simulacion']['hijo'], a['fecha_base'], 'Cuenta hija de excedentes', 'MXN'), width='stretch')
    con_fx = a['simulacion']['capital'][:, -1]
    sin_fx = a['simulacion']['capital_sin_fx'][:, -1]
    efecto_fx = float(np.median(con_fx-sin_fx))
    fx1,fx2,fx3 = st.columns(3)
    fx1.metric('USD/MXN inicial', f'${a["spot"]:.4f}')
    fx2.metric('USD/MXN final mediano', f'${np.median(a["fx"][:, -1]):.4f}')
    fx3.metric('Efecto cambiario mediano al cierre',
               f'{"−" if efecto_fx < 0 else "+"}${abs(efecto_fx):,.0f} MXN')
    st.caption('Para cada escenario se calcula: capital final con el dólar simulado menos capital final de ese mismo escenario usando el dólar inicial. El indicador es la mediana de esas diferencias, no la diferencia entre dos medianas. Un valor negativo significa que el tipo de cambio redujo el valor en pesos frente al supuesto de dólar constante; no significa que el portafolio completo haya perdido esa cantidad.')
    st.caption('Las bandas se calculan con todas las rutas, no con una muestra de líneas. Las fechas son días hábiles aproximados; pueden incluir festivos bursátiles. Si se activa la cuenta hija, el excedente sobre el umbral mensual se traspasa al cierre de cada bloque de 21 sesiones.')

with modelo:
    p = a['parametros_fx']
    c1,c2,c3 = st.columns(3)
    c1.metric('Último USD/MXN observado', f'${a["spot"]:.4f}')
    c2.metric('Fecha del ajuste', p.fecha_ajuste)
    c3.metric('Persistencia GARCH', f'{p.alpha+p.beta:.4f}')
    st.write('Se ajusta ARMA(2,2) a rendimientos logarítmicos diarios y GARCH(1,1) con innovaciones t a sus residuos. Se simulan 252 sesiones diarias de dólar y activos. El ajuste se recalcula con los precios disponibles al ejecutar la app; los parámetros de octubre de 2026 del ejercicio de futuros no se arrastran a una fecha nueva.')
    tabla_parametros = pd.DataFrame({
        'Parámetro': list(vars(p)),
        'Estimación': [f'{v:.6g}' if isinstance(v, float) else str(v)
                       for v in vars(p).values()],
    })
    st.dataframe(tabla_parametros, hide_index=True, width='stretch')
    st.warning('Proyectar un año amplifica el riesgo de especificación. Antes de usar los resultados para decidir una inversión se requiere evaluación fuera de muestra de coberturas, sensibilidad a parámetros y comparación con un modelo de referencia. Activos y dólar se simulan independientemente por diseño.')

with datos:
    st.write(f'Histórico: {len(a["retornos"]):,} meses completos, hasta {a["retornos"].index.max().date()}. Spot USD/MXN hasta {a["fecha_base"].date()}.')
    correl = a['retornos'].corr()
    st.plotly_chart(px.imshow(correl, zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title='Correlación histórica de activos'), width='stretch')
    st.download_button('Descargar resultados en Excel', excel_descargable(a),
                       file_name='portafolio_internacional_prospectivo.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', type='primary')

st.divider()
st.caption('Análisis académico de escenarios; no es una predicción garantizada ni asesoría financiera. No se incluyen impuestos, costos, deslizamientos ni coberturas cambiarias.')
