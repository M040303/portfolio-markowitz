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
    ajustar_fx_arma_garch, estadisticas_riesgo, optimizar_markowitz,
    preparar_rendimientos, rendimientos_esperados, simular_fx,
    simular_portafolio, sortear_tasa,
)

st.set_page_config(page_title='Portafolio internacional | México y EUA', page_icon='🌎', layout='wide')
st.markdown('<style>.block-container{padding-top:1.4rem;padding-bottom:3rem}</style>', unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_precios(tickers: tuple[str, ...], inicio: date, fin: date) -> pd.DataFrame:
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
    if faltan:
        raise ValueError('Sin precios para: ' + ', '.join(faltan))
    return precios[list(tickers)].sort_index().ffill(limit=5)


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
    fechas = [inicio + pd.DateOffset(months=i) for i in range(resultados.shape[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas, y=q[2], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fechas, y=q[0], mode='lines', fill='tonexty',
                             fillcolor='rgba(41,105,163,.18)', line=dict(width=0), name='Banda 5–95%'))
    fig.add_trace(go.Scatter(x=fechas, y=q[1], mode='lines+markers',
                             line=dict(color='#174A77', width=3), name='Mediana'))
    fig.update_layout(title=nombre, xaxis_title='Mes proyectado', yaxis_title=eje,
                      height=430, hovermode='x unified')
    return fig


def animacion_capital(simulacion: dict[str, np.ndarray], inicio: pd.Timestamp) -> go.Figure:
    """Capital medio por saldo: los tres componentes suman el total en cada mes."""
    fechas = [inicio + pd.DateOffset(months=i) for i in range(13)]
    series = {
        'EUA · sin hija': simulacion['eua'].mean(axis=0),
        'México · sin hija': simulacion['mexico'].mean(axis=0),
        'Hija · ambas monedas': simulacion['hijo'].mean(axis=0),
        'Capital total': simulacion['capital'].mean(axis=0),
    }
    colores = {'EUA · sin hija': '#2874A6', 'México · sin hija': '#16A085',
               'Hija · ambas monedas': '#D68910', 'Capital total': '#243447'}
    dominios = [(0, .24), (.26, .49), (.51, .74), (.76, 1)]

    def trazas(mes: int) -> list[go.BaseTraceType]:
        lineas = [go.Scatter(x=fechas[:mes+1], y=valores[:mes+1],
                            mode='lines+markers', name=nombre,
                            line=dict(color=colores[nombre], width=4 if nombre == 'Capital total' else 2.5),
                            marker=dict(size=8),
                            hovertemplate='%{x|%b %Y}<br>$%{y:,.0f} MXN<extra>'+nombre+'</extra>')
                  for nombre, valores in series.items()]
        indicadores = [go.Indicator(mode='number', value=float(valores[mes]),
                                   title=dict(text=nombre, font=dict(size=12)),
                                   number=dict(prefix='$', valueformat=',.0f',
                                               font=dict(size=21, color=colores[nombre])),
                                   domain=dict(x=list(dominios[k]), y=[.79, .98]))
                       for k, (nombre, valores) in enumerate(series.items())]
        return [*lineas, *indicadores]

    frames = [go.Frame(name=str(i), data=trazas(i),
                       layout=go.Layout(title=f'Proyección del capital · mes {i} de 12'))
              for i in range(13)]
    fig = go.Figure(data=trazas(0), frames=frames)
    maximo = float(max(np.max(v) for v in series.values()))
    fig.update_layout(title='Proyección del capital · mes 0 de 12', height=690,
                      xaxis=dict(title='Mes proyectado', domain=[0, 1],
                                 range=[fechas[0]-pd.Timedelta(days=8), fechas[-1]+pd.Timedelta(days=8)],
                                 tickformat='%b %Y'),
                      yaxis=dict(title='Capital promedio (MXN)', domain=[0, .71],
                                 range=[-maximo*.04, maximo*1.12], tickformat='$,.0f'),
                      legend=dict(orientation='h', x=0, y=.76),
                      margin=dict(t=75, b=100, l=80), hovermode='x unified',
                      updatemenus=[dict(type='buttons', x=0, y=-.13, buttons=[
                          dict(label='▶ Reproducir', method='animate',
                               args=[None, dict(frame=dict(duration=850, redraw=True),
                                                transition=dict(duration=200), fromcurrent=True)]),
                          dict(label='⏸ Pausar', method='animate',
                               args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
                      ])],
                      sliders=[dict(x=.22, y=-.13, len=.78,
                                    currentvalue=dict(prefix='Mes: '),
                                    steps=[dict(label=str(i), method='animate',
                                                args=[[str(i)], dict(mode='immediate',
                                                                     frame=dict(duration=0, redraw=True))])
                                           for i in range(13)])])
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
        'Capital final sin cambio FX': simul['capital_sin_fx'][:, -1],
        'Cuenta hija final MXN': simul['hijo'][:, -1],
        'USD/MXN final': analisis['fx'][:, -1], 'Rf EUA anual': analisis['rf_us'],
        'Rf México anual': analisis['rf_mx']})
    resumen = pd.DataFrame([analisis['riesgo']]).T.rename(columns={0:'Valor'})
    trayectorias = pd.DataFrame({
        'Mes': np.arange(13), 'Capital p05': np.quantile(simul['capital'], .05, axis=0),
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
        escenarios.to_excel(writer, sheet_name='Escenarios', index=False)
        trayectorias.to_excel(writer, sheet_name='Bandas_mensuales', index=False)
        analisis['ganadores_us'].to_excel(writer, sheet_name='Markowitz_EUA', index=False)
        analisis['ganadores_mx'].to_excel(writer, sheet_name='Markowitz_Mexico', index=False)
        parametros.to_excel(writer, sheet_name='Modelo_USDMXN')
        analisis['retornos'].to_excel(writer, sheet_name='Retornos_historicos')
    return b.getvalue()


st.title('🌎 Portafolio internacional prospectivo')
st.write('Dos carteras Markowitz, una en USD y otra en MXN. El dólar se proyecta por separado y el resultado conjunto se expresa en pesos.')
st.caption('La asignación 60/40 y el reparto 70/30 se hacen **una sola vez al inicio**; después las posiciones conservan sus unidades. No hay comparación con resultados futuros todavía no observados.')

with st.sidebar:
    st.header('Configurar escenario')
    with st.form('configuracion'):
        st.subheader('Activos')
        t_us = [st.text_input(f'EUA · activo {i+1}', value=t, key=f'us_{i}') for i,t in enumerate(['MSFT','AAPL','NVDA','JNJ','XOM'])]
        t_mx = [st.text_input(f'México · activo {i+1}', value=t, key=f'mx_{i}') for i,t in enumerate(['WALMEX.MX','GFNORTEO.MX','FEMSAUBD.MX','AMXB.MX','GMEXICOB.MX'])]
        st.subheader('Horizonte y capital')
        inicio = st.date_input('Inicio del histórico', value=date(2018,1,1))
        corte = st.date_input('Fecha de valoración', value=date.today())
        capital = st.number_input('Capital inicial (MXN)', min_value=1_000., value=1_000_000., step=50_000.)
        fr_us = st.slider('Asignación inicial a EUA (%)', 0, 100, 60, 5) / 100
        fr_riesgo = st.slider('Activos con riesgo dentro de cada país (%)', 0, 100, 70, 5) / 100
        st.subheader('Tasas libres de riesgo')
        st.caption('Entradas editables, expresadas como tasas anuales efectivas. Son supuestos; verifica CETES y T-Bills antes de tomar decisiones.')
        rf_us_par = tasa_controls('T-Bills / EUA', 'rf_us', 4.)
        rf_mx_par = tasa_controls('CETES / México', 'rf_mx', 8.)
        st.subheader('Simulación')
        corridas = st.number_input('Corridas Markowitz por país', 10, 1_000, 1_000, step=10)
        carteras = st.number_input('Carteras por corrida', 100, 5_000, 5_000, step=100)
        escenarios = st.number_input('Escenarios prospectivos', 100, 20_000, 5_000, step=100)
        semilla = st.number_input('Semilla', 0, 2_147_483_647, 20261119)
        aplicar_hijo = st.checkbox('Trasladar excedente mensual a cartera hija segura', value=True)
        umbral = st.number_input('Umbral mensual (%)', min_value=0., max_value=99., value=1.5, step=.1,
                                 disabled=not aplicar_hijo) / 100
        ejecutar = st.form_submit_button('Generar análisis', type='primary', width='stretch')

if ejecutar:
    st.session_state.pop('prospectivo', None)
    try:
        us = tuple(t.strip().upper() for t in t_us)
        mx = tuple(t.strip().upper() for t in t_mx)
        todos = us + mx
        if any(not t for t in todos) or len(set(todos)) != len(todos):
            raise ValueError('Completa diez símbolos distintos: cinco por país.')
        if not inicio < corte <= date.today():
            raise ValueError('El histórico debe iniciar antes de la valoración y ésta no puede ser futura.')
        if not (rf_us_par[0] <= rf_us_par[1] <= rf_us_par[2] and rf_mx_par[0] <= rf_mx_par[1] <= rf_mx_par[2]):
            raise ValueError('En cada tasa debe cumplirse mínimo ≤ moda ≤ máximo.')
        with st.spinner('Descargando históricos y estimando el modelo USD/MXN…'):
            precios = descargar_precios(todos, inicio, corte)
            fx_historico = descargar_dolar(inicio, corte)
            retornos = preparar_rendimientos(precios, pd.Timestamp(corte))
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
            fx_diario = simular_fx(s0, p_fx, int(escenarios), 252, int(semilla)+5)
            fx = fx_diario[:, np.rint(np.linspace(0, 252, 13)).astype(int)]
            rf_us = sortear_tasa(int(escenarios), *rf_us_par, int(semilla)+6)
            rf_mx = sortear_tasa(int(escenarios), *rf_mx_par, int(semilla)+7)
            simul = simular_portafolio(retornos, pesos_us['sharpe'], pesos_mx['sharpe'],
                                      list(us), list(mx), fx, rf_us, rf_mx,
                                      float(capital), fr_us, fr_riesgo, int(semilla)+8,
                                      umbral if aplicar_hijo else None)
            riesgo = estadisticas_riesgo(simul['capital'][:, -1], float(capital))
            esperados = rendimientos_esperados(simul, float(capital), fr_us)
        st.session_state['prospectivo'] = dict(us=us, mx=mx, retornos=retornos,
            ganadores_us=gan_us, ganadores_mx=gan_mx, pesos_us=pesos_us,
            pesos_mx=pesos_mx, cuadros_us=cuadros_us, cuadros_mx=cuadros_mx,
            parametros_fx=p_fx, spot=s0, fecha_base=fecha_base, fx=fx,
            rf_us=rf_us, rf_mx=rf_mx, simulacion=simul, riesgo=riesgo,
            esperados=esperados,
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
if a['parametros_fx'].alpha + a['parametros_fx'].beta >= .995:
    st.warning('El GARCH estimado tiene persistencia muy alta. A un año, las bandas pueden ser especialmente sensibles a los parámetros y a los choques extremos.')

resumen, markowitz, trayectorias, modelo, datos = st.tabs(['Decisión', 'Markowitz y línea de capital', 'Escenarios a 12 meses', 'Modelo del dólar', 'Datos y descarga'])
with resumen:
    st.subheader('Rendimientos esperados a 12 meses')
    t1,t2,t3 = st.columns(3)
    t1.metric('EUA · inversión inicial en MXN',
              f'{e["eua_retorno"]:.2%}' if e['eua_retorno'] is not None else 'Sin asignación')
    t2.metric('México · inversión inicial en MXN',
              f'{e["mexico_retorno"]:.2%}' if e['mexico_retorno'] is not None else 'Sin asignación')
    t3.metric('Global · inversión inicial en MXN', f'{e["global_retorno"]:.2%}')
    st.dataframe(tabla_rendimientos(a).style.format(
        {'Base (MXN)': '${:,.0f}', 'Final esperado (MXN)': '${:,.0f}',
         'Rendimiento esperado': '{:.2%}'}, na_rep='—'),
        hide_index=True, width='stretch')
    st.caption('EUA y México incluyen los excedentes originados en cada país para calcular su rendimiento sobre la inversión inicial. La hija se muestra aparte: inicia en $0 y su rendimiento se calcula sobre sus aportaciones acumuladas, sin anualizar. Los valores esperados son promedios de los escenarios simulados; EUA ya incluye la conversión USD/MXN.')
    st.subheader('Riesgo global')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Capital mediano final', f'${r["mediana"]:,.0f} MXN')
    c2.metric('Prob. de terminar bajo el capital inicial', f'{r["prob_perdida"]:.1%}')
    c3.metric('VaR 95% · pérdida en MXN', f'${r["var95"]:,.0f}')
    c4.metric('ES 95% · pérdida media de cola', f'${r["es95"]:,.0f}')
    st.caption(f'Capital inicial ${a["capital"]:,.0f} MXN · EUA {a["fr_us"]:.0%} · México {1-a["fr_us"]:.0%} · dentro de cada país: activos {a["fr_riesgo"]:.0%} / tasa {1-a["fr_riesgo"]:.0%}.')
    st.plotly_chart(banda(a['simulacion']['capital'], a['fecha_base'], 'Capital total expresado en MXN', 'MXN'), width='stretch')
    st.write('**Rango final por escenarios**')
    st.dataframe(pd.DataFrame([{'Percentil 5':r['p05'], 'Mediana':r['mediana'], 'Media':r['media'], 'Percentil 95':r['p95']}]).style.format('${:,.0f}'), hide_index=True, width='stretch')
    st.caption('El VaR mide el percentil 95 de la pérdida respecto del capital inicial; el ES es la pérdida media del 5% peor. Ambas cifras dependen del modelo y son pérdidas al cierre, no necesidades intermedias de liquidez.')

with markowitz:
    st.write('Los puntos rojos conservan el máximo Sharpe encontrado en corridas previas; el rombo identifica la mínima varianza de la corrida y el punto naranja muestra la asignación entre activos y tasa que elegiste. La tasa cambia entre corridas conforme a su distribución configurada.')
    pesos_tabla = tabla_pesos(a)
    lado_us, lado_mx = st.columns(2)
    with lado_us:
        st.plotly_chart(trazar_optimizacion(a['cuadros_us'], a['ganadores_us'], 'EUA · USD', a['fr_riesgo']), width='stretch')
        mejor_us = a['ganadores_us'].loc[a['ganadores_us']['Sharpe máximo'].idxmax()]
        retorno_us = float(mejor_us['Retorno máximo Sharpe'])
        st.metric('Rendimiento esperado de la cartera riesgosa · EUA', f'{(1+retorno_us)**12-1:.2%}')
        st.caption(f'Estimación basada en rendimientos históricos: {retorno_us:.2%} mensual, en USD.')
        st.dataframe(pesos_tabla.loc[pesos_tabla['País']=='EUA'].style.format(
            {'Peso en cartera riesgosa':'{:.1%}', 'Peso en capital del país':'{:.1%}',
             'Peso en capital global':'{:.1%}'}, na_rep='—'), hide_index=True, width='stretch')
    with lado_mx:
        st.plotly_chart(trazar_optimizacion(a['cuadros_mx'], a['ganadores_mx'], 'México · MXN', a['fr_riesgo']), width='stretch')
        mejor_mx = a['ganadores_mx'].loc[a['ganadores_mx']['Sharpe máximo'].idxmax()]
        retorno_mx = float(mejor_mx['Retorno máximo Sharpe'])
        st.metric('Rendimiento esperado de la cartera riesgosa · México', f'{(1+retorno_mx)**12-1:.2%}')
        st.caption(f'Estimación basada en rendimientos históricos: {retorno_mx:.2%} mensual, en MXN.')
        st.dataframe(pesos_tabla.loc[pesos_tabla['País']=='México'].style.format(
            {'Peso en cartera riesgosa':'{:.1%}', 'Peso en capital del país':'{:.1%}',
             'Peso en capital global':'{:.1%}'}, na_rep='—'), hide_index=True, width='stretch')
    st.caption('Los pesos de cada activo aparecen dentro de la cartera riesgosa, dentro del capital de su país y respecto al capital global inicial. La fila de tasa libre de riesgo completa el 100% de cada país.')
    st.caption('Cada cartera riesgosa se optimiza en su propia moneda. Las líneas son de asignación de capital (CAL); no se les denomina CML porque las cinco acciones seleccionadas no representan necesariamente al mercado completo.')

with trayectorias:
    st.subheader('Evolución mensual del capital invertido')
    st.write('Pulsa **Reproducir** o mueve el control de mes. Las líneas muestran el capital promedio de todos los escenarios en MXN; EUA y México excluyen la hija en esta gráfica para que los tres saldos sumen exactamente el total.')
    st.plotly_chart(animacion_capital(a['simulacion'], a['fecha_base']), width='stretch')
    tabla_mensual = pd.DataFrame({
        'Mes': range(13),
        'EUA sin hija (MXN)': a['simulacion']['eua'].mean(axis=0),
        'México sin hija (MXN)': a['simulacion']['mexico'].mean(axis=0),
        'Hija (MXN)': a['simulacion']['hijo'].mean(axis=0),
        'Total (MXN)': a['simulacion']['capital'].mean(axis=0),
    })
    with st.expander('Ver cifras de cada mes'):
        st.dataframe(tabla_mensual.style.format({col:'${:,.0f}' for col in tabla_mensual if col != 'Mes'}),
                     hide_index=True, width='stretch')
    st.plotly_chart(banda(a['fx'], a['fecha_base'], 'USD/MXN proyectado · percentiles 5, 50 y 95', 'MXN por USD'), width='stretch')
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(banda(a['simulacion']['capital_sin_fx'], a['fecha_base'], 'Capital con el dólar congelado', 'MXN'), width='stretch')
    with c2:
        st.plotly_chart(banda(a['simulacion']['hijo'], a['fecha_base'], 'Cuenta hija de excedentes', 'MXN'), width='stretch')
    con_fx = a['simulacion']['capital'][:, -1]
    sin_fx = a['simulacion']['capital_sin_fx'][:, -1]
    st.metric('Contribución cambiaria mediana al cierre', f'${np.median(con_fx-sin_fx):,.0f} MXN')
    st.caption('La contribución cambiaria compara cada escenario con el mismo escenario valuado al dólar inicial. Las bandas se calculan con todas las rutas, no con una muestra de líneas. Si se activa la cuenta hija, el excedente sobre el umbral mensual pasa a la tasa del país correspondiente al cierre de ese mes.')

with modelo:
    p = a['parametros_fx']
    c1,c2,c3 = st.columns(3)
    c1.metric('Último USD/MXN observado', f'${a["spot"]:.4f}')
    c2.metric('Fecha del ajuste', p.fecha_ajuste)
    c3.metric('Persistencia GARCH', f'{p.alpha+p.beta:.4f}')
    st.write('Se ajusta ARMA(2,2) a rendimientos logarítmicos diarios y GARCH(1,1) con innovaciones t a sus residuos. Después se simulan 252 sesiones y se toman 12 cortes mensuales. El ajuste se recalcula con los precios disponibles al ejecutar la app; los parámetros de octubre de 2026 del ejercicio de futuros no se arrastran a una fecha nueva.')
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
