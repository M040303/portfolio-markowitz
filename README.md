# Portafolio internacional prospectivo

Aplicación Streamlit para construir dos carteras Markowitz (EUA en USD y México en MXN), simularlas durante 12 meses y expresar el resultado conjunto en pesos mediante un modelo independiente de USD/MXN.

## Ejecutar

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud, conecta este repositorio y usa `app.py` como archivo principal. La versión histórica anterior sigue disponible con `streamlit run app_historico.py`.

## Decisiones incorporadas

- Cinco tickers editables por país. Yahoo Finance entrega los precios ajustados. Selecciona acciones estadounidenses cotizadas en USD y mexicanas cotizadas en MXN; la app no detecta automáticamente la moneda de un símbolo.
- Capital inicial en MXN y distribución EUA/México editable, inicialmente 60/40.
- Dentro de cada país, división inicial editable, inicialmente 70% cartera riesgosa / 30% tasa libre de riesgo. Las posiciones se compran una sola vez y no se rebalancean.
- Cada cartera riesgosa se elige mediante corridas Markowitz independientes. En cada corrida se sortea la tasa libre de riesgo anual (fija o triangular), se generan carteras con pesos no negativos que suman 100% y se guardan máximo Sharpe y mínima varianza. Valores iniciales: 1,000 corridas × 5,000 carteras por país; los controles permiten reducirlos si el equipo tiene menos recursos.
- Tasa EUA identificada como T-Bills y México como CETES, ambas editables. Sus valores predeterminados son **supuestos**, no cotizaciones actualizadas. Si la tasa es triangular, se sortea una tasa anual por escenario y se mantiene durante ese año; para la optimización se sortea una por corrida.
- El rendimiento de los diez activos se simula con un bootstrap mensual conjunto. Así se conserva la correlación histórica entre activos, mientras que USD/MXN se simula de manera independiente, como se definió para este proyecto.
- El USD/MXN se descarga como `MXN=X` de Yahoo Finance. Se reestiman ARMA(2,2) y GARCH(1,1) con innovaciones t sobre rendimientos logarítmicos diarios. Se generan 252 sesiones y se toman 12 puntos equiespaciados como aproximación mensual. **No** se reutiliza el estado de octubre de 2026 del modelo de futuros: el ajuste se actualiza con los datos disponibles en la fecha de valoración.
- Si se activa la cuenta hija, al final de cada mes se transfiere desde la parte riesgosa el importe que excede el rendimiento umbral (1.5% mensual por defecto); a partir del mes siguiente gana la tasa segura de ese país.
- Se muestran bandas 5–95%, mediana, probabilidad simulada de pérdida, VaR y ES al 95% en MXN, composición de carteras, puntos de mínima varianza y máximo Sharpe, líneas de asignación de capital y exportación a Excel.

## Interpretación y límites

La fecha de valoración es el límite de los datos descargados; sólo se usan meses completos anteriores al mes de valoración para el bootstrap bursátil. El último spot disponible inicia la simulación cambiaria. El horizonte de 252 sesiones y sus 12 cortes son una aproximación al año, no un calendario exacto de días hábiles de México y EUA.

El VaR es el percentil 95 de la **pérdida de capital al cierre** y el ES es la pérdida media de la cola peor. No miden necesidades intermedias de liquidez. La independencia entre activos y dólar fue una decisión de simplificación: puede subestimar o sobreestimar riesgo si en la realidad se mueven juntos. Los resultados también son sensibles a la elección de activos, la ventana histórica y la calibración GARCH, especialmente a un año. Se recomienda validar coberturas fuera de muestra y comparar con referencias simples antes de tomar decisiones.

No se incluyen costos, impuestos, deslizamiento, rebalanceo ni coberturas. El resultado es académico y no constituye asesoría financiera.

## Estructura

- `app.py`: interfaz y visualizaciones prospectivas.
- `prospectivo.py`: optimización, calibración cambiaria, simulación y métricas.
- `app_historico.py`, `backtest.py`: proyecto histórico anterior, conservado para consulta.
- `test_prospectivo.py`: pruebas de contabilidad y riesgo con ejemplos controlados.
- `requirements.txt`: dependencias de Streamlit Community Cloud.
