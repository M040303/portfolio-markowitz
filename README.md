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
- En la tabla de tasas se muestra, por país, la tasa anual media sorteada y sus percentiles 5 y 95, el capital inicialmente colocado en la parte segura, los intereses esperados en MXN valorados al dólar inicial, el saldo final en MXN y el rendimiento en moneda local. Para EUA también se muestra el rendimiento convertido a MXN, que incorpora el USD/MXN. La cuenta hija usa las mismas tasas, pero se presenta por separado.
- El rendimiento de los diez activos se simula con un bootstrap mensual conjunto. Así se conserva la correlación histórica entre activos, mientras que USD/MXN se simula de manera independiente, como se definió para este proyecto.
- El USD/MXN se descarga como `MXN=X` de Yahoo Finance. Se reestiman ARMA(2,2) y GARCH(1,1) con innovaciones t sobre rendimientos logarítmicos diarios. Se generan 252 sesiones y se toman 12 puntos equiespaciados como aproximación mensual. **No** se reutiliza el estado de octubre de 2026 del modelo de futuros: el ajuste se actualiza con los datos disponibles en la fecha de valoración.
- Si se activa la cuenta hija, al final de cada mes se transfiere desde la parte riesgosa el importe que excede el rendimiento umbral (1.5% mensual por defecto); a partir del mes siguiente gana la tasa segura de ese país.
- Se muestran bandas 5–95%, mediana, probabilidad simulada de pérdida, VaR y ES al 95% en MXN, composición de carteras, puntos de mínima varianza y máximo Sharpe, líneas de asignación de capital y exportación a Excel.
- El tablero muestra los pesos de cada acción dentro de su cartera riesgosa, del capital de su país y del capital global. También muestra el rendimiento histórico esperado de cada cartera riesgosa y el rendimiento prospectivo esperado de EUA, México y el conjunto.
- La animación mensual separa el capital promedio en cartera EUA, cartera México y cuenta hija, con el total. Los tres saldos son excluyentes y suman el capital global; el Excel incluye las mismas columnas por mes.

## Interpretación y límites

La fecha de valoración es el límite de los datos descargados; sólo se usan meses completos anteriores al mes de valoración para el bootstrap bursátil. El último spot disponible inicia la simulación cambiaria. El horizonte de 252 sesiones y sus 12 cortes son una aproximación al año, no un calendario exacto de días hábiles de México y EUA.

El VaR es el percentil 95 de la **pérdida de capital al cierre** y el ES es la pérdida media de la cola peor. No miden necesidades intermedias de liquidez. La independencia entre activos y dólar fue una decisión de simplificación: puede subestimar o sobreestimar riesgo si en la realidad se mueven juntos. Los resultados también son sensibles a la elección de activos, la ventana histórica y la calibración GARCH, especialmente a un año. Se recomienda validar coberturas fuera de muestra y comparar con referencias simples antes de tomar decisiones.

El rendimiento prospectivo esperado es el promedio de los valores finales simulados dividido entre el capital inicial, menos uno. Para EUA y México se incluye la parte de la cuenta hija originada en cada país, aunque en la gráfica animada se separa para no duplicar capital. La hija no tiene capital inicial: su rendimiento mostrado es el saldo final esperado dividido entre las aportaciones esperadas, menos uno, sin anualizar. Las tasas de rendimiento de las dos carteras riesgosas Markowitz se estiman por separado con promedios históricos de sus activos y no son la rentabilidad futura garantizada.

Con tasa triangular, cada corrida Markowitz recibe una tasa anual nueva y cada escenario prospectivo recibe otro sorteo independiente; dentro de ese escenario, la tasa permanece fija los doce meses. EUA y México usan semillas distintas. Con tasa fija, todas las corridas y escenarios reciben el valor indicado. La semilla hace reproducible el conjunto de sorteos.

El efecto cambiario se calcula por escenario como el capital final convertido con el USD/MXN simulado menos ese mismo capital convertido al USD/MXN inicial. El indicador muestra la **mediana de esas diferencias**. Una cifra negativa indica que el cambio de divisa redujo el valor final en MXN respecto al escenario hipotético de dólar constante; no es la pérdida total del portafolio.

No se incluyen costos, impuestos, deslizamiento, rebalanceo ni coberturas. El resultado es académico y no constituye asesoría financiera.

## Estructura

- `app.py`: interfaz y visualizaciones prospectivas.
- `prospectivo.py`: optimización, calibración cambiaria, simulación y métricas.
- `app_historico.py`, `backtest.py`: proyecto histórico anterior, conservado para consulta.
- `test_prospectivo.py`: pruebas de contabilidad y riesgo con ejemplos controlados.
- `requirements.txt`: dependencias de Streamlit Community Cloud.
