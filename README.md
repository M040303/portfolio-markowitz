# Portafolio internacional prospectivo

Aplicación Streamlit para construir dos carteras Markowitz (EUA en USD y México en MXN), simularlas durante 12 meses y expresar el resultado conjunto en pesos mediante un modelo independiente de USD/MXN.

## Ejecutar

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud, conecta este repositorio y usa `app.py` como archivo principal. La versión histórica anterior sigue disponible con `streamlit run app_historico.py`.

## Decisiones incorporadas

- Cinco tickers por país. El modo manual acepta símbolos editables. El modo automático busca cinco entre una lista acotada y editable de candidatas grandes de EUA y México; no se presenta como un ranking vivo de capitalización. La app excluye candidatas sin al menos 253 precios diarios, 25 cierres mensuales previos a la valoración y rendimientos diarios coincidentes. Yahoo Finance entrega precios ajustados. Selecciona acciones estadounidenses cotizadas en USD y mexicanas cotizadas en MXN; la app no detecta automáticamente la moneda de un símbolo.
- La selección automática busca un grupo de tres acciones con correlación negativa frente a cada una de otras dos acciones del mismo país: seis pares cruzados negativos, calculados sobre rendimientos diarios. La ventana es editable (252, 504, 756 sesiones o todo el histórico). Entre las combinaciones que cumplen, elige la de menor varianza histórica con pesos iguales. Si ninguna cumple, favorece la que tenga más pares negativos y luego menor varianza, mostrando una advertencia explícita. En modo manual se muestra el mismo diagnóstico sin sustituir los símbolos elegidos.
- Capital inicial en MXN. La distribución EUA/México se estima mediante un segundo Markowitz entre los dos portafolios país completos, ambos expresados en MXN y con pesos no negativos que suman 100%. La asignación resultante se aplica una sola vez al comienzo de la proyección.
- Dentro de cada país, división inicial editable, inicialmente 70% cartera riesgosa / 30% tasa libre de riesgo. Las posiciones se compran una sola vez y no se rebalancean.
- Cada cartera riesgosa se elige mediante corridas Markowitz independientes. En cada corrida se sortea la tasa libre de riesgo anual (fija o triangular), se generan carteras con pesos no negativos que suman 100% y se guardan máximo Sharpe y mínima varianza. Valores iniciales: 1,000 corridas × 5,000 carteras por país; los controles permiten reducirlos si el equipo tiene menos recursos.
- El segundo Markowitz considera los rendimientos diarios históricos de cada cartera completa, incluyendo su proporción de acciones y tasa segura. La parte de EUA se convierte usando el movimiento histórico USD/MXN de la misma fecha. Examina 1,001 pesos EUA/México entre 0% y 100% y elige el máximo Sharpe frente a la tasa mexicana central. La frontera histórica utiliza ponderaciones constantes para medir media y covarianza; la proyección posterior compra una sola vez y deja fluctuar los pesos.
- Tasa EUA identificada como T-Bills y México como CETES, ambas editables. Sus valores predeterminados son **supuestos**, no cotizaciones actualizadas. Si la tasa es triangular, se sortea una tasa anual por escenario y se mantiene durante ese año; para la optimización se sortea una por corrida.
- En la tabla de tasas se muestra, por país, la tasa anual media sorteada y sus percentiles 5 y 95, el capital inicialmente colocado en la parte segura, los intereses esperados en MXN valorados al dólar inicial, el saldo final en MXN y el rendimiento en moneda local. Para EUA también se muestra el rendimiento convertido a MXN, que incorpora el USD/MXN. La cuenta hija usa las mismas tasas, pero se presenta por separado.
- El rendimiento de los diez activos se simula con un bootstrap **diario** conjunto de filas históricas. Así se conserva la dependencia entre activos de una misma sesión, mientras que USD/MXN se simula de manera independiente, como se definió para este proyecto. La optimización Markowitz sigue usando rendimientos mensuales históricos.
- El USD/MXN se descarga como `MXN=X` de Yahoo Finance. Se reestiman ARMA(2,2) y GARCH(1,1) con innovaciones t sobre rendimientos logarítmicos diarios. Se generan 252 sesiones y se valúan las carteras cada día. **No** se reutiliza el estado de octubre de 2026 del modelo de futuros: el ajuste se actualiza con los datos disponibles en la fecha de valoración.
- Si se activa la cuenta hija, al final de cada bloque de 21 sesiones se transfiere desde la parte riesgosa el importe que excede el rendimiento umbral (1.5% por bloque por defecto); desde la sesión siguiente gana la tasa segura de ese país.
- Se muestran bandas 5–95%, mediana, probabilidad simulada de pérdida, VaR global al 95% como porcentaje del capital inicial y ES al 95% en MXN, composición de carteras, puntos de mínima varianza y máximo Sharpe, líneas de asignación de capital y exportación a Excel.
- La tabla resumen separa acciones México/EUA, tasas seguras México/EUA y las dos partes de la hija. Reporta inversión inicial, aportaciones, valor final esperado, ganancia económica, rendimiento, dispersión de resultados y VaR porcentual. Los traspasos desde las acciones a la hija se suman a la ganancia atribuida a la cartera de origen y se restan de la ganancia de la hija, de modo que las ganancias del desglose concilian con la ganancia global. Las tasas de las hijas usan sus aportaciones esperadas como base, ya que parten de cero.
- El tablero muestra los pesos de cada acción dentro de su cartera riesgosa, del capital de su país y del capital global. También muestra el rendimiento histórico esperado de cada cartera riesgosa y el rendimiento prospectivo esperado de EUA, México y el conjunto.
- La animación diaria muestra por defecto una ruta individual irregular del bootstrap; también permite ver el promedio, que suele ser más suave al agregar muchos escenarios. Separa cartera EUA, cartera México y cuenta hija, con el total. Los tres saldos son excluyentes y suman el capital global en las 253 observaciones; el Excel incluye las mismas columnas por sesión.
- Por defecto se generan **10,000 escenarios prospectivos**. Sobre los 10,000 rendimientos globales finales se realizan **10,000 remuestreos bootstrap con reemplazo**, ambos recuentos editables. La pestaña Decisión informa media, varianza, desviación, mediana, cuantiles, probabilidad de pérdida, VaR y ES de los escenarios y el intervalo bootstrap percentil al 95% de cada estadístico. El Excel contiene el resumen y las 10,000 estimaciones remuestreadas. La varianza se expresa en puntos porcentuales al cuadrado.

## Interpretación y límites

La fecha de valoración es el límite de los datos descargados; Markowitz usa sólo meses completos anteriores al mes de valoración y el bootstrap diario usa sesiones disponibles hasta la fecha. El último spot disponible inicia la simulación cambiaria. Las 252 sesiones y los bloques de 21 sesiones por mes son aproximaciones: las fechas mostradas avanzan por días hábiles de lunes a viernes y no excluyen festivos de México ni EUA.

Si la fecha inicial elegida deja menos de 24 rendimientos mensuales antes de la valoración, la opción «Ampliar histórico» (activada por defecto) descarga datos desde tres años antes de la fecha de valoración. La app muestra ambas fechas para distinguir el inicio solicitado del que realmente se usó. Se puede desactivar; en ese caso, el mensaje indica cuántos rendimientos hay y qué activos tienen poco historial. La valoración nunca se mueve hacia adelante para completar datos.

El VaR es el percentil 95 de la **pérdida de capital al cierre**, dividido entre el capital inicial para mostrarlo en porcentaje; el ES es la pérdida media de la cola peor en MXN. No miden necesidades intermedias de liquidez. La conversión cambiaria histórica participa en la optimización EUA/México; en la proyección se simulan activos y dólar de forma independiente, una simplificación que puede subestimar o sobreestimar riesgo si en la realidad se mueven juntos. Los resultados también son sensibles a la elección de activos, la ventana histórica y la calibración GARCH, especialmente a un año. Se recomienda validar coberturas fuera de muestra y comparar con referencias simples antes de tomar decisiones.

El bootstrap del rendimiento global remuestrea **resultados finales ya simulados**, no vuelve a generar 10,000 caminos diarios en cada repetición. Sus intervalos expresan la variación de los estadísticos por tener un número finito de escenarios, condicional a este modelo y estos datos históricos. No son intervalos de predicción del mercado ni incorporan incertidumbre de parámetros, elección de activos o cambios estructurales.

El rendimiento prospectivo esperado es el promedio de los valores finales simulados dividido entre el capital inicial, menos uno. Para EUA y México se incluye la parte de la cuenta hija originada en cada país, aunque en la gráfica animada se separa para no duplicar capital. La hija no tiene capital inicial: su rendimiento mostrado es el saldo final esperado dividido entre las aportaciones esperadas, menos uno, sin anualizar. Las tasas de rendimiento de las dos carteras riesgosas Markowitz se estiman por separado con promedios históricos de sus activos y no son la rentabilidad futura garantizada.

Con tasa triangular, cada corrida Markowitz recibe una tasa anual nueva y cada escenario prospectivo recibe otro sorteo independiente; dentro de ese escenario, la tasa permanece fija los doce meses. EUA y México usan semillas distintas. Con tasa fija, todas las corridas y escenarios reciben el valor indicado. La semilla hace reproducible el conjunto de sorteos.

El efecto cambiario se calcula por escenario como el capital final convertido con el USD/MXN simulado menos ese mismo capital convertido al USD/MXN inicial. El indicador muestra la **mediana de esas diferencias**. Una cifra negativa indica que el cambio de divisa redujo el valor final en MXN respecto al escenario hipotético de dólar constante; no es la pérdida total del portafolio.

No se incluyen costos, impuestos, deslizamiento, rebalanceo ni coberturas. El resultado es académico y no constituye asesoría financiera.

La correlación se calcula con la ventana histórica elegida y puede cambiar de signo en el futuro. Las listas automáticas son subconjuntos manejables de emisoras grandes, no todos los componentes de sus índices ni una actualización automática del orden por capitalización. El universo puede ajustarse desde la barra lateral antes de ejecutar el análisis.

Como referencia para acotar el universo se usan el [S&P 500 Top 50](https://www.spglobal.com/spdji/en/indices/equity/sp-500-top-50-index/) en EUA y el [S&P/BMV IPC](https://www.spglobal.com/spdji/es/indices/equity/sp-bmv-ipc/) en México. La aplicación valida los precios de las candidatas con Yahoo Finance antes de seleccionarlas.

## Estructura

- `app.py`: interfaz y visualizaciones prospectivas.
- `prospectivo.py`: optimización, calibración cambiaria, simulación y métricas.
- `app_historico.py`, `backtest.py`: proyecto histórico anterior, conservado para consulta.
- `test_prospectivo.py`: pruebas de contabilidad y riesgo con ejemplos controlados.
- `requirements.txt`: dependencias de Streamlit Community Cloud.
