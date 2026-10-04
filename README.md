# Simulador Markowitz en Streamlit

Aplicación interactiva que permite elegir cinco activos de Yahoo Finance y ejecutar simulaciones de portafolios buscando el índice de Sharpe máximo.

## Funciones

- Cinco tickers configurables.
- Precios y rendimientos mensuales descargados con `yfinance`.
- Bootstrap histórico de `^IRX` o tasa libre de riesgo fija.
- Hasta 10,000 corridas y 25,000 portafolios por corrida.
- Pesos no negativos que suman 100%.
- Gráficas interactivas y animación.
- Línea de asignación de capital con el punto 70/30 y el portafolio riesgoso.
- Descarga de resultados en Excel.
- Ajuste de pesos únicamente con datos anteriores al corte.
- Evaluación mensual fuera de muestra y comparación de capital esperado con observado.
- Asignación inicial 70% a la cartera de acciones y 30% a la tasa libre de riesgo.
- Transferencia a una cuenta segura de la ganancia mensual riesgosa que supere
  el umbral configurable (1.5% por defecto). Esa cuenta devenga la tasa desde
  el siguiente intervalo; el Excel incluye el detalle de cada transferencia.

## Ejecutar localmente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

En Windows, activa el entorno con:

```powershell
.venv\Scripts\activate
```

## Publicar en Streamlit Community Cloud

1. Sube `app.py` y `requirements.txt` a un repositorio de GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io/).
3. Selecciona el repositorio y establece `app.py` como archivo principal.
4. Presiona **Deploy**.

## Nota sobre monedas

Los cinco activos deben estar expresados en una moneda comparable. `^IRX` es una referencia de Estados Unidos, por lo que resulta apropiada para activos cotizados en USD. Para América Móvil puede utilizarse `AMX`, su ADR en dólares, en lugar de `AMXL.MX`.

Este proyecto es educativo y no constituye asesoría financiera.

## Cálculo del excedente

Para cada intervalo, si el capital riesgoso al inicio es `P`, su rendimiento
real es `r` y la meta mensual es `h`, el importe transferido al cierre es
`P × max(r − h, 0)`. Es una regla sobre el rendimiento **del periodo**, no
una comparación con la rentabilidad acumulada del año. El 1.5% mensual
compuesto equivale a cerca de 19.56% anual; el equivalente mensual exacto de
20% anual es `(1.20) ** (1 / 12) - 1`, cerca de 1.53%.

La valoración utiliza el primer precio disponible de los cinco activos
desde cada fecha mensual objetivo y la tasa conocida antes del comienzo de
cada intervalo. Los pesos del portafolio riesgoso se restablecen cada mes.
Se omiten costos de transacción, impuestos y cambios de divisa.
