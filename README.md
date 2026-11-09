# HMM para inferir la condición real del pavimento a partir de mediciones ruidosas del IRI

Código, datos simulados y artículo del proyecto **«Modelo oculto de Markov para inferir la condición real del pavimento a partir de mediciones ruidosas del IRI: formulación, simulación Monte Carlo y análisis de sensibilidad reproducible»**.

Autores: Johan, Juan y Jhojan · Versión 1.0 · Lenguaje: Python 3.12

## Problema y modelo

Las agencias viales clasifican cada tramo según su rugosidad (IRI, m/km), pero el IRI **medido** trae error, sobre todo con sensores de bajo costo. El proyecto modela la condición real como un **modelo oculto de Markov (HMM)** λ = (A, B, π):

| Elemento | Definición |
|---|---|
| Estados ocultos S | Bueno, Regular, Malo, Crítico (IRI real en [0.80, 1.50), [1.50, 2.68), [2.68, 4.00), [4.00, 6.00] m/km) |
| Símbolos V | Banda del IRI **medido**: < 1.50, 1.50–2.68, 2.68–4.00, ≥ 4.00 m/km |
| A (4×4) | Deterioro anual + intervención preventiva (r_M) + rehabilitación (r_C); ver `hmm_pavimento/modelo.py::construir_A` |
| B (4×4) | b_j(k) = P(banda k \| estado j), derivada de la convolución Uniforme(IRI real) * error del sensor (normal en forma cerrada; t de Student por integración numérica) |
| π | Perfil de condición inicial de la red (por defecto, red envejecida: 0.40, 0.30, 0.20, 0.10) |

Los algoritmos Forward, Backward, Viterbi y Baum–Welch están implementados desde cero (`hmm_pavimento/hmm.py`), vectorizados y con escalamiento numérico.

## Estructura del repositorio

```
├── config/escenarios.json        # TODOS los parámetros: A, error de medición, π, semilla, R, N, T, sensibilidad
├── hmm_pavimento/
│   ├── hmm.py                    # clase HMMDiscreto: simular, forward, backward, viterbi, baum_welch + propiedades de la cadena
│   ├── modelo.py                 # especificación del problema: construir_A, construir_B, simulador físico del IRI
│   ├── experimentos.py           # diseño Monte Carlo con semillas fijas (escenarios, sensibilidad, reproducibilidad)
│   ├── metricas.py               # exactitud, kappa, IC 95 %, Wilcoxon, Kruskal–Wallis, Holm, SRC
│   └── figuras.py                # figuras de calidad de publicación (PNG 300 ppp + PDF)
├── tests/test_hmm.py             # 17 pruebas unitarias (fuerza bruta, hmmlearn, EM monótono, reproducibilidad…)
├── ejecutar_todo.py              # una sola orden regenera todos los resultados, tablas y figuras
├── resultados/
│   ├── tablas/                   # CSV/JSON con todas las réplicas y resúmenes
│   ├── figuras/                  # fig1 … fig5, figA1, figB1 (PNG y PDF)
│   ├── entorno.json              # versiones de Python y bibliotecas, semilla, sistema
│   ├── reproducibilidad.json     # huellas SHA-256 de corridas repetidas
│   └── huellas_sha256.json       # huella de cada tabla
├── articulo/
│   ├── articulo.md               # fuente del artículo (cifras enlazadas a resultados/)
│   ├── construir_articulo.py     # genera articulo.docx con ecuaciones nativas de Word
│   ├── articulo.docx             # artículo final (carta, interlineado 1.5, IEEE)
│   └── articulo.pdf              # versión PDF del artículo
├── pitch/guion_pitch.md          # guion del video (3 min, tres integrantes)
├── requirements.txt              # versiones exactas
└── LICENSE
```

## Instalación

Requiere Python 3.12 (probado con 3.12.10 en Windows 11).

```bash
python -m venv .venv
```

Activar el entorno — Windows (PowerShell): `.venv\Scripts\Activate.ps1` · macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python -m pytest -q
```

```bash
python ejecutar_todo.py
```

- `python ejecutar_todo.py` corre los 7 experimentos (≈ 7 min en un portátil) y escribe todo en `resultados/`.
- `python ejecutar_todo.py --rapido` hace una prueba de humo con pocas réplicas (≈ 1 min) en `resultados_rapido/`.
- `python ejecutar_todo.py --solo-figuras` regenera las figuras desde los CSV guardados.

Para regenerar el artículo (requiere Microsoft Office para las ecuaciones nativas):

```bash
pip install -r articulo/requirements-articulo.txt
```

```bash
python articulo/construir_articulo.py
```

## Diseño experimental

| Escenario | Cambio respecto a E1 | Qué evalúa |
|---|---|---|
| E1 Base | Tránsito medio, política correctiva, perfilómetro (σ = 0.15 m/km, error normal) | Referencia |
| E2 Tránsito alto | Probabilidades de deterioro × 1.5 | Efecto de A |
| E3 Política preventiva | 30 % de los tramos Malos intervenidos cada año | Efecto de A |
| E4 Medición de bajo costo | σ = 0.50 m/km, error normal | Efecto de B |
| E5 Error de colas pesadas | σ = 0.50 m/km, error t de Student (ν = 3) | Efecto de la forma de la distribución |

Cada escenario: R = 100 réplicas × N = 200 tramos × T = 25 años. Sensibilidad adicional: σ de 0.05 a 1.00 m/km, N de 25 a 400, tres distribuciones iniciales, ±20 % en los parámetros de A (OAT y global con SRC) y 21 inicializaciones de Baum–Welch.

## Resultados principales

| Escenario | Exactitud lectura directa | Exactitud HMM (λ estimado por Baum–Welch) | Red deficiente estacionaria |
|---|---|---|---|
| E1 Base | 91.9 % | 97.2 % | 38.5 % |
| E2 Tránsito alto | 92.2 % | 96.5 % | 43.2 % |
| E3 Política preventiva | 91.4 % | 96.4 % | 22.6 % |
| E4 Bajo costo | 73.9 % | 88.8 % | 38.5 % |
| E5 Colas pesadas | 80.0 % | 91.5 % | 38.5 % |

Todas las mejoras son significativas (Wilcoxon con corrección de Holm, p < 0.001). Las tablas completas con media, desviación estándar, mínimo, máximo e IC 95 % están en `resultados/tablas/resumen_escenarios.csv`.

## Reproducibilidad

- **Semillas fijas:** semilla maestra 20261107 en `config/escenarios.json`. Cada réplica usa un flujo independiente `PCG64(SeedSequence([semilla, experimento, índice]))`, así que los resultados no dependen del orden de ejecución.
- **Evidencia:** dos corridas con la misma semilla producen la misma huella SHA-256 (`resultados/reproducibilidad.json`); con otra semilla los valores cambian, pero los IC 95 % se solapan.
- **Verificación:** después de ejecutar, comparen `resultados/huellas_sha256.json` con la versión publicada. Las huellas se calculan sobre valores redondeados a 8 decimales; en otro sistema operativo o con otra biblioteca BLAS pueden variar en el último dígito, pero las cifras del artículo no cambian.
- **Entorno:** versiones exactas en `requirements.txt` y en `resultados/entorno.json`.

## Licencia

MIT (ver `LICENSE`).