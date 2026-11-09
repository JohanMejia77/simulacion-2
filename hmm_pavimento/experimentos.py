"""
Diseño experimental Monte Carlo (con semillas fijas) para el HMM de deterioro de pavimento.

Cada experimento usa un flujo de números aleatorios independiente derivado de
SeedSequence([semilla_maestra, código_experimento, índice]) -> reproducible bit a bit
y sin solapamiento entre réplicas, escenarios ni experimentos.
"""
from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import metricas as met
from .hmm import (
    HMMDiscreto,
    distancia_variacion_total,
    distribucion_estacionaria,
    distribucion_marginal,
    segundo_valor_propio,
    tiempo_medio_permanencia,
    tiempo_medio_primer_paso,
)
from .modelo import Escenario, construir_A, construir_B, construir_escenario, escenarios, simular_auscultaciones

DEFICIENTES = [2, 3]  # índices de Malo y Crítico

# Códigos de experimento (segunda componente de la semilla)
EXP_ESCENARIOS, EXP_SIGMA, EXP_TAMANO, EXP_PI, EXP_REINICIOS, EXP_REPRO, EXP_GLOBAL_A = 1, 2, 3, 4, 5, 6, 7


def generadores(semilla: int, n: int, *claves: int) -> list[np.random.Generator]:
    ss = np.random.SeedSequence([semilla, *claves])
    return [np.random.default_rng(s) for s in ss.spawn(n)]


def huella(df: pd.DataFrame, decimales: int = 8) -> str:
    """SHA-256 del contenido numérico redondeado (evidencia de reproducibilidad)."""
    return hashlib.sha256(df.round(decimales).to_csv(index=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- #
# Utilidades de Baum-Welch
# ---------------------------------------------------------------------- #
def parametros_iniciales(A_real: np.ndarray, M: int, diag_B0: float):
    """Inicialización estructurada: misma estructura de ceros de A, B diagonal dominante, pi uniforme."""
    K = A_real.shape[0]
    estructura = (A_real > 0).astype(float)
    A0 = estructura / estructura.sum(axis=1, keepdims=True)
    B0 = np.full((K, M), (1 - diag_B0) / (M - 1))
    np.fill_diagonal(B0, diag_B0)
    return A0, B0, np.full(K, 1.0 / K)


def parametros_aleatorios(A_real: np.ndarray, M: int, rng: np.random.Generator):
    """Inicialización aleatoria Dirichlet(1) respetando solo la estructura de ceros de A."""
    K = A_real.shape[0]
    A0 = np.zeros_like(A_real)
    for i in range(K):
        soporte = np.flatnonzero(A_real[i] > 0)
        A0[i, soporte] = rng.dirichlet(np.ones(soporte.size))
    B0 = rng.dirichlet(np.ones(M), size=K)
    return A0, B0, rng.dirichlet(np.ones(K))


def alinear_estados(modelo: HMMDiscreto, B_real: np.ndarray):
    """Resuelve el posible intercambio de etiquetas emparejando filas de B (algoritmo húngaro)."""
    costo = np.abs(modelo.B[:, None, :] - B_real[None, :, :]).sum(axis=2)
    _, col = linear_sum_assignment(costo)
    if np.array_equal(col, np.arange(len(col))):
        return modelo, col, False
    orden = np.argsort(col)
    alineado = HMMDiscreto(modelo.A[np.ix_(orden, orden)], modelo.B[orden], modelo.pi[orden])
    return alineado, col, True


# ---------------------------------------------------------------------- #
# Una réplica Monte Carlo
# ---------------------------------------------------------------------- #
def deteccion_deficientes(X: np.ndarray, X_est: np.ndarray) -> tuple[float, float]:
    """Sensibilidad (recall) y precisión al detectar tramos deficientes (Malo o Crítico)."""
    real, est = np.isin(X, DEFICIENTES), np.isin(X_est, DEFICIENTES)
    vp = np.sum(real & est)
    return float(vp / max(real.sum(), 1)), float(vp / max(est.sum(), 1))


def correr_replica(
    esc: Escenario,
    cfg: dict,
    rng: np.random.Generator,
    N: int,
    T: int,
    ajustar: bool = True,
    B_supuesta: np.ndarray | None = None,
) -> dict:
    hmm_real = esc.hmm()
    X = hmm_real.simular_estados(N, T, rng)
    O, _, _ = simular_auscultaciones(X, cfg["rangos_iri_estado"], cfg["umbrales_iri"], esc.error, rng)
    K = hmm_real.K

    X_vit, _ = hmm_real.viterbi(O)
    gamma = hmm_real.posteriores(O)
    _, _, logver = hmm_real.forward(O)

    fila = {
        "exactitud_ingenua": met.exactitud(X, O),
        "exactitud_viterbi": met.exactitud(X, X_vit),
        "exactitud_mpm": met.exactitud(X, gamma.argmax(axis=2)),
        "kappa_ingenua": met.kappa_cohen(X, O, K),
        "kappa_viterbi": met.kappa_cohen(X, X_vit, K),
        "logver_por_obs": float(logver.sum()) / (N * T),
        "pdef_real_T": float(np.isin(X[:, -1], DEFICIENTES).mean()),
        "pdef_ingenua_T": float(np.isin(O[:, -1], DEFICIENTES).mean()),
        "pdef_hmm_T": float(gamma[:, -1, DEFICIENTES].sum(axis=1).mean()),
    }
    fila["error_pdef_ingenua"] = abs(fila["pdef_ingenua_T"] - fila["pdef_real_T"])
    fila["error_pdef_hmm"] = abs(fila["pdef_hmm_T"] - fila["pdef_real_T"])
    fila["sens_def_ingenua"], fila["prec_def_ingenua"] = deteccion_deficientes(X, O)
    fila["sens_def_viterbi"], fila["prec_def_viterbi"] = deteccion_deficientes(X, X_vit)

    if B_supuesta is not None:  # decodificador que asume (erróneamente) error gaussiano
        X_mal, _ = HMMDiscreto(esc.A, B_supuesta, esc.pi0).viterbi(O)
        fila["exactitud_viterbi_B_gauss"] = met.exactitud(X, X_mal)

    if ajustar:
        bw_cfg = cfg["baum_welch"]
        A0, B0, p0 = parametros_iniciales(esc.A, hmm_real.M, bw_cfg["diag_B0"])
        bw = HMMDiscreto.baum_welch(O, A0, B0, p0, max_iter=bw_cfg["max_iter"], tol=bw_cfg["tol"])
        m, _, permutado = alinear_estados(bw.modelo, esc.B)
        X_bw, _ = m.viterbi(O)
        fila["sens_def_bw"], fila["prec_def_bw"] = deteccion_deficientes(X, X_bw)
        fila.update(
            {
                "exactitud_bw": met.exactitud(X, X_bw),
                "mae_A": met.error_absoluto_medio(m.A, esc.A, esc.A > 0),
                "mae_B": met.error_absoluto_medio(m.B, esc.B),
                "mae_pi": met.error_absoluto_medio(m.pi, esc.pi0),
                "iteraciones_bw": bw.iteraciones,
                "convergio_bw": bw.convergio,
                "permutado_bw": permutado,
                "logver_bw_por_obs": bw.historial_logver[-1] / (N * T),
                "pdef_estacionaria_bw": float(distribucion_estacionaria(m.A)[DEFICIENTES].sum()),
            }
        )
    return fila


def _B_gaussiana_equivalente(cfg: dict, esc: Escenario) -> np.ndarray | None:
    if esc.error["distribucion"] == "normal":
        return None
    return construir_B(cfg["rangos_iri_estado"], cfg["umbrales_iri"], {"distribucion": "normal", "sigma": esc.error["sigma"]})


# ---------------------------------------------------------------------- #
# Experimento 1: escenarios principales
# ---------------------------------------------------------------------- #
def propiedades_analiticas(esc: Escenario, cfg: dict) -> dict:
    A, B = esc.A, esc.B
    pi_est = distribucion_estacionaria(A)
    perm = tiempo_medio_permanencia(A)
    mfpt = tiempo_medio_primer_paso(A, DEFICIENTES)
    lam2 = segundo_valor_propio(A)
    # tiempo de mezcla: primer t con max_i TV(e_i A^t, pi*) <= 0.01
    P, t_mix = np.eye(A.shape[0]), None
    for t in range(1, 500):
        P = P @ A
        if distancia_variacion_total(P, pi_est).max() <= 0.01:
            t_mix = t
            break
    fila = {"escenario": esc.id, "nombre": esc.nombre}
    for j, nombre in enumerate(cfg["estados"]):
        fila[f"pi_est_{nombre}"] = pi_est[j]
    for j, nombre in enumerate(cfg["estados"]):
        fila[f"permanencia_{nombre}"] = perm[j]
    fila.update(
        {
            "pdef_estacionaria": float(pi_est[DEFICIENTES].sum()),
            "mfpt_Bueno_a_deficiente": float(mfpt[0]),
            "lambda2": lam2,
            "t_mezcla_0.01": t_mix,
            "intervenciones_100km_anio": float(100 * (pi_est[2] * A[2, 0] + pi_est[3] * A[3, 0])),
            "diag_B_media": float(np.mean(np.diag(B))),
            "exactitud_ingenua_estacionaria": float(pi_est @ np.diag(B)),
        }
    )
    return fila


def experimento_escenarios(cfg: dict, R: int | None = None, N: int | None = None, T: int | None = None,
                           semilla: int | None = None, ajustar: bool = True, solo: list[str] | None = None):
    sim = cfg["simulacion"]
    R, N, T = R or sim["R_replicas"], N or sim["N_segmentos"], T or sim["T_anios"]
    semilla = cfg["semilla_maestra"] if semilla is None else semilla
    filas, analitico = [], []
    for idx, esc in enumerate(escenarios(cfg)):
        if solo and esc.id not in solo:
            continue
        analitico.append(propiedades_analiticas(esc, cfg))
        B_g = _B_gaussiana_equivalente(cfg, esc)
        t0 = time.perf_counter()
        for r, rng in enumerate(generadores(semilla, R, EXP_ESCENARIOS, idx)):
            fila = correr_replica(esc, cfg, rng, N, T, ajustar=ajustar, B_supuesta=B_g)
            fila.update({"escenario": esc.id, "replica": r})
            filas.append(fila)
        print(f"  {esc.id} ({esc.nombre}): {R} réplicas en {time.perf_counter() - t0:.1f} s", flush=True)
    df = pd.DataFrame(filas)
    columnas = ["escenario", "replica"] + [c for c in df.columns if c not in ("escenario", "replica")]
    return df[columnas], pd.DataFrame(analitico)


# ---------------------------------------------------------------------- #
# Experimento 2: sensibilidad al ruido de medición (matriz B)
# ---------------------------------------------------------------------- #
def experimento_barrido_sigma(cfg: dict) -> pd.DataFrame:
    s = cfg["sensibilidad"]
    N, T = cfg["simulacion"]["N_segmentos"], cfg["simulacion"]["T_anios"]
    filas = []
    for i, sigma in enumerate(s["sigmas"]):
        base = copy.deepcopy(cfg["escenarios"][0])
        base.update({"id": f"sigma={sigma:.2f}", "error": {"distribucion": "normal", "sigma": sigma}})
        esc = construir_escenario(cfg, base)
        for r, rng in enumerate(generadores(cfg["semilla_maestra"], s["R_sigma"], EXP_SIGMA, i)):
            fila = correr_replica(esc, cfg, rng, N, T, ajustar=True)
            fila.update({"sigma": sigma, "replica": r})
            filas.append(fila)
        print(f"  sigma = {sigma:.2f} listo", flush=True)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------- #
# Experimento 3: tamaño de muestra vs. error de estimación (Baum-Welch)
# ---------------------------------------------------------------------- #
def experimento_tamano_muestra(cfg: dict, ids: tuple[str, ...] = ("E1", "E4")) -> pd.DataFrame:
    s = cfg["sensibilidad"]
    T = cfg["simulacion"]["T_anios"]
    filas = []
    for idx, esc in enumerate(escenarios(cfg)):
        if esc.id not in ids:
            continue
        for j, N in enumerate(s["tamanos_N"]):
            for r, rng in enumerate(generadores(cfg["semilla_maestra"], s["R_tamano"], EXP_TAMANO, idx, j)):
                fila = correr_replica(esc, cfg, rng, N, T, ajustar=True)
                fila.update({"escenario": esc.id, "N": N, "replica": r})
                filas.append(fila)
        print(f"  curva de aprendizaje {esc.id} lista", flush=True)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------- #
# Experimento 4: sensibilidad a la distribución inicial pi
# ---------------------------------------------------------------------- #
def experimento_pi(cfg: dict):
    s = cfg["sensibilidad"]
    H, R, N = s["horizonte_pi"], s["R_pi"], cfg["simulacion"]["N_segmentos"]
    esc = escenarios(cfg)[0]
    pi_est = distribucion_estacionaria(esc.A)
    lam2 = segundo_valor_propio(esc.A)
    analitico, simulado = [], []
    for j, (nombre, pi0) in enumerate(cfg["distribuciones_iniciales"].items()):
        pi0 = np.asarray(pi0, dtype=float)
        marg = distribucion_marginal(pi0, esc.A, H)
        tv = distancia_variacion_total(marg, pi_est)
        for t in range(H + 1):
            analitico.append({"pi0": nombre, "anio": t, "pdef": marg[t, DEFICIENTES].sum(), "tv": tv[t],
                              "cota_lambda2": lam2**t})
        hmm = HMMDiscreto(esc.A, esc.B, pi0)
        props = np.empty((R, H + 1))
        for r, rng in enumerate(generadores(cfg["semilla_maestra"], R, EXP_PI, j)):
            X = hmm.simular_estados(N, H + 1, rng)
            props[r] = np.isin(X, DEFICIENTES).mean(axis=0)
        for t in range(H + 1):
            res = met.resumen(props[:, t])
            simulado.append({"pi0": nombre, "anio": t, "media": res["media"], "ic_inf": res["ic_inf"],
                             "ic_sup": res["ic_sup"], "de": res["de"]})
    return pd.DataFrame(analitico), pd.DataFrame(simulado), pi_est, lam2


# ---------------------------------------------------------------------- #
# Experimento 5: sensibilidad de la matriz de transición A
# ---------------------------------------------------------------------- #
PARAMETROS_A = ["d_BR", "d_BM", "d_RM", "d_RC", "d_MC", "r_C"]


def _salidas_A(p: dict) -> tuple[float, float]:
    A = construir_A(p)
    return float(distribucion_estacionaria(A)[DEFICIENTES].sum()), float(tiempo_medio_primer_paso(A, DEFICIENTES)[0])


def experimento_sensibilidad_A(cfg: dict):
    s = cfg["sensibilidad"]
    base = cfg["transicion_base"]
    delta = s["variacion_relativa_A"]
    y0, m0 = _salidas_A(base)
    oat = []
    for nombre in PARAMETROS_A:
        fila = {"parametro": nombre, "valor_base": base[nombre]}
        for etiqueta, factor in (("menos", 1 - delta), ("mas", 1 + delta)):
            p = dict(base)
            p[nombre] = base[nombre] * factor
            y, m = _salidas_A(p)
            fila[f"pdef_{etiqueta}"] = y
            fila[f"mfpt_{etiqueta}"] = m
        fila["rango_pdef_pp"] = 100 * abs(fila["pdef_mas"] - fila["pdef_menos"])
        oat.append(fila)
    oat = pd.DataFrame(oat).sort_values("rango_pdef_pp", ascending=False)

    # Sensibilidad global: muestreo simultáneo uniforme en +/- delta y coeficientes SRC
    rng = generadores(cfg["semilla_maestra"], 1, EXP_GLOBAL_A)[0]
    n = s["muestras_globales_A"]
    factores = rng.uniform(1 - delta, 1 + delta, size=(n, len(PARAMETROS_A)))
    y = np.empty(n)
    mf = np.empty(n)
    for i in range(n):
        p = dict(base)
        for k, nombre in enumerate(PARAMETROS_A):
            p[nombre] = base[nombre] * factores[i, k]
        y[i], mf[i] = _salidas_A(p)
    src_pdef, r2_pdef = met.regresion_estandarizada(factores, y, PARAMETROS_A)
    src_mfpt, r2_mfpt = met.regresion_estandarizada(factores, mf, PARAMETROS_A)
    glob = src_pdef.rename(columns={"SRC": "SRC_pdef"}).merge(src_mfpt.rename(columns={"SRC": "SRC_mfpt"}), on="parametro")
    resumen_global = {
        "pdef_base": y0, "mfpt_base": m0, "R2_pdef": r2_pdef, "R2_mfpt": r2_mfpt,
        "pdef_global": met.resumen(y), "mfpt_global": met.resumen(mf),
        "pdef_p2.5": float(np.percentile(y, 2.5)), "pdef_p97.5": float(np.percentile(y, 97.5)),
        "mfpt_p2.5": float(np.percentile(mf, 2.5)), "mfpt_p97.5": float(np.percentile(mf, 97.5)),
    }
    return oat, glob, resumen_global


# ---------------------------------------------------------------------- #
# Experimento 6: sensibilidad de Baum-Welch a la inicialización
# ---------------------------------------------------------------------- #
def experimento_reinicios(cfg: dict, id_escenario: str = "E4"):
    s = cfg["sensibilidad"]
    N, T = cfg["simulacion"]["N_segmentos"], cfg["simulacion"]["T_anios"]
    esc = next(e for e in escenarios(cfg) if e.id == id_escenario)
    hmm_real = esc.hmm()
    rng_datos = generadores(cfg["semilla_maestra"], 1, EXP_REINICIOS, 0)[0]
    X = hmm_real.simular_estados(N, T, rng_datos)
    O, _, _ = simular_auscultaciones(X, cfg["rangos_iri_estado"], cfg["umbrales_iri"], esc.error, rng_datos)
    logver_real = hmm_real.log_verosimilitud(O)
    bw_cfg = cfg["baum_welch"]
    inicios = [("estructurada", parametros_iniciales(esc.A, hmm_real.M, bw_cfg["diag_B0"]))]
    for k, rng in enumerate(generadores(cfg["semilla_maestra"], s["reinicios_bw"], EXP_REINICIOS, 1)):
        inicios.append((f"aleatoria_{k + 1:02d}", parametros_aleatorios(esc.A, hmm_real.M, rng)))
    filas, curvas = [], []
    for etiqueta, (A0, B0, p0) in inicios:
        bw = HMMDiscreto.baum_welch(O, A0, B0, p0, max_iter=bw_cfg["max_iter"], tol=bw_cfg["tol"])
        m, _, permutado = alinear_estados(bw.modelo, esc.B)
        X_bw, _ = m.viterbi(O)
        filas.append({
            "inicializacion": etiqueta, "logver_final": bw.historial_logver[-1], "iteraciones": bw.iteraciones,
            "convergio": bw.convergio, "permutado": permutado, "exactitud_bw": met.exactitud(X, X_bw),
            "mae_A": met.error_absoluto_medio(m.A, esc.A, esc.A > 0), "mae_B": met.error_absoluto_medio(m.B, esc.B),
        })
        for it, lv in enumerate(bw.historial_logver):
            curvas.append({"inicializacion": etiqueta, "iteracion": it, "logver": lv})
    return pd.DataFrame(filas), pd.DataFrame(curvas), logver_real


# ---------------------------------------------------------------------- #
# Experimento 6b: "mejoras" espurias que vería una cadena visible ajustada al IRI medido
# ---------------------------------------------------------------------- #
def experimento_transiciones_espurias(cfg: dict, R: int = 30) -> pd.DataFrame:
    """
    Proporción de transiciones anuales observadas que muestran mejora (banda menor) sin que haya
    ocurrido una mejora real (rehabilitación). Una cadena de Markov visible estimada con el IRI
    medido absorbería ese ruido como transiciones de mejora inexistentes.
    """
    N, T = cfg["simulacion"]["N_segmentos"], cfg["simulacion"]["T_anios"]
    filas = []
    for idx, esc in enumerate(escenarios(cfg)):
        hmm = esc.hmm()
        for r, rng in enumerate(generadores(cfg["semilla_maestra"], R, EXP_REPRO + 10, idx)):
            X = hmm.simular_estados(N, T, rng)
            O, _, _ = simular_auscultaciones(X, cfg["rangos_iri_estado"], cfg["umbrales_iri"], esc.error, rng)
            mejora_obs, mejora_real = O[:, 1:] < O[:, :-1], X[:, 1:] < X[:, :-1]
            filas.append({
                "escenario": esc.id, "replica": r,
                "tasa_mejora_observada": float(mejora_obs.mean()),
                "tasa_mejora_real": float(mejora_real.mean()),
                "fraccion_espurias": float((mejora_obs & ~mejora_real).sum() / max(mejora_obs.sum(), 1)),
            })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------- #
# Experimento 7: reproducibilidad
# ---------------------------------------------------------------------- #
def experimento_reproducibilidad(cfg: dict, R: int = 10) -> dict:
    """Ejecuta E1 dos veces con la misma semilla y una vez con otra semilla."""
    corrida_1, _ = experimento_escenarios(cfg, R=R, solo=["E1"])
    corrida_2, _ = experimento_escenarios(cfg, R=R, solo=["E1"])
    otra, _ = experimento_escenarios(cfg, R=R, solo=["E1"], semilla=cfg["semilla_maestra"] + 1)
    h1, h2, h3 = huella(corrida_1), huella(corrida_2), huella(otra)
    comparacion = {}
    for m in ("exactitud_viterbi", "exactitud_bw", "exactitud_ingenua"):
        a, b = met.resumen(corrida_1[m]), met.resumen(otra[m])
        comparacion[m] = {"semilla_base": a, "semilla_alternativa": b,
                          "ic_se_solapan": bool(a["ic_inf"] <= b["ic_sup"] and b["ic_inf"] <= a["ic_sup"])}
    return {
        "R": R,
        "sha256_corrida_1": h1,
        "sha256_corrida_2": h2,
        "identicas_misma_semilla": h1 == h2,
        "sha256_semilla_alternativa": h3,
        "difiere_con_otra_semilla": h1 != h3,
        "comparacion_estadistica": comparacion,
    }
