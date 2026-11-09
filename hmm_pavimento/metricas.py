"""Métricas de desempeño y análisis estadístico de las réplicas Monte Carlo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def exactitud(X: np.ndarray, X_est: np.ndarray) -> float:
    """Proporción de (tramo, año) cuyo estado oculto se recupera correctamente."""
    return float(np.mean(np.asarray(X) == np.asarray(X_est)))


def kappa_cohen(X: np.ndarray, X_est: np.ndarray, K: int) -> float:
    """Concordancia corregida por azar entre estado real y estimado."""
    X, X_est = np.ravel(X), np.ravel(X_est)
    conf = np.zeros((K, K))
    np.add.at(conf, (X, X_est), 1)
    n = conf.sum()
    p_o = np.trace(conf) / n
    p_e = (conf.sum(0) @ conf.sum(1)) / n**2
    return float((p_o - p_e) / (1 - p_e))


def error_absoluto_medio(estimado: np.ndarray, real: np.ndarray, mascara: np.ndarray | None = None) -> float:
    dif = np.abs(np.asarray(estimado) - np.asarray(real))
    if mascara is not None:
        dif = dif[mascara]
    return float(dif.mean())


def resumen(valores, confianza: float = 0.95) -> dict:
    """Media, DE, mínimo, máximo e IC t-Student de la media."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    media = v.mean()
    de = v.std(ddof=1) if n > 1 else np.nan
    if n > 1 and de > 0:
        semi = stats.t.ppf(0.5 + confianza / 2, n - 1) * de / np.sqrt(n)
    else:
        semi = 0.0
    return {
        "n": n,
        "media": media,
        "de": de,
        "min": v.min(),
        "max": v.max(),
        "ic_inf": media - semi,
        "ic_sup": media + semi,
        "semiamplitud": semi,
    }


def tabla_resumen(df: pd.DataFrame, metricas: list[str], por: str = "escenario") -> pd.DataFrame:
    filas = []
    for grupo, sub in df.groupby(por, sort=False):
        for m in metricas:
            if m not in sub or sub[m].isna().all():
                continue  # métrica no definida para este grupo (p. ej., B gaussiana supuesta solo en E5)
            fila = {por: grupo, "metrica": m}
            fila.update(resumen(sub[m]))
            filas.append(fila)
    return pd.DataFrame(filas)


def holm(pvalores) -> np.ndarray:
    """Corrección de Holm-Bonferroni para comparaciones múltiples."""
    p = np.asarray(pvalores, dtype=float)
    orden = np.argsort(p)
    m = p.size
    ajustado = np.empty(m)
    acumulado = 0.0
    for rango, idx in enumerate(orden):
        acumulado = max(acumulado, (m - rango) * p[idx])
        ajustado[idx] = min(1.0, acumulado)
    return ajustado


def comparacion_pareada(x, y, confianza: float = 0.95) -> dict:
    """
    Compara dos métricas medidas sobre las mismas réplicas (p. ej., HMM vs. clasificación directa):
    diferencia media con IC t, prueba de rangos con signo de Wilcoxon y tamaño de efecto d_z.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    r = resumen(d, confianza)
    w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    dz = r["media"] / r["de"] if r["de"] > 0 else np.inf
    return {
        "dif_media": r["media"],
        "dif_ic_inf": r["ic_inf"],
        "dif_ic_sup": r["ic_sup"],
        "wilcoxon_W": float(w.statistic),
        "p_valor": float(w.pvalue),
        "d_z": float(dz),
    }


def comparacion_escenarios(df: pd.DataFrame, metrica: str, referencia: str, por: str = "escenario") -> tuple[dict, pd.DataFrame]:
    """Kruskal-Wallis global + Mann-Whitney de cada escenario contra la referencia (Holm)."""
    grupos = {g: s[metrica].to_numpy() for g, s in df.groupby(por, sort=False)}
    kw = stats.kruskal(*grupos.values())
    filas = []
    ref = grupos[referencia]
    for g, valores in grupos.items():
        if g == referencia:
            continue
        mw = stats.mannwhitneyu(valores, ref, alternative="two-sided")
        # Tamaño de efecto: probabilidad de superioridad A12 de Vargha-Delaney
        a12 = mw.statistic / (len(valores) * len(ref))
        filas.append({"escenario": g, "vs": referencia, "metrica": metrica,
                      "dif_medias": valores.mean() - ref.mean(), "U": float(mw.statistic),
                      "p_valor": float(mw.pvalue), "A12": float(a12)})
    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        tabla["p_holm"] = holm(tabla["p_valor"])
    return {"H": float(kw.statistic), "p_valor": float(kw.pvalue), "gl": len(grupos) - 1}, tabla


def regresion_estandarizada(X: np.ndarray, y: np.ndarray, nombres: list[str]) -> tuple[pd.DataFrame, float]:
    """Coeficientes de regresión estandarizados (SRC) para análisis de sensibilidad global."""
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    ys = (y - y.mean()) / y.std(ddof=1)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ys)), Xs]), ys, rcond=None)
    pred = np.column_stack([np.ones(len(ys)), Xs]) @ coef
    r2 = 1 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2)
    return pd.DataFrame({"parametro": nombres, "SRC": coef[1:]}), float(r2)
