"""
Especificación del problema: deterioro del pavimento observado mediante el IRI.

Estado oculto X_t (condición real del tramo en el año t):
    s_1 = Bueno, s_2 = Regular, s_3 = Malo, s_4 = Crítico
Observación O_t: banda del IRI *medido* en la auscultación anual
    v_1: IRI < 1.50 m/km, v_2: [1.50, 2.68), v_3: [2.68, 4.00), v_4: >= 4.00 m/km
(1.50 y 2.68 m/km equivalen a los umbrales de 95 y 170 in/mi de la FHWA; 4.00 m/km
es un supuesto del estudio para separar la condición crítica).

Modelo generativo de la emisión: IRI_real | X_t = s_j ~ Uniforme(a_j, b_j) y
IRI_medido = IRI_real + epsilon, con epsilon ~ D(0, sigma^2) (normal, t de Student o Laplace).
Por tanto b_j(k) = P(c_{k-1} <= IRI_medido < c_k | X_t = s_j) se obtiene de la convolución
Uniforme * D, con forma cerrada en el caso gaussiano.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import integrate, stats

from .hmm import HMMDiscreto

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RAIZ / "config" / "escenarios.json"


def cargar_config(ruta: Path | str = RUTA_CONFIG) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------- #
# Matriz de transición A (cadena de deterioro con política de intervención)
# ---------------------------------------------------------------------- #
def construir_A(p: dict) -> np.ndarray:
    """
    Construye A a partir de probabilidades anuales interpretables:
      d_BR, d_BM : Bueno -> Regular / Malo        d_RM, d_RC : Regular -> Malo / Crítico
      d_MC       : Malo -> Crítico                 r_M : intervención preventiva en Malo (-> Bueno)
      r_C        : rehabilitación en Crítico (-> Bueno)
      factor_transito : multiplica todas las probabilidades de deterioro d_*.
    """
    f = p.get("factor_transito", 1.0)
    dBR, dBM = f * p["d_BR"], f * p["d_BM"]
    dRM, dRC = f * p["d_RM"], f * p["d_RC"]
    dMC = f * p["d_MC"]
    rM, rC = p.get("r_M", 0.0), p["r_C"]
    A = np.array(
        [
            [1 - dBR - dBM, dBR, dBM, 0.0],
            [0.0, 1 - dRM - dRC, dRM, dRC],
            [rM, 0.0, (1 - rM) * (1 - dMC), (1 - rM) * dMC],
            [rC, 0.0, 0.0, 1 - rC],
        ]
    )
    if np.any(A < 0) or np.any(A > 1):
        raise ValueError(f"Parámetros de transición fuera de rango: {p}")
    return A


# ---------------------------------------------------------------------- #
# Matriz de emisión B (derivada del modelo de error de medición)
# ---------------------------------------------------------------------- #
def distribucion_error(nombre: str, sigma: float, gl: float | None = None):
    """Distribución del error de medición con media 0 y desviación estándar sigma."""
    if nombre == "normal":
        return stats.norm(loc=0.0, scale=sigma)
    if nombre == "t":
        if gl is None or gl <= 2:
            raise ValueError("La t de Student requiere gl > 2 para tener varianza finita")
        return stats.t(df=gl, loc=0.0, scale=sigma * np.sqrt((gl - 2) / gl))
    if nombre == "laplace":
        return stats.laplace(loc=0.0, scale=sigma / np.sqrt(2))
    raise ValueError(f"Distribución de error no soportada: {nombre}")


def _G(z):
    """Antiderivada de Phi: G(z) = z Phi(z) + phi(z)."""
    return z * stats.norm.cdf(z) + stats.norm.pdf(z)


def _cdf_medido_normal(c: float, a: float, b: float, sigma: float) -> float:
    """P(U + eps < c), U ~ Unif(a, b), eps ~ N(0, sigma^2): forma cerrada."""
    return sigma / (b - a) * (_G((c - a) / sigma) - _G((c - b) / sigma))


def _cdf_medido_numerica(c: float, a: float, b: float, dist) -> float:
    """P(U + eps < c) = (1/(b-a)) * integral_a^b F_eps(c - x) dx (cuadratura adaptativa)."""
    valor, _ = integrate.quad(lambda x: dist.cdf(c - x), a, b, epsabs=1e-12, epsrel=1e-10, limit=200)
    return valor / (b - a)


def construir_B(rangos: list, umbrales: list, error: dict, forzar_numerica: bool = False) -> np.ndarray:
    """b_j(k) = F_j(c_k) - F_j(c_{k-1}), con F_j la CDF del IRI medido en el estado j."""
    nombre, sigma, gl = error["distribucion"], error["sigma"], error.get("gl")
    dist = distribucion_error(nombre, sigma, gl)
    K, M = len(rangos), len(umbrales) + 1
    B = np.zeros((K, M))
    for j, (a, b) in enumerate(rangos):
        if nombre == "normal" and not forzar_numerica:
            F = [_cdf_medido_normal(c, a, b, sigma) for c in umbrales]
        else:
            F = [_cdf_medido_numerica(c, a, b, dist) for c in umbrales]
        acumulada = np.concatenate([[0.0], F, [1.0]])
        B[j] = np.diff(acumulada)
    B = np.clip(B, 0.0, None)
    return B / B.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------- #
# Escenarios
# ---------------------------------------------------------------------- #
@dataclass
class Escenario:
    id: str
    nombre: str
    descripcion: str
    parametros_A: dict
    error: dict
    nombre_pi0: str
    A: np.ndarray
    B: np.ndarray
    pi0: np.ndarray

    def hmm(self) -> HMMDiscreto:
        return HMMDiscreto(self.A, self.B, self.pi0)


def construir_escenario(cfg: dict, esc: dict) -> Escenario:
    pA = copy.deepcopy(cfg["transicion_base"])
    pA.update(esc.get("transicion", {}))
    error = copy.deepcopy(cfg["error_base"])
    error.update(esc.get("error", {}))
    nombre_pi0 = esc.get("pi0", cfg["pi0_base"])
    return Escenario(
        id=esc["id"],
        nombre=esc["nombre"],
        descripcion=esc["descripcion"],
        parametros_A=pA,
        error=error,
        nombre_pi0=nombre_pi0,
        A=construir_A(pA),
        B=construir_B(cfg["rangos_iri_estado"], cfg["umbrales_iri"], error),
        pi0=np.asarray(cfg["distribuciones_iniciales"][nombre_pi0], dtype=float),
    )


def escenarios(cfg: dict | None = None) -> list[Escenario]:
    cfg = cfg or cargar_config()
    return [construir_escenario(cfg, e) for e in cfg["escenarios"]]


# ---------------------------------------------------------------------- #
# Simulador "físico": IRI real + error de medición -> banda observada
# ---------------------------------------------------------------------- #
def muestrear_error(error: dict, size, rng: np.random.Generator) -> np.ndarray:
    nombre, sigma = error["distribucion"], error["sigma"]
    if nombre == "normal":
        return rng.normal(0.0, sigma, size)
    if nombre == "t":
        gl = error["gl"]
        return rng.standard_t(gl, size) * sigma * np.sqrt((gl - 2) / gl)
    if nombre == "laplace":
        return rng.laplace(0.0, sigma / np.sqrt(2), size)
    raise ValueError(nombre)


def simular_auscultaciones(
    X: np.ndarray, rangos: list, umbrales: list, error: dict, rng: np.random.Generator
):
    """
    Dado X (N, T) genera IRI_real ~ Unif(a_X, b_X), IRI_medido = IRI_real + eps y
    la banda observada O = digitize(IRI_medido, umbrales). Devuelve (O, IRI_medido, IRI_real).
    """
    rangos = np.asarray(rangos, dtype=float)
    a, b = rangos[X, 0], rangos[X, 1]
    iri_real = a + (b - a) * rng.random(X.shape)
    iri_medido = iri_real + muestrear_error(error, X.shape, rng)
    O = np.digitize(iri_medido, umbrales)
    return O.astype(np.int64), iri_medido, iri_real
