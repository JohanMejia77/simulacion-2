"""
Modelo Oculto de Markov (HMM) discreto, implementado desde cero con NumPy.

Notación (Rabiner, 1989):
    S = {s_1, ..., s_K}      estados ocultos
    V = {v_1, ..., v_M}      símbolos observables
    A = [a_ij],  a_ij = P(X_{t+1} = s_j | X_t = s_i)        matriz de transición (K x K)
    B = [b_j(k)], b_j(k) = P(O_t = v_k | X_t = s_j)         matriz de emisión   (K x M)
    pi = [pi_i],  pi_i = P(X_1 = s_i)                       distribución inicial (K)

Todos los algoritmos están vectorizados sobre un lote de N secuencias de igual
longitud T (arreglo de enteros de forma (N, T)). Se usa el escalamiento de
Rabiner en Forward-Backward y logaritmos en Viterbi para evitar subdesbordamiento.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TOL_ESTOCASTICA = 1e-9


def _validar_estocastica(M: np.ndarray, nombre: str) -> None:
    if np.any(M < -_TOL_ESTOCASTICA):
        raise ValueError(f"{nombre} contiene probabilidades negativas")
    sumas = M.sum(axis=-1)
    if not np.allclose(sumas, 1.0, atol=1e-8):
        raise ValueError(f"Las filas de {nombre} deben sumar 1 (sumas = {sumas})")


@dataclass
class ResultadoBaumWelch:
    """Salida del algoritmo de Baum-Welch."""

    modelo: "HMMDiscreto"
    historial_logver: list[float] = field(default_factory=list)
    iteraciones: int = 0
    convergio: bool = False


class HMMDiscreto:
    """HMM de primer orden con emisiones categóricas, lambda = (A, B, pi)."""

    def __init__(self, A, B, pi):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        K, M = self.B.shape
        if self.A.shape != (K, K):
            raise ValueError("A debe ser K x K y compatible con B (K x M)")
        if self.pi.shape != (K,):
            raise ValueError("pi debe tener longitud K")
        _validar_estocastica(self.A, "A")
        _validar_estocastica(self.B, "B")
        _validar_estocastica(self.pi[None, :], "pi")
        self.K, self.M = K, M

    # ------------------------------------------------------------------ #
    # Simulación
    # ------------------------------------------------------------------ #
    @staticmethod
    def _muestrear_categorica(P: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Muestreo por transformada inversa: P (n x K) filas de probabilidad, u ~ U(0,1)."""
        acum = np.cumsum(P, axis=1)
        idx = (u[:, None] > acum).sum(axis=1)
        return np.minimum(idx, P.shape[1] - 1)

    def simular_estados(self, N: int, T: int, rng: np.random.Generator) -> np.ndarray:
        """Simula N trayectorias de la cadena oculta X_1..X_T (forma (N, T))."""
        X = np.empty((N, T), dtype=np.int64)
        X[:, 0] = self._muestrear_categorica(np.broadcast_to(self.pi, (N, self.K)), rng.random(N))
        for t in range(1, T):
            X[:, t] = self._muestrear_categorica(self.A[X[:, t - 1]], rng.random(N))
        return X

    def simular(self, N: int, T: int, rng: np.random.Generator):
        """Simula (X, O) muestreando las emisiones directamente de B."""
        X = self.simular_estados(N, T, rng)
        U = rng.random((N, T))
        O = self._muestrear_categorica(self.B[X.ravel()], U.ravel()).reshape(N, T)
        return X, O

    # ------------------------------------------------------------------ #
    # Problema 1: evaluación -> algoritmo Forward (escalado)
    # ------------------------------------------------------------------ #
    def forward(self, O: np.ndarray):
        """Devuelve alpha escalado (N,T,K), factores c (N,T) y log P(O|lambda) por secuencia."""
        O = np.atleast_2d(O)
        N, T = O.shape
        alpha = np.empty((N, T, self.K))
        c = np.empty((N, T))
        a = self.pi[None, :] * self.B[:, O[:, 0]].T
        c[:, 0] = a.sum(axis=1)
        alpha[:, 0] = a / c[:, 0, None]
        for t in range(1, T):
            a = (alpha[:, t - 1] @ self.A) * self.B[:, O[:, t]].T
            c[:, t] = a.sum(axis=1)
            alpha[:, t] = a / c[:, t, None]
        with np.errstate(divide="ignore"):
            logver = np.log(c).sum(axis=1)
        return alpha, c, logver

    def backward(self, O: np.ndarray, c: np.ndarray) -> np.ndarray:
        """beta escalado (N,T,K) usando los mismos factores c del Forward."""
        O = np.atleast_2d(O)
        N, T = O.shape
        beta = np.empty((N, T, self.K))
        beta[:, T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            b = self.B[:, O[:, t + 1]].T * beta[:, t + 1]
            beta[:, t] = (b @ self.A.T) / c[:, t + 1, None]
        return beta

    def log_verosimilitud(self, O: np.ndarray) -> float:
        """log P(O | lambda) total del lote."""
        return float(self.forward(O)[2].sum())

    def posteriores(self, O: np.ndarray) -> np.ndarray:
        """gamma_t(i) = P(X_t = s_i | O, lambda), forma (N, T, K)."""
        alpha, c, _ = self.forward(O)
        beta = self.backward(O, c)
        gamma = alpha * beta
        return gamma / gamma.sum(axis=2, keepdims=True)

    # ------------------------------------------------------------------ #
    # Problema 2: decodificación -> algoritmo de Viterbi (log-espacio)
    # ------------------------------------------------------------------ #
    def viterbi(self, O: np.ndarray):
        """Secuencia de estados más probable por secuencia y su log-probabilidad conjunta."""
        O = np.atleast_2d(O)
        N, T = O.shape
        with np.errstate(divide="ignore"):
            logA, logB, logpi = np.log(self.A), np.log(self.B), np.log(self.pi)
        delta = logpi[None, :] + logB[:, O[:, 0]].T
        psi = np.zeros((N, T, self.K), dtype=np.int64)
        for t in range(1, T):
            puntajes = delta[:, :, None] + logA[None, :, :]  # (N, origen, destino)
            psi[:, t] = puntajes.argmax(axis=1)
            delta = puntajes.max(axis=1) + logB[:, O[:, t]].T
        camino = np.empty((N, T), dtype=np.int64)
        camino[:, T - 1] = delta.argmax(axis=1)
        filas = np.arange(N)
        for t in range(T - 2, -1, -1):
            camino[:, t] = psi[filas, t + 1, camino[:, t + 1]]
        return camino, delta.max(axis=1)

    # ------------------------------------------------------------------ #
    # Problema 3: aprendizaje -> Baum-Welch (EM)
    # ------------------------------------------------------------------ #
    def _paso_em(self, O: np.ndarray):
        """Un paso E+M. Devuelve (A, B, pi) re-estimados y log P(O|lambda) actual."""
        N, T = O.shape
        alpha, c, logver = self.forward(O)
        beta = self.backward(O, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=2, keepdims=True)

        xi = np.zeros((self.K, self.K))
        for t in range(T - 1):
            b = self.B[:, O[:, t + 1]].T * beta[:, t + 1] / c[:, t + 1, None]
            xi += alpha[:, t].T @ b
        xi *= self.A  # conserva los ceros estructurales de A

        una_caliente = np.eye(self.M)[O]  # (N, T, M)
        num_B = np.einsum("ntk,ntm->km", gamma, una_caliente)
        A_nueva, B_nueva = self.A.copy(), self.B.copy()
        filas_A = xi.sum(axis=1) > 0  # un estado nunca visitado conserva su fila anterior
        A_nueva[filas_A] = xi[filas_A] / xi[filas_A].sum(axis=1, keepdims=True)
        filas_B = num_B.sum(axis=1) > 0
        B_nueva[filas_B] = num_B[filas_B] / num_B[filas_B].sum(axis=1, keepdims=True)
        pi_nueva = gamma[:, 0].mean(axis=0)
        return A_nueva, B_nueva, pi_nueva, float(logver.sum())

    @classmethod
    def baum_welch(
        cls,
        O: np.ndarray,
        A0: np.ndarray,
        B0: np.ndarray,
        pi0: np.ndarray,
        max_iter: int = 500,
        tol: float = 1e-6,
        piso: float = 1e-12,
    ) -> ResultadoBaumWelch:
        """
        Estima lambda = (A, B, pi) por máxima verosimilitud (EM) a partir de O (N, T).

        Las entradas nulas de A0 se mantienen nulas (estructura de deterioro conocida).
        Criterio de parada: mejora relativa de log P(O|lambda) < tol.
        `piso` evita probabilidades de emisión exactamente nulas.
        """
        O = np.atleast_2d(O)
        modelo = cls(A0, B0, pi0)
        historial: list[float] = []
        convergio = False
        it = 0
        for it in range(1, max_iter + 1):
            A, B, pi, logver = modelo._paso_em(O)
            historial.append(logver)
            B = np.maximum(B, piso)
            B /= B.sum(axis=1, keepdims=True)
            modelo = cls(A, B, pi)
            if len(historial) > 1:
                mejora = historial[-1] - historial[-2]
                if abs(mejora) < tol * abs(historial[-2]):
                    convergio = True
                    break
        historial.append(modelo.log_verosimilitud(O))
        return ResultadoBaumWelch(modelo=modelo, historial_logver=historial, iteraciones=it, convergio=convergio)


# ---------------------------------------------------------------------- #
# Propiedades de la cadena de Markov visible subyacente (A, pi)
# ---------------------------------------------------------------------- #
def distribucion_estacionaria(A: np.ndarray) -> np.ndarray:
    """Resuelve pi* A = pi*, sum(pi*) = 1 (cadena ergódica)."""
    K = A.shape[0]
    sistema = np.vstack([A.T - np.eye(K), np.ones((1, K))])
    lado_derecho = np.concatenate([np.zeros(K), [1.0]])
    pi_est, *_ = np.linalg.lstsq(sistema, lado_derecho, rcond=None)
    pi_est = np.clip(pi_est, 0.0, None)
    return pi_est / pi_est.sum()


def distribucion_marginal(pi0: np.ndarray, A: np.ndarray, pasos: int) -> np.ndarray:
    """pi_t = pi_0 A^t para t = 0..pasos (forma (pasos+1, K)) - ecuación de Chapman-Kolmogorov."""
    salida = np.empty((pasos + 1, len(pi0)))
    salida[0] = pi0
    for t in range(1, pasos + 1):
        salida[t] = salida[t - 1] @ A
    return salida


def tiempo_medio_permanencia(A: np.ndarray) -> np.ndarray:
    """E[permanencia en s_i] = 1 / (1 - a_ii): la permanencia sigue una distribución geométrica."""
    with np.errstate(divide="ignore"):
        return 1.0 / (1.0 - np.diag(A))


def tiempo_medio_primer_paso(A: np.ndarray, destino: list[int]) -> np.ndarray:
    """
    Tiempo esperado (en pasos) hasta alcanzar por primera vez el conjunto `destino`
    desde cada estado transitorio: (I - Q) m = 1 (matriz fundamental N = (I - Q)^-1).
    """
    K = A.shape[0]
    origen = [i for i in range(K) if i not in destino]
    Q = A[np.ix_(origen, origen)]
    m = np.linalg.solve(np.eye(len(origen)) - Q, np.ones(len(origen)))
    salida = np.zeros(K)
    salida[origen] = m
    return salida


def segundo_valor_propio(A: np.ndarray) -> float:
    """|lambda_2| de A: controla la velocidad geométrica de convergencia a pi*."""
    modulos = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    return float(modulos[1])


def distancia_variacion_total(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """TV(p, q) = 0.5 * sum |p - q| (sobre el último eje)."""
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1)
