"""Pruebas de verificación del HMM (ejecutar con: python -m pytest -q)."""
import itertools

import numpy as np
import pytest

from hmm_pavimento.experimentos import alinear_estados, experimento_escenarios, huella, parametros_iniciales
from hmm_pavimento.hmm import (
    HMMDiscreto,
    distribucion_estacionaria,
    tiempo_medio_permanencia,
    tiempo_medio_primer_paso,
)
from hmm_pavimento.modelo import cargar_config, construir_B, escenarios, simular_auscultaciones

CFG = cargar_config()
ESC = escenarios(CFG)


def _fuerza_bruta(hmm: HMMDiscreto, o: np.ndarray):
    """Enumera los K^T caminos: P(O|lambda) y el camino de máxima probabilidad."""
    total, mejor, mejor_p = 0.0, None, -1.0
    for camino in itertools.product(range(hmm.K), repeat=len(o)):
        p = hmm.pi[camino[0]] * hmm.B[camino[0], o[0]]
        for t in range(1, len(o)):
            p *= hmm.A[camino[t - 1], camino[t]] * hmm.B[camino[t], o[t]]
        total += p
        if p > mejor_p:
            mejor, mejor_p = camino, p
    return total, np.array(mejor), mejor_p


@pytest.mark.parametrize("esc", ESC, ids=[e.id for e in ESC])
def test_matrices_estocasticas(esc):
    assert np.allclose(esc.A.sum(axis=1), 1) and np.all(esc.A >= 0)
    assert np.allclose(esc.B.sum(axis=1), 1) and np.all(esc.B >= 0)
    assert np.isclose(esc.pi0.sum(), 1)


@pytest.mark.parametrize("semilla", [1, 2, 3])
def test_forward_y_viterbi_igual_a_fuerza_bruta(semilla):
    rng = np.random.default_rng(semilla)
    hmm = ESC[3].hmm()  # escenario ruidoso: decodificación no trivial
    _, O = hmm.simular(1, 6, rng)
    p_bruta, camino_bruto, p_camino = _fuerza_bruta(hmm, O[0])
    assert np.isclose(hmm.forward(O)[2][0], np.log(p_bruta), atol=1e-10)
    camino, logp = hmm.viterbi(O)
    assert np.isclose(logp[0], np.log(p_camino), atol=1e-10)
    assert np.array_equal(camino[0], camino_bruto)


def test_posteriores_suman_uno():
    hmm = ESC[0].hmm()
    _, O = hmm.simular(20, 15, np.random.default_rng(7))
    gamma = hmm.posteriores(O)
    assert np.allclose(gamma.sum(axis=2), 1)


def test_baum_welch_monotono_y_conserva_ceros():
    esc = ESC[3]
    hmm = esc.hmm()
    _, O = hmm.simular(100, 25, np.random.default_rng(11))
    A0, B0, p0 = parametros_iniciales(esc.A, hmm.M, 0.7)
    res = HMMDiscreto.baum_welch(O, A0, B0, p0, max_iter=200, tol=1e-9)
    hist = np.array(res.historial_logver)
    assert np.all(np.diff(hist) >= -1e-6), "EM debe ser monótono en la log-verosimilitud"
    assert np.all(res.modelo.A[esc.A == 0] == 0), "Los ceros estructurales de A deben conservarse"


def test_baum_welch_recupera_parametros_con_muchos_datos():
    esc = ESC[0]
    hmm = esc.hmm()
    _, O = hmm.simular(2000, 25, np.random.default_rng(5))
    A0, B0, p0 = parametros_iniciales(esc.A, hmm.M, 0.7)
    m, _, _ = alinear_estados(HMMDiscreto.baum_welch(O, A0, B0, p0, max_iter=500, tol=1e-9).modelo, esc.B)
    assert np.abs(m.A - esc.A).max() < 0.02
    assert np.abs(m.B - esc.B).max() < 0.02


def test_B_forma_cerrada_igual_a_integracion_numerica():
    for sigma in (0.05, 0.15, 0.5, 1.0):
        err = {"distribucion": "normal", "sigma": sigma}
        B_cerrada = construir_B(CFG["rangos_iri_estado"], CFG["umbrales_iri"], err)
        B_num = construir_B(CFG["rangos_iri_estado"], CFG["umbrales_iri"], err, forzar_numerica=True)
        assert np.allclose(B_cerrada, B_num, atol=1e-8)


@pytest.mark.parametrize("esc", [ESC[3], ESC[4]], ids=["normal", "t3"])
def test_simulador_fisico_reproduce_B(esc):
    """Las frecuencias empíricas de la banda observada por estado coinciden con B."""
    rng = np.random.default_rng(3)
    X = np.repeat(np.arange(4), 100_000)[None, :]
    O, _, _ = simular_auscultaciones(X, CFG["rangos_iri_estado"], CFG["umbrales_iri"], esc.error, rng)
    emp = np.array([np.bincount(O[0][X[0] == j], minlength=4) / 100_000 for j in range(4)])
    assert np.abs(emp - esc.B).max() < 0.006


def test_propiedades_cadena():
    A = ESC[0].A
    pi_est = distribucion_estacionaria(A)
    assert np.allclose(pi_est @ A, pi_est)
    assert np.allclose(tiempo_medio_permanencia(A), 1 / (1 - np.diag(A)))
    # Simulación de tiempos de primer paso Bueno -> {Malo, Crítico}
    rng = np.random.default_rng(9)
    hmm = HMMDiscreto(A, ESC[0].B, np.array([1.0, 0, 0, 0]))
    X = hmm.simular_estados(20_000, 200, rng)
    llegada = np.argmax(X >= 2, axis=1)
    assert abs(llegada.mean() - tiempo_medio_primer_paso(A, [2, 3])[0]) < 0.15


def test_reproducibilidad_semilla():
    a, _ = experimento_escenarios(CFG, R=2, N=20, T=10, solo=["E1"])
    b, _ = experimento_escenarios(CFG, R=2, N=20, T=10, solo=["E1"])
    c, _ = experimento_escenarios(CFG, R=2, N=20, T=10, solo=["E1"], semilla=CFG["semilla_maestra"] + 1)
    assert huella(a) == huella(b)
    assert huella(a) != huella(c)


def test_coincide_con_hmmlearn():
    hmmlearn = pytest.importorskip("hmmlearn.hmm")
    esc = ESC[3]
    hmm = esc.hmm()
    _, O = hmm.simular(5, 25, np.random.default_rng(21))
    ref = hmmlearn.CategoricalHMM(n_components=4, init_params="", params="")
    ref.startprob_, ref.transmat_, ref.emissionprob_ = esc.pi0, esc.A, esc.B
    for o in O:
        assert np.isclose(ref.score(o[:, None]), hmm.forward(o[None])[2][0], atol=1e-8)
        _, camino_ref = ref.decode(o[:, None], algorithm="viterbi")
        assert np.array_equal(camino_ref, hmm.viterbi(o[None])[0][0])
