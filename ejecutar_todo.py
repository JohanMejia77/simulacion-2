"""
Ejecuta todos los experimentos, tablas y figuras del artículo.

Uso:
    python ejecutar_todo.py                 # corrida completa (resultados/), ~15 min
    python ejecutar_todo.py --rapido        # prueba de humo con pocas réplicas (resultados_rapido/)
    python ejecutar_todo.py --solo-figuras  # regenera figuras desde los CSV ya guardados
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_pavimento import experimentos as ex
from hmm_pavimento import figuras as fg
from hmm_pavimento import metricas as met
from hmm_pavimento.modelo import cargar_config, escenarios

RAIZ = Path(__file__).resolve().parent
ETIQUETAS_A = {"d_BR": "d_BR  Bueno → Regular", "d_BM": "d_BM  Bueno → Malo", "d_RM": "d_RM  Regular → Malo",
               "d_RC": "d_RC  Regular → Crítico", "d_MC": "d_MC  Malo → Crítico", "r_C": "r_C  rehabilitación de Crítico"}
METRICAS_PRINCIPALES = [
    "exactitud_ingenua", "exactitud_viterbi", "exactitud_bw", "kappa_ingenua", "kappa_viterbi",
    "sens_def_ingenua", "sens_def_bw", "prec_def_ingenua", "prec_def_bw",
    "mae_A", "mae_B", "iteraciones_bw", "logver_por_obs", "error_pdef_ingenua", "error_pdef_hmm",
]


def _json(obj, ruta: Path) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    ruta.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def registrar_entorno(cfg: dict, salida: Path) -> dict:
    paquetes = {}
    for p in ("numpy", "scipy", "pandas", "matplotlib", "hmmlearn", "pytest"):
        try:
            paquetes[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            paquetes[p] = "no instalado"
    info = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "implementacion": platform.python_implementation(),
        "sistema": platform.platform(),
        "procesador": platform.processor(),
        "paquetes": paquetes,
        "semilla_maestra": cfg["semilla_maestra"],
        "generador": "numpy.random.Generator(PCG64) con SeedSequence([semilla, experimento, índice])",
        "sha256_config": hashlib.sha256((RAIZ / "config" / "escenarios.json").read_bytes()).hexdigest(),
    }
    _json(info, salida / "entorno.json")
    return info


def correr(cfg: dict, salida: Path) -> None:
    tablas = salida / "tablas"
    tablas.mkdir(parents=True, exist_ok=True)
    t_inicio = time.perf_counter()
    registrar_entorno(cfg, salida)

    print("[1/7] Escenarios principales E1-E5")
    df, analitico = ex.experimento_escenarios(cfg)
    df.to_csv(tablas / "replicas_escenarios.csv", index=False)
    analitico.to_csv(tablas / "propiedades_analiticas.csv", index=False)
    resumen = met.tabla_resumen(df, METRICAS_PRINCIPALES + (["exactitud_viterbi_B_gauss"] if "exactitud_viterbi_B_gauss" in df else []))
    resumen.to_csv(tablas / "resumen_escenarios.csv", index=False)

    pruebas = []
    for e, sub in df.groupby("escenario", sort=False):
        for a, b in (("exactitud_bw", "exactitud_ingenua"), ("exactitud_viterbi", "exactitud_bw"),
                     ("sens_def_bw", "sens_def_ingenua")):
            fila = {"escenario": e, "comparacion": f"{a} - {b}"}
            fila.update(met.comparacion_pareada(sub[a], sub[b]))
            pruebas.append(fila)
    pruebas = pd.DataFrame(pruebas)
    pruebas["p_holm"] = met.holm(pruebas["p_valor"])
    pruebas.to_csv(tablas / "pruebas_pareadas.csv", index=False)
    kw = {}
    comparaciones = []
    for m in ("exactitud_bw", "exactitud_ingenua", "mae_B"):
        kw[m], tabla = met.comparacion_escenarios(df, m, referencia="E1")
        comparaciones.append(tabla)
    pd.concat(comparaciones).to_csv(tablas / "comparacion_escenarios.csv", index=False)
    _json(kw, tablas / "kruskal_wallis.json")

    print("[2/7] Barrido del ruido de medición sigma (matriz B)")
    df_sigma = ex.experimento_barrido_sigma(cfg)
    df_sigma.to_csv(tablas / "barrido_sigma.csv", index=False)
    met.tabla_resumen(df_sigma, ["exactitud_ingenua", "exactitud_viterbi", "exactitud_bw", "mae_B"], por="sigma") \
        .to_csv(tablas / "resumen_barrido_sigma.csv", index=False)

    print("[3/7] Curva de aprendizaje de Baum-Welch")
    df_tam = ex.experimento_tamano_muestra(cfg)
    df_tam.to_csv(tablas / "tamano_muestra.csv", index=False)
    df_tam["grupo"] = df_tam["escenario"] + "_N" + df_tam["N"].astype(str)
    met.tabla_resumen(df_tam, ["mae_A", "mae_B", "exactitud_bw"], por="grupo").to_csv(tablas / "resumen_tamano_muestra.csv", index=False)

    print("[4/7] Sensibilidad a la distribución inicial pi")
    pi_an, pi_sim, pi_est, lam2 = ex.experimento_pi(cfg)
    pi_an.to_csv(tablas / "pi_analitico.csv", index=False)
    pi_sim.to_csv(tablas / "pi_simulado.csv", index=False)

    print("[5/7] Sensibilidad de la matriz de transición A")
    oat, glob, res_glob = ex.experimento_sensibilidad_A(cfg)
    oat.to_csv(tablas / "sensibilidad_A_oat.csv", index=False)
    glob.to_csv(tablas / "sensibilidad_A_global_src.csv", index=False)
    _json(res_glob, tablas / "sensibilidad_A_global_resumen.json")

    print("[6/7] Reinicios de Baum-Welch")
    reinicios, curvas, logver_real = ex.experimento_reinicios(cfg)
    reinicios.to_csv(tablas / "reinicios_bw.csv", index=False)
    curvas.to_csv(tablas / "curvas_bw.csv", index=False)
    _json({"logver_real": logver_real}, tablas / "reinicios_bw_referencia.json")
    esp = ex.experimento_transiciones_espurias(cfg)
    esp.to_csv(tablas / "transiciones_espurias.csv", index=False)
    met.tabla_resumen(esp, ["tasa_mejora_observada", "tasa_mejora_real", "fraccion_espurias"]) \
        .to_csv(tablas / "resumen_transiciones_espurias.csv", index=False)

    print("[7/7] Reproducibilidad")
    repro = ex.experimento_reproducibilidad(cfg)
    _json(repro, salida / "reproducibilidad.json")

    figuras(cfg, salida)
    huellas = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tablas.glob("*.csv"))}
    huellas["_sha256_contenido_replicas_escenarios(redondeo 8 dec.)"] = ex.huella(df)
    _json(huellas, salida / "huellas_sha256.json")
    print(f"Listo en {(time.perf_counter() - t_inicio) / 60:.1f} min -> {salida}")


def figuras(cfg: dict, salida: Path) -> None:
    tablas, figs = salida / "tablas", salida / "figuras"
    fg.estilo()
    df = pd.read_csv(tablas / "replicas_escenarios.csv")
    nombres = {e.id: e.nombre for e in escenarios(cfg)}
    fg.fig_diagrama_flujo(figs / "fig1_diagrama_flujo")
    fg.fig_modelo(cfg, figs / "fig2_modelo_A_B")
    info = fg.fig_ejemplo(cfg, figs / "figB1_ejemplo_decodificacion")
    _json(info, tablas / "figB1_tramo_ilustrativo.json")
    fg.fig_escenarios(df, nombres, figs / "fig3_comparacion_escenarios")
    fg.fig_sensibilidad_ruido(pd.read_csv(tablas / "barrido_sigma.csv"), pd.read_csv(tablas / "tamano_muestra.csv"),
                              figs / "fig4_sensibilidad_ruido_y_muestra")
    esc = escenarios(cfg)[0]
    from hmm_pavimento.hmm import distribucion_estacionaria, segundo_valor_propio
    fg.fig_sensibilidad_pi_A(pd.read_csv(tablas / "pi_analitico.csv"), pd.read_csv(tablas / "pi_simulado.csv"),
                             pd.read_csv(tablas / "sensibilidad_A_oat.csv"), distribucion_estacionaria(esc.A),
                             segundo_valor_propio(esc.A), figs / "fig5_sensibilidad_pi_A", ETIQUETAS_A)
    ref = json.loads((tablas / "reinicios_bw_referencia.json").read_text(encoding="utf-8"))
    n_obs = cfg["simulacion"]["N_segmentos"] * cfg["simulacion"]["T_anios"]
    fg.fig_reinicios(pd.read_csv(tablas / "curvas_bw.csv"), pd.read_csv(tablas / "reinicios_bw.csv"),
                     ref["logver_real"], n_obs, figs / "figA1_reinicios_baum_welch")
    print(f"Figuras guardadas en {figs}")


def config_rapida(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["simulacion"]["R_replicas"] = 8
    s = c["sensibilidad"]
    s.update({"sigmas": [0.05, 0.15, 0.30, 0.50, 1.00], "R_sigma": 4, "tamanos_N": [25, 100, 400],
              "R_tamano": 4, "muestras_globales_A": 300, "R_pi": 10, "reinicios_bw": 4})
    return c


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="pocas réplicas, salida en resultados_rapido/")
    ap.add_argument("--solo-figuras", action="store_true", help="regenera las figuras desde los CSV guardados")
    args = ap.parse_args()
    cfg = cargar_config()
    salida = RAIZ / ("resultados_rapido" if args.rapido else "resultados")
    if args.rapido:
        cfg = config_rapida(cfg)
    if args.solo_figuras:
        figuras(cfg, salida)
    else:
        correr(cfg, salida)


if __name__ == "__main__":
    main()
