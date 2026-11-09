"""Figuras de calidad de publicación (PNG a 300 ppp + PDF vectorial)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

from .experimentos import generadores  # noqa: E402
from .metricas import resumen  # noqa: E402
from .modelo import escenarios, simular_auscultaciones  # noqa: E402

# Paleta (referencia validada: categóricos azul/naranja/aqua; estados con la paleta de estado)
TINTA, TINTA2, TENUE, REJILLA, EJE, FONDO, NEUTRO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff", "#f0efec"
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR_ESTADO = ["#0ca30c", "#fab219", "#ec835a", "#d03b3b"]
TEXTO_ESTADO = ["#ffffff", "#0b0b0b", "#ffffff", "#ffffff"]
RAMPA_AZUL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_AZUL = LinearSegmentedColormap.from_list("rampa_azul", RAMPA_AZUL)
METODOS = [
    ("exactitud_ingenua", "Clasificación directa del IRI medido", TENUE),
    ("exactitud_viterbi", "HMM-Viterbi con λ verdadero (cota)", AZUL),
    ("exactitud_bw", "HMM-Viterbi con λ estimado (Baum–Welch)", NARANJA),
]
ANCHO = 6.5  # pulgadas (ancho de texto en carta con márgenes de 2.5 cm)


def estilo() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.2,
        "axes.edgecolor": EJE, "axes.linewidth": 0.8, "axes.labelcolor": TINTA2,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.titlecolor": TINTA,
        "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "legend.frameon": False, "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 300, "figure.dpi": 110, "pdf.fonttype": 42,
    })


def guardar(fig, ruta_base: Path) -> None:
    ruta_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(ruta_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _pct(ax, eje="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f} %")
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------- #
# Figura 1: diagrama de flujo de la metodología
# ---------------------------------------------------------------------- #
def _caja(ax, x, y, w, h, texto, tipo="proceso", acento=False):
    if tipo == "terminal":
        p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                           fc=TINTA2, ec="none")
        ax.add_patch(p)
        ax.text(x, y, texto, ha="center", va="center", color="#ffffff", fontsize=8.2, fontweight="semibold")
        return
    if tipo == "decision":
        pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
        ax.add_patch(Polygon(pts, closed=True, fc=FONDO, ec=AZUL, lw=1.2))
        ax.text(x, y, texto, ha="center", va="center", color=TINTA, fontsize=8)
        return
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                       fc="#e8f1fc" if acento else NEUTRO, ec=AZUL if acento else EJE, lw=1.0)
    ax.add_patch(p)
    ax.text(x, y, texto, ha="center", va="center", color=TINTA, fontsize=6.9, linespacing=1.3)


def _flecha(ax, xy0, xy1, texto=None, desplazamiento=(0.012, 0.0), conn="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=9, color=TINTA2, lw=1.0,
                                 shrinkA=0, shrinkB=0, connectionstyle=conn))
    if texto:
        xm, ym = (xy0[0] + xy1[0]) / 2 + desplazamiento[0], (xy0[1] + xy1[1]) / 2 + desplazamiento[1]
        ax.text(xm, ym, texto, fontsize=7.5, color=TINTA2, ha="left", va="center")


def fig_diagrama_flujo(ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(ANCHO, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.445, 0.085
    xl, xr = 0.235, 0.765
    izq = [
        (0.955, "Inicio", "terminal"),
        (0.845, "1. Problema: estados ocultos S (condición real)\ny símbolos V (bandas del IRI medido)", "proceso"),
        (0.720, "2. Parámetros λ = (A, B, π) por escenario E1–E5\n(config/escenarios.json)", "proceso"),
        (0.595, "3. Simulación con semilla fija: 200 tramos × 25 años\nX ~ Markov(A, π);  O = banda(IRI real + ε)", "acento"),
        (0.470, "4. Inferencia: Forward · Viterbi · Baum–Welch\n(" + r"$\hat{\lambda}$" + " estimado solo con O)", "acento"),
        (0.345, "5. Métricas por réplica: exactitud, κ, sensibilidad,\nerror de Â y B̂, % de red deficiente", "proceso"),
    ]
    for y, texto, tipo in izq:
        _caja(ax, xl, y, w if tipo != "terminal" else 0.16, h if tipo != "terminal" else 0.05, texto,
              tipo="terminal" if tipo == "terminal" else "proceso", acento=tipo == "acento")
    for (y0, _, t0), (y1, _, t1) in zip(izq[:-1], izq[1:]):
        a = (0.05 if t0 == "terminal" else h) / 2
        _flecha(ax, (xl, y0 - a), (xl, y1 + h / 2))
    yd = 0.18
    _caja(ax, xl, yd, 0.25, 0.13, "¿Quedan réplicas\no escenarios?", tipo="decision")
    _flecha(ax, (xl, izq[-1][0] - h / 2), (xl, yd + 0.065))
    # Lazo "Sí" de regreso a la simulación
    xlazo = 0.006
    ax.plot([xl - 0.125, xlazo], [yd, yd], color=AZUL, lw=1.0)
    ax.plot([xlazo, xlazo], [yd, 0.595], color=AZUL, lw=1.0)
    ax.add_patch(FancyArrowPatch((xlazo, 0.595), (xl - w / 2, 0.595), arrowstyle="-|>", mutation_scale=9,
                                 color=AZUL, lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(0.018, yd - 0.035, "Sí (r ← r + 1)", fontsize=7.2, color=AZUL)
    der = [
        (0.845, "6. Estadística por escenario: media, DE, mín–máx,\nIC 95 %; Wilcoxon y Kruskal–Wallis con Holm", "proceso"),
        (0.720, "7. Sensibilidad: ruido σ (matriz B), π inicial y\nmatriz A (OAT ± 20 % y SRC global)", "acento"),
        (0.595, "8. Robustez de Baum–Welch: 21 inicializaciones\ny curva de aprendizaje (N = 25 a 400)", "proceso"),
        (0.470, "9. Reproducibilidad: huellas SHA-256, pruebas\nunitarias y registro del entorno", "proceso"),
        (0.345, "10. Figuras, tablas y respuesta a P1–P3", "proceso"),
        (0.230, "Fin", "terminal"),
    ]
    for y, texto, tipo in der:
        _caja(ax, xr, y, w if tipo != "terminal" else 0.16, h if tipo != "terminal" else 0.05, texto,
              tipo="terminal" if tipo == "terminal" else "proceso", acento=tipo == "acento")
    for (y0, _, t0), (y1, _, t1) in zip(der[:-1], der[1:]):
        b = (0.05 if t1 == "terminal" else h) / 2
        _flecha(ax, (xr, y0 - h / 2), (xr, y1 + b))
    # "No": de la decisión a la columna derecha
    ax.plot([xl + 0.125, 0.5], [yd, yd], color=TINTA2, lw=1.0)
    ax.plot([0.5, 0.5], [yd, 0.845], color=TINTA2, lw=1.0)
    _flecha(ax, (0.5, 0.845), (xr - w / 2, 0.845))
    ax.text(xl + 0.14, yd - 0.035, "No", fontsize=7.2, color=TINTA2)
    ax.text(0.5, 0.04, "Bloques resaltados en azul: núcleo probabilístico (simulación, inferencia y sensibilidad).",
            ha="center", fontsize=7.0, color=TENUE)
    guardar(fig, ruta)


# ---------------------------------------------------------------------- #
# Figura 2: especificación del modelo (diagrama de estados + matrices B)
# ---------------------------------------------------------------------- #
def _arco(ax, p0, p1, rad, texto, pos_texto, color=TINTA2, estilo="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color=color, lw=1.1,
                                 linestyle=estilo, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    ax.text(*pos_texto, texto, ha="center", va="center", fontsize=8.2, color=TINTA,
            bbox=dict(boxstyle="round,pad=0.12", fc=FONDO, ec="none"))


def fig_modelo(cfg: dict, ruta: Path) -> None:
    esc = escenarios(cfg)
    A = esc[0].A
    fig = plt.figure(figsize=(ANCHO, 5.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 10)
    ax.set_ylim(0.0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Cadena oculta de deterioro y rehabilitación (escenario base E1, paso = 1 año)", pad=2)
    xs, y, r = [1.0, 3.6, 6.2, 8.8], 2.0, 0.56
    for i, (x, nombre) in enumerate(zip(xs, cfg["estados"])):
        ax.add_patch(Circle((x, y), r, fc=COLOR_ESTADO[i], ec="none", zorder=3))
        ax.text(x, y + 0.08, nombre, ha="center", va="center", color=TEXTO_ESTADO[i], fontsize=8.4,
                fontweight="bold", zorder=4)
        ax.text(x, y - 0.22, f"$s_{i + 1}$", ha="center", va="center", color=TEXTO_ESTADO[i], fontsize=8.4, zorder=4)
        # Lazo de permanencia a_ii (arriba)
        p0, p1 = (x - 0.28, y + r * 0.86), (x + 0.28, y + r * 0.86)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=TINTA2, lw=1.1,
                                     connectionstyle="arc3,rad=-1.6", shrinkA=0, shrinkB=0))
        ax.text(x, y + r + 0.62, f"{A[i, i]:.2f}", ha="center", va="center", fontsize=8.2, color=TINTA)
    for i in range(3):  # deterioro de un nivel
        p0, p1 = (xs[i] + r, y), (xs[i + 1] - r, y)
        _arco(ax, p0, p1, 0.0, f"{A[i, i + 1]:.2f}", ((p0[0] + p1[0]) / 2, y + 0.2))
    # Saltos de dos niveles: Bueno -> Malo (arriba) y Regular -> Crítico (abajo)
    a60 = np.deg2rad(55)
    _arco(ax, (xs[0] + r * np.cos(a60), y + r * np.sin(a60)), (xs[2] - r * np.cos(a60), y + r * np.sin(a60)),
          -0.56, f"{A[0, 2]:.2f}", (3.6, 3.74))
    _arco(ax, (xs[1] + r * np.cos(a60), y - r * np.sin(a60)), (xs[3] - r * np.cos(a60), y - r * np.sin(a60)),
          0.30, f"{A[1, 3]:.2f}", (6.2, 0.86))
    # Rehabilitación Crítico -> Bueno (abajo, largo)
    _arco(ax, (xs[3] - r * 0.3, y - r * 0.95), (xs[0] + r * 0.3, y - r * 0.95), -0.30,
          f"rehabilitación {A[3, 0]:.2f}", (4.9, 0.35), color=AZUL)

    titulos = [("(b) B en E1 (σ = 0.15, normal)", esc[0].B),
               ("(c) B en E4 (σ = 0.50, normal)", esc[3].B),
               ("(d) B en E5 (σ = 0.50, t₃)", esc[4].B)]
    simbolos = ["<1.50", "1.50–2.68", "2.68–4.00", "≥4.00"]
    for k, (titulo, B) in enumerate(titulos):
        axb = fig.add_subplot(gs[1, k])
        axb.imshow(B, cmap=CMAP_AZUL, vmin=0, vmax=1, aspect="auto")
        axb.grid(False)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                v = B[i, j]
                axb.text(j, i, f"{v:.2f}" if v >= 0.005 else "0", ha="center", va="center", fontsize=7.6,
                         color="#ffffff" if v > 0.55 else TINTA)
        axb.set_xticks(range(4), simbolos, rotation=35, ha="right", fontsize=7.4)
        axb.set_yticks(range(4), cfg["estados"] if k == 0 else [""] * 4, fontsize=7.8)
        axb.tick_params(length=0)
        for s in axb.spines.values():
            s.set_visible(False)
        axb.set_title(titulo, fontsize=8.2)
        if k == 0:
            axb.set_ylabel("Estado oculto (condición real)")
        if k == 1:
            axb.set_xlabel("Banda del IRI medido, m/km (símbolo observado)")
    guardar(fig, ruta)


# ---------------------------------------------------------------------- #
# Figura 3: ejemplo de decodificación de un tramo
# ---------------------------------------------------------------------- #
def fig_ejemplo(cfg: dict, ruta: Path, id_escenario: str = "E4") -> dict:
    esc = next(e for e in escenarios(cfg) if e.id == id_escenario)
    hmm = esc.hmm()
    rng = generadores(cfg["semilla_maestra"], 1, 8)[0]
    N, T = 200, cfg["simulacion"]["T_anios"]
    X = hmm.simular_estados(N, T, rng)
    O, iri_med, iri_real = simular_auscultaciones(X, cfg["rangos_iri_estado"], cfg["umbrales_iri"], esc.error, rng)
    Xv, _ = hmm.viterbi(O)
    # Tramo ilustrativo: el que visita los 4 estados con mayor ganancia del HMM sobre la lectura directa
    visita_todos = np.array([len(np.unique(x)) == 4 for x in X])
    ganancia = (Xv == X).sum(1) - (O == X).sum(1)
    n = int(np.argmax(np.where(visita_todos, ganancia, -999)))
    anios = np.arange(1, T + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(ANCHO, 4.4), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    for c in cfg["umbrales_iri"]:
        a1.axhline(c, color=EJE, lw=0.9)
    a1.plot(anios, iri_real[n], color=TINTA2, lw=1.2, label="IRI real (no observable)")
    a1.plot(anios, iri_med[n], "o", ms=4.2, mfc=AQUA, mec=FONDO, mew=0.8, label="IRI medido (ε ~ N(0, 0.50²))")
    a1.set_ylabel("IRI (m/km)")
    a1.set_ylim(-0.2, 7.6)
    a1.set_yticks([0, 2, 4, 6])
    a1.set_title(f"(a) Tramo ilustrativo del escenario {id_escenario}: IRI real vs. medido", pad=3)
    a1.legend(loc="upper left", ncol=2)
    for c, txt in zip([0.9, 2.05, 3.3, 4.6], ["Bueno", "Regular", "Malo", "Crítico"]):
        a1.text(T + 0.7, c, txt, fontsize=7.6, color=TENUE, va="center")
    a1.set_xlim(0.5, T + 2.6)
    a2.step(anios, X[n], where="mid", color=TINTA2, lw=2.2, label="Estado real")
    a2.step(anios, Xv[n], where="mid", color=AZUL, lw=1.5, label="Estado decodificado (Viterbi)")
    err = O[n] != X[n]
    a2.plot(anios[~err], O[n][~err], "o", ms=4.2, mfc=FONDO, mec=TENUE, mew=1.1, label="Lectura directa correcta")
    a2.plot(anios[err], O[n][err], "X", ms=6, mfc=NARANJA, mec=FONDO, mew=0.6, label="Lectura directa errónea")
    a2.set_yticks(range(4), cfg["estados"])
    a2.set_ylim(-0.5, 3.5)
    a2.set_xlabel("Año de la concesión")
    a2.set_title("(b) Estado real, decodificación HMM y clasificación directa", pad=3)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, handlelength=1.6, columnspacing=1.0)
    guardar(fig, ruta)
    return {"tramo": n, "aciertos_ingenua": int((O[n] == X[n]).sum()), "aciertos_viterbi": int((Xv[n] == X[n]).sum()), "T": T}


# ---------------------------------------------------------------------- #
# Figura 4: comparación de escenarios
# ---------------------------------------------------------------------- #
def fig_escenarios(df: pd.DataFrame, nombres: dict, ruta: Path) -> None:
    ids = list(dict.fromkeys(df["escenario"]))
    fig, ax = plt.subplots(figsize=(ANCHO, 3.3))
    desplaz = [-0.24, 0.0, 0.24]
    rng = np.random.default_rng(0)  # solo para el jitter visual
    for k, (col, etiqueta, color) in enumerate(METODOS):
        medias, li, ls = [], [], []
        for i, e in enumerate(ids):
            v = df.loc[df["escenario"] == e, col].to_numpy()
            r = resumen(v)
            medias.append(r["media"])
            li.append(r["media"] - r["ic_inf"])
            ls.append(r["ic_sup"] - r["media"])
            x = i + desplaz[k] + rng.uniform(-0.06, 0.06, v.size)
            ax.scatter(x, v, s=5, color=color, alpha=0.22, lw=0, zorder=2)
        ax.errorbar(np.arange(len(ids)) + desplaz[k], medias, yerr=[li, ls], fmt="o", ms=5.5, color=color,
                    mec=FONDO, mew=1.2, elinewidth=1.4, capsize=0, zorder=4)
    for i, e in enumerate(ids):
        sub = df[df["escenario"] == e]
        g = 100 * (sub["exactitud_bw"].mean() - sub["exactitud_ingenua"].mean())
        ax.text(i, sub[["exactitud_viterbi", "exactitud_bw"]].max().max() + 0.012, f"+{g:.1f} pp",
                ha="center", fontsize=8, color=TINTA, fontweight="semibold")
    import textwrap
    ax.set_xticks(range(len(ids)), [f"{e}\n{textwrap.fill(nombres[e], 13)}" for e in ids], fontsize=8)
    ax.set_ylabel("Exactitud de clasificación del estado")
    _pct(ax)
    ax.set_ylim(0.68, 1.01)
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=6, color=c, mec=FONDO, label=t) for _, t, c in METODOS],
              loc="lower left", ncol=1, title="Media e IC 95 % (puntos claros: réplicas)", title_fontsize=7.8,
              alignment="left")
    guardar(fig, ruta)


# ---------------------------------------------------------------------- #
# Figura 5: sensibilidad a la matriz de emisión (ruido) y al tamaño de muestra
# ---------------------------------------------------------------------- #
def _banda(ax, x, sub, col, color, etiqueta, marcador="o"):
    g = sub.groupby(x)[col]
    xs = np.array(sorted(sub[x].unique()))
    res = [resumen(g.get_group(v)) for v in xs]
    m = np.array([r["media"] for r in res])
    lo = np.array([r["ic_inf"] for r in res])
    hi = np.array([r["ic_sup"] for r in res])
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(xs, m, color=color, lw=1.8, label=etiqueta)
    ax.plot(xs, m, marcador, ms=4.5, color=color, mec=FONDO, mew=1.0)
    return xs, m


def fig_sensibilidad_ruido(df_sigma: pd.DataFrame, df_tam: pd.DataFrame, ruta: Path) -> None:
    fig = plt.figure(figsize=(ANCHO, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    for col, etiqueta, color in METODOS:
        _banda(ax, "sigma", df_sigma, col, color, etiqueta)
    for s, txt in ((0.15, "perfilómetro (E1)"), (0.50, "bajo costo (E4)")):
        ax.axvline(s, color=EJE, lw=1.0, zorder=0)
        ax.text(s + 0.012, 0.565, txt, fontsize=7.6, color=TINTA2, va="bottom")
    ax.set_xlabel("Desviación estándar del error de medición σ (m/km)")
    ax.set_ylabel("Exactitud")
    _pct(ax)
    ax.set_ylim(0.55, 1.005)
    R = int(df_sigma.groupby("sigma").size().iloc[0])
    ax.set_title(f"(a) Exactitud de clasificación vs. ruido de medición (media e IC 95 %, R = {R} por punto)", pad=3)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.09))
    for k, (col, titulo) in enumerate((("mae_A", "(b) Error de Â (MAE) vs. N"), ("mae_B", "(c) Error de B̂ (MAE) vs. N"))):
        axk = fig.add_subplot(gs[1, k])
        for e, color, etq in (("E1", AZUL, "E1 (σ = 0.15)"), ("E4", NARANJA, "E4 (σ = 0.50)")):
            xs, m = _banda(axk, "N", df_tam[df_tam["escenario"] == e], col, color, etq)
        ref = m[0] * np.sqrt(xs[0] / xs)
        axk.plot(xs, ref, color=TENUE, lw=1.0, label=r"referencia $\propto 1/\sqrt{N}$")
        axk.set_xscale("log")
        axk.set_yscale("log")
        axk.set_xticks(xs, [str(int(v)) for v in xs])
        axk.minorticks_off()
        axk.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        axk.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axk.set_xlabel("Número de tramos N (T = 25 años)")
        axk.set_title(titulo, pad=3)
        if k == 0:
            axk.set_ylabel("Error absoluto medio")
            axk.legend(loc="lower left", fontsize=7.6)
    guardar(fig, ruta)


# ---------------------------------------------------------------------- #
# Figura 6: sensibilidad a pi y a A
# ---------------------------------------------------------------------- #
def fig_sensibilidad_pi_A(an: pd.DataFrame, sim: pd.DataFrame, oat: pd.DataFrame, pi_est: np.ndarray,
                          lam2: float, ruta: Path, etiquetas_param: dict) -> None:
    fig = plt.figure(figsize=(ANCHO, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.3)
    colores = {"nueva": AZUL, "envejecida": NARANJA, "deteriorada": AQUA}
    nombres = {"nueva": "π₀ red nueva", "envejecida": "π₀ red envejecida (base)", "deteriorada": "π₀ red deteriorada"}
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    pdef_est = pi_est[2:].sum()
    for nombre, color in colores.items():
        s = sim[sim["pi0"] == nombre]
        a1.fill_between(s["anio"], s["ic_inf"], s["ic_sup"], color=color, alpha=0.18, lw=0)
        a = an[an["pi0"] == nombre]
        a1.plot(a["anio"], a["pdef"], color=color, lw=1.8, label=nombres[nombre])
        cada5 = s[s["anio"] % 5 == 0]
        a1.plot(cada5["anio"], cada5["media"], "o", ms=4.2, mfc=FONDO, mec=color, mew=1.2, zorder=4)
        a2.semilogy(a["anio"], np.maximum(a["tv"], 1e-12), color=color, lw=1.8)
    a1.axhline(pdef_est, color=TINTA2, lw=1.0)
    a1.text(40, pdef_est + 0.045, f"π* deficiente = {100 * pdef_est:.1f} %", ha="right", fontsize=7.4, color=TINTA2)
    a1.set_xlabel("Año")
    a1.set_ylabel("Red en estado Malo o Crítico")
    _pct(a1)
    a1.set_title("(a) Red deficiente: π₀Aᵗ (línea) y simulación (○)", pad=3, fontsize=8.6)
    a1.legend(loc="lower right", fontsize=7.4)
    a = an[an["pi0"] == "nueva"]
    a2.semilogy(a["anio"], a["cota_lambda2"], color=TENUE, lw=1.0, label=f"referencia |λ₂|ᵗ, |λ₂| = {lam2:.3f}")
    a2.legend(loc="upper right", fontsize=7.2)
    a2.axhline(0.01, color=EJE, lw=1.0)
    a2.text(40, 0.0125, "TV = 0.01", fontsize=7.2, color=TINTA2, ha="right")
    a2.set_ylim(1e-4, 1.2)
    a2.set_xlabel("Año")
    a2.set_ylabel("Distancia de variación total a π*")
    a2.set_title("(b) Distancia TV a π* (escala log)", pad=3, fontsize=8.6)
    a3 = fig.add_subplot(gs[1, :])
    o = oat.sort_values("rango_pdef_pp")
    yb = np.arange(len(o))
    base = pdef_est
    a3.barh(yb + 0.18, 100 * (o["pdef_menos"] - base), height=0.34, color=AZUL, label="parámetro − 20 %")
    a3.barh(yb - 0.18, 100 * (o["pdef_mas"] - base), height=0.34, color=NARANJA, label="parámetro + 20 %")
    a3.axvline(0, color=TINTA2, lw=0.9)
    a3.set_yticks(yb, [etiquetas_param[p] for p in o["parametro"]], fontsize=8)
    a3.set_xlabel("Cambio en la proporción estacionaria de red deficiente (puntos porcentuales)")
    a3.set_title(f"(c) Sensibilidad local (OAT) de π* deficiente = {100 * base:.1f} % a los parámetros de A", pad=3,
                 fontsize=8.6)
    a3.grid(axis="y", visible=False)
    a3.legend(loc="lower right")
    guardar(fig, ruta)


# ---------------------------------------------------------------------- #
# Figura A1: convergencia de Baum-Welch con distintas inicializaciones
# ---------------------------------------------------------------------- #
def fig_reinicios(curvas: pd.DataFrame, filas: pd.DataFrame, logver_real: float, n_obs: int, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(ANCHO, 3.0))
    mejor = filas["logver_final"].max()
    for nombre, c in curvas.groupby("inicializacion"):
        fin = filas.loc[filas["inicializacion"] == nombre, "logver_final"].iloc[0]
        local = (mejor - fin) > 1e-3 * abs(mejor)
        color = AZUL if nombre == "estructurada" else (NARANJA if local else TENUE)
        lw = 2.0 if nombre == "estructurada" else 0.9
        ax.plot(c["iteracion"] + 1, c["logver"] / n_obs, color=color, lw=lw, alpha=1 if nombre == "estructurada" else 0.75,
                zorder=3 if nombre == "estructurada" else 2)
    ax.axhline(logver_real / n_obs, color=TINTA2, lw=1.0)
    ax.text(1.05, logver_real / n_obs + 0.004, "log P(O | λ verdadero) / n", fontsize=7.6, color=TINTA2)
    ax.set_xscale("log")
    ax.set_xlabel("Iteración de Baum–Welch (escala log)")
    ax.set_ylabel("Log-verosimilitud por observación")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=AZUL, lw=2, label="Inicialización estructurada"),
                       Line2D([], [], color=TENUE, lw=1, label="Aleatoria → máximo global"),
                       Line2D([], [], color=NARANJA, lw=1, label="Aleatoria → máximo local")], loc="lower right")
    ymin = max(curvas["logver"].min() / n_obs, logver_real / n_obs - 0.35)
    ax.set_ylim(ymin, logver_real / n_obs + 0.03)
    guardar(fig, ruta)
