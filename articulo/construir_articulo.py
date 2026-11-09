"""
Genera articulo/articulo.docx a partir de articulo/articulo.md y de los resultados en resultados/.

- Las cifras del texto se escriben como §clave§ en el Markdown y se rellenan con valores
  calculados aquí a partir de los CSV de resultados (el documento es reproducible).
- Las ecuaciones LaTeX ($...$ y $$...$$) se convierten a ecuaciones nativas de Word (OMML)
  con latex2mathml + la hoja XSLT MML2OMML de Microsoft Office.

Uso:  python articulo/construir_articulo.py
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from latex2mathml.converter import convert as latex_a_mathml
from lxml import etree

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from hmm_pavimento.modelo import cargar_config, escenarios  # noqa: E402

RES = RAIZ / "resultados"
TAB = RES / "tablas"
FIG = RES / "figuras"
FUENTE = RAIZ / "articulo" / "articulo.md"
SALIDA = RAIZ / "articulo" / "articulo.docx"
XSL_CANDIDATOS = [
    Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL"),
]
LETRA, LETRA_MATE = "Times New Roman", "Cambria Math"
ANCHO_TEXTO_CM = 16.59  # carta (21.59 cm) - 2 x 2.5 cm
GRIS_CABECERA = "E7E6E1"


# ---------------------------------------------------------------------- #
# Ecuaciones: LaTeX -> MathML -> OMML
# ---------------------------------------------------------------------- #
class Mate:
    def __init__(self):
        xsl = next((p for p in XSL_CANDIDATOS if p.exists()), None)
        self.xslt = etree.XSLT(etree.parse(str(xsl))) if xsl else None
        if self.xslt is None:
            warnings.warn("No se encontró MML2OMML.XSL (Office); las ecuaciones se escribirán como texto LaTeX.")

    def omml(self, latex: str, bloque: bool = False):
        """Ecuación OMML: en línea (m:oMath) o en modo presentación (m:oMathPara) si bloque=True."""
        # MML2OMML descarta <mspace>: los espacios se expresan como texto normal
        latex = latex.replace(r"\quad", r"\text{    }").replace(r"\;", r"\text{ }")
        mathml = latex_a_mathml(latex, display="block" if bloque else "inline")
        resto = re.findall(r"\\[a-zA-Z]+", mathml)
        if resto:
            warnings.warn(f"Comandos LaTeX no convertidos {resto} en: {latex}")
        if self.xslt is None:
            return None
        raiz = self.xslt(etree.fromstring(mathml.encode("utf-8"))).getroot()
        if bloque:
            if not raiz.tag.endswith("oMathPara"):
                para = OxmlElement("m:oMathPara")
                para.append(raiz)
                raiz = para
            return raiz
        if raiz.tag.endswith("oMathPara"):
            raiz = raiz.find(qn("m:oMath"))
        return raiz


MATE = Mate()


# ---------------------------------------------------------------------- #
# Utilidades de formato
# ---------------------------------------------------------------------- #
def fijar_fuente(run, tam=None, negrita=None, cursiva=None, color=None, fuente=LETRA):
    run.font.name = fuente
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for att in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(att), fuente)
    if tam is not None:
        run.font.size = Pt(tam)
    if negrita is not None:
        run.bold = negrita
    if cursiva is not None:
        run.italic = cursiva
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def formato_parrafo(p, alin=WD_ALIGN_PARAGRAPH.JUSTIFY, interlineado=1.5, antes=0, despues=6, sangria=None,
                    mantener=False):
    pf = p.paragraph_format
    p.alignment = alin
    pf.line_spacing = interlineado
    pf.space_before = Pt(antes)
    pf.space_after = Pt(despues)
    if sangria is not None:
        pf.first_line_indent = Cm(sangria)
    pf.keep_with_next = mantener
    pf.widow_control = True


PATRON_MATE = re.compile(r"\$([^$]+)\$")
PATRON_ESTILO = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\{\{[^}]+\}\}|`[^`]+`)")


def texto_enriquecido(p, texto: str, tam=11, color=None, base_negrita=False, base_cursiva=False):
    """Agrega texto con **negrita**, *cursiva*, `código`, {{marcador}} y $matemática$ en línea."""
    partes = PATRON_MATE.split(texto)
    for i, parte in enumerate(partes):
        if not parte:
            continue
        if i % 2 == 1:
            elem = MATE.omml(parte)
            if elem is None:
                r = p.add_run(parte)
                fijar_fuente(r, tam, cursiva=True)
            else:
                if tam != 11:  # en tablas y notas, la ecuación hereda el tamaño del texto
                    for mr in elem.iter(qn("m:r")):
                        wrpr = OxmlElement("w:rPr")
                        rf = OxmlElement("w:rFonts")
                        for att in ("w:ascii", "w:hAnsi"):
                            rf.set(qn(att), LETRA_MATE)
                        sz = OxmlElement("w:sz")
                        sz.set(qn("w:val"), str(int(round(tam * 2))))
                        wrpr.append(rf)
                        wrpr.append(sz)
                        mrpr = mr.find(qn("m:rPr"))
                        mr.insert(0 if mrpr is None else 1, wrpr)
                p._p.append(elem)
            continue
        for trozo in PATRON_ESTILO.split(parte):
            if not trozo:
                continue
            negrita, cursiva, codigo, marcador = base_negrita, base_cursiva, False, False
            if trozo.startswith("**") and trozo.endswith("**"):
                trozo, negrita = trozo[2:-2], True
            elif trozo.startswith("{{") and trozo.endswith("}}"):
                trozo, marcador = trozo[2:-2], True
            elif trozo.startswith("`") and trozo.endswith("`"):
                trozo, codigo = trozo[1:-1], True
            elif trozo.startswith("*") and trozo.endswith("*") and len(trozo) > 2:
                trozo, cursiva = trozo[1:-1], True
            r = p.add_run(trozo)
            fijar_fuente(r, tam * (0.92 if codigo else 1), negrita, cursiva, color, fuente="Consolas" if codigo else LETRA)
            if marcador:
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def borde_celda(celda, **lados):
    tcPr = celda._tc.get_or_add_tcPr()
    bordes = tcPr.find(qn("w:tcBorders"))
    if bordes is None:
        bordes = OxmlElement("w:tcBorders")
        tcPr.append(bordes)
    for lado, (tam, color) in lados.items():
        el = OxmlElement(f"w:{lado}")
        el.set(qn("w:val"), "single" if tam > 0 else "nil")
        el.set(qn("w:sz"), str(tam))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        bordes.append(el)


def sombrear(celda, color_hex):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def margenes_celda(tabla, sup=40, inf=40, izq=70, der=70):
    tblPr = tabla._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for lado, v in (("top", sup), ("bottom", inf), ("left", izq), ("right", der)):
        el = OxmlElement(f"w:{lado}")
        el.set(qn("w:w"), str(v))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def campo_pagina(p):
    r = p.add_run()
    fijar_fuente(r, 9, color="52514E")
    for tipo, texto in (("begin", None), (None, "PAGE"), ("separate", None), (None, "1"), ("end", None)):
        if tipo:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), tipo)
            r._r.append(fc)
        elif texto == "PAGE":
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = " PAGE "
            r._r.append(it)
        else:
            t = OxmlElement("w:t")
            t.text = texto
            r._r.append(t)


# ---------------------------------------------------------------------- #
# Citas IEEE: [@clave; @clave2] -> [n1], [n2] numeradas por orden de aparición
# ---------------------------------------------------------------------- #
CITAS: dict[str, int] = {}
PATRON_CITA = re.compile(r"\[(@[^\]]+)\]")


def claves_en(texto: str) -> list[str]:
    return [k.strip().lstrip("@") for grupo in PATRON_CITA.findall(texto) for k in grupo.split(";")]


def citar(texto: str) -> str:
    def reemplazo(mo):
        claves = [k.strip().lstrip("@") for k in mo.group(1).split(";")]
        faltan = [k for k in claves if k not in CITAS]
        if faltan:
            raise KeyError(f"Cita sin referencia: {faltan}")
        nums = sorted({CITAS[k] for k in claves})
        partes, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            partes += [f"[{nums[i]}]–[{nums[j]}]"] if j - i >= 2 else [f"[{n}]" for n in nums[i:j + 1]]
            i = j + 1
        return ", ".join(partes)
    return PATRON_CITA.sub(reemplazo, texto)


def numerar_citas(cuerpo: str) -> None:
    CITAS.clear()
    for linea in cuerpo.splitlines():
        claves = []
        if linea.startswith("[[TABLA:estado_arte"):
            for fila in FILAS_ESTADO_ARTE:
                for celda in fila:
                    claves += claves_en(celda)
        else:
            claves = claves_en(linea)
        for k in claves:
            CITAS.setdefault(k, len(CITAS) + 1)


# ---------------------------------------------------------------------- #
# Valores numéricos (todas las cifras del texto salen de aquí)
# ---------------------------------------------------------------------- #
def f1(x):
    return f"{x:.1f}".replace("-", "−")


def f2(x):
    return f"{x:.2f}".replace("-", "−")


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def p_txt(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def valores() -> dict:
    cfg = cargar_config()
    V: dict[str, str] = {}
    res = pd.read_csv(TAB / "resumen_escenarios.csv")
    an = pd.read_csv(TAB / "propiedades_analiticas.csv")
    rep = pd.read_csv(TAB / "replicas_escenarios.csv")
    pru = pd.read_csv(TAB / "pruebas_pareadas.csv")
    kw = json.loads((TAB / "kruskal_wallis.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(TAB / "comparacion_escenarios.csv")

    def m(e, metrica, campo="media"):
        return float(res.loc[(res["escenario"] == e) & (res["metrica"] == metrica), campo].iloc[0])

    for e in [x["id"] for x in cfg["escenarios"]]:
        V[f"{e}_dir"] = f1(100 * m(e, "exactitud_ingenua"))
        V[f"{e}_ver"] = f1(100 * m(e, "exactitud_viterbi"))
        V[f"{e}_bw"] = f1(100 * m(e, "exactitud_bw"))
        V[f"{e}_bw_ic"] = f"[{f1(100 * m(e, 'exactitud_bw', 'ic_inf'))}, {f1(100 * m(e, 'exactitud_bw', 'ic_sup'))}]"
        V[f"{e}_dir_ic"] = f"[{f1(100 * m(e, 'exactitud_ingenua', 'ic_inf'))}, {f1(100 * m(e, 'exactitud_ingenua', 'ic_sup'))}]"
        V[f"{e}_gan"] = f1(100 * (m(e, "exactitud_bw") - m(e, "exactitud_ingenua")))
        V[f"{e}_brecha"] = f1(100 * (m(e, "exactitud_viterbi") - m(e, "exactitud_bw")))
        V[f"{e}_err_dir"] = f1(100 * (1 - m(e, "exactitud_ingenua")))
        V[f"{e}_err_bw"] = f1(100 * (1 - m(e, "exactitud_bw")))
        V[f"{e}_red_err"] = f1(100 * (1 - (1 - m(e, "exactitud_bw")) / (1 - m(e, "exactitud_ingenua"))))
        V[f"{e}_sens_dir"] = f1(100 * m(e, "sens_def_ingenua"))
        V[f"{e}_sens_bw"] = f1(100 * m(e, "sens_def_bw"))
        V[f"{e}_prec_dir"] = f1(100 * m(e, "prec_def_ingenua"))
        V[f"{e}_prec_bw"] = f1(100 * m(e, "prec_def_bw"))
        V[f"{e}_maeA"] = f3(m(e, "mae_A"))
        V[f"{e}_maeB"] = f3(m(e, "mae_B"))
        V[f"{e}_iter"] = f"{m(e, 'iteraciones_bw'):.0f}"
        V[f"{e}_kappa_dir"] = f2(m(e, "kappa_ingenua"))
        V[f"{e}_kappa_ver"] = f2(m(e, "kappa_viterbi"))
        V[f"{e}_errpdef_dir"] = f1(100 * m(e, "error_pdef_ingenua"))
        V[f"{e}_errpdef_hmm"] = f1(100 * m(e, "error_pdef_hmm"))
        a = an[an["escenario"] == e].iloc[0]
        V[f"{e}_pdef"] = f1(100 * a["pdef_estacionaria"])
        V[f"{e}_mfpt"] = f1(a["mfpt_Bueno_a_deficiente"])
        V[f"{e}_lam2"] = f3(a["lambda2"])
        V[f"{e}_tmix"] = f"{int(a['t_mezcla_0.01'])}"
        V[f"{e}_interv"] = f1(a["intervenciones_100km_anio"])
        V[f"{e}_diagB"] = f2(a["diag_B_media"])
        V[f"{e}_dir_est"] = f1(100 * a["exactitud_ingenua_estacionaria"])
        sub = pru[(pru["escenario"] == e) & (pru["comparacion"] == "exactitud_bw - exactitud_ingenua")].iloc[0]
        V[f"{e}_dz"] = f1(sub["d_z"])
        V[f"{e}_p"] = p_txt(sub["p_holm"])
    V["E5_gauss"] = f1(100 * m("E5", "exactitud_viterbi_B_gauss"))
    V["E5_costo_gauss"] = f1(100 * (m("E5", "exactitud_viterbi") - m("E5", "exactitud_viterbi_B_gauss")))
    V["R"] = f"{cfg['simulacion']['R_replicas']}"
    V["N"] = f"{cfg['simulacion']['N_segmentos']}"
    V["T"] = f"{cfg['simulacion']['T_anios']}"
    V["NT"] = f"{cfg['simulacion']['N_segmentos'] * cfg['simulacion']['T_anios']:,}".replace(",", ".")
    V["semilla"] = f"{cfg['semilla_maestra']}"
    V["kw_H"] = f1(kw["exactitud_bw"]["H"])
    V["kw_p"] = p_txt(kw["exactitud_bw"]["p_valor"])
    V["kw_gl"] = f"{kw['exactitud_bw']['gl']}"
    V["n_replicas_total"] = f"{len(rep):,}".replace(",", ".")
    V["pmax_holm"] = p_txt(pru.loc[pru["comparacion"] == "exactitud_bw - exactitud_ingenua", "p_holm"].max())
    V["conv_bw"] = f1(100 * rep["convergio_bw"].mean())
    V["perm_bw"] = f"{int(rep['permutado_bw'].sum())}"
    c = comp[comp["metrica"] == "exactitud_bw"].set_index("escenario")
    for e in c.index:
        V[f"{e}_A12"] = f2(c.loc[e, "A12"])
        V[f"{e}_pMW"] = p_txt(c.loc[e, "p_holm"])
    V["rango_gan_bajo"] = f1(min(100 * (m(e, "exactitud_bw") - m(e, "exactitud_ingenua")) for e in ("E1", "E2", "E3")))
    V["rango_gan_alto"] = f1(max(100 * (m(e, "exactitud_bw") - m(e, "exactitud_ingenua")) for e in ("E1", "E2", "E3")))
    V["max_brecha"] = f1(max(100 * (m(e, "exactitud_viterbi") - m(e, "exactitud_bw")) for e in ("E1", "E2", "E3", "E4", "E5")))
    todos = [x["id"] for x in cfg["escenarios"]]
    iters = [m(e, "iteraciones_bw") for e in todos]
    V["iter_min"], V["iter_max"] = f"{min(iters):.0f}", f"{max(iters):.0f}"
    dzs = pru.loc[pru["comparacion"] == "exactitud_bw - exactitud_ingenua", "d_z"]
    V["dz_min"], V["dz_max"] = f1(dzs.min()), f1(dzs.max())
    mf = an.set_index("escenario")["mfpt_Bueno_a_deficiente"]
    V["dif_mfpt"] = f1(mf["E1"] - mf["E2"])
    V["pdef_dif_E2"] = f1(100 * (an.set_index("escenario").loc["E2", "pdef_estacionaria"] - an.set_index("escenario").loc["E1", "pdef_estacionaria"]))
    V["pdef_dif_E3"] = f1(100 * (an.set_index("escenario").loc["E1", "pdef_estacionaria"] - an.set_index("escenario").loc["E3", "pdef_estacionaria"]))
    semi = res[res["metrica"].str.startswith("exactitud")]["semiamplitud"].max()
    V["semi_max"] = f2(100 * semi)

    # Barrido de sigma
    sg = pd.read_csv(TAB / "resumen_barrido_sigma.csv")

    def s(sig, met):
        return float(sg.loc[(np.isclose(sg["sigma"], sig)) & (sg["metrica"] == met), "media"].iloc[0])
    sigmas = sorted(sg["sigma"].unique())
    V["sig_min"], V["sig_max"] = f2(sigmas[0]), f2(sigmas[-1])
    V["sig_dir_min"], V["sig_dir_max"] = f1(100 * s(sigmas[0], "exactitud_ingenua")), f1(100 * s(sigmas[-1], "exactitud_ingenua"))
    V["sig_bw_min"], V["sig_bw_max"] = f1(100 * s(sigmas[0], "exactitud_bw")), f1(100 * s(sigmas[-1], "exactitud_bw"))
    V["sig_ver_max"] = f1(100 * s(sigmas[-1], "exactitud_viterbi"))
    gan = [(sig, 100 * (s(sig, "exactitud_bw") - s(sig, "exactitud_ingenua"))) for sig in sigmas]
    sig_gmax, gmax = max(gan, key=lambda t: t[1])
    V["sig_gan_max"], V["gan_max"] = f2(sig_gmax), f1(gmax)
    V["sig_gan_min_val"] = f1(gan[0][1])
    dirs = np.array([s(sig, "exactitud_ingenua") for sig in sigmas])
    bws = np.array([s(sig, "exactitud_bw") for sig in sigmas])
    V["sig_dir_90"] = f2(np.interp(-0.90, -dirs, sigmas))
    V["sig_bw_90"] = f2(np.interp(-0.90, -bws, sigmas))
    V["R_sigma"] = f"{cfg['sensibilidad']['R_sigma']}"

    # Curva de aprendizaje
    tm = pd.read_csv(TAB / "tamano_muestra.csv")
    for e in ("E1", "E4"):
        g = tm[tm["escenario"] == e].groupby("N")[["mae_A", "mae_B"]].mean()
        for col in ("mae_A", "mae_B"):
            pend = np.polyfit(np.log(g.index.to_numpy(float)), np.log(g[col].to_numpy()), 1)[0]
            V[f"{e}_{col}_pend"] = f2(pend)
            V[f"{e}_{col}_N25"] = f3(g[col].iloc[0])
            V[f"{e}_{col}_N400"] = f3(g[col].iloc[-1])
    V["R_tamano"] = f"{cfg['sensibilidad']['R_tamano']}"

    # Sensibilidad a pi
    pa = pd.read_csv(TAB / "pi_analitico.csv")
    for nombre in cfg["distribuciones_iniciales"]:
        sub = pa[pa["pi0"] == nombre]
        V[f"tmix_{nombre}"] = f"{int(sub.loc[sub['tv'] <= 0.01, 'anio'].min())}"
        V[f"pdef0_{nombre}"] = f1(100 * sub["pdef"].iloc[0])
        V[f"pdef10_{nombre}"] = f1(100 * sub.loc[sub["anio"] == 10, "pdef"].iloc[0])
        V[f"pdef25_{nombre}"] = f1(100 * sub.loc[sub["anio"] == 25, "pdef"].iloc[0])

    # Sensibilidad de A
    oat = pd.read_csv(TAB / "sensibilidad_A_oat.csv")
    glob = pd.read_csv(TAB / "sensibilidad_A_global_src.csv").set_index("parametro")
    gres = json.loads((TAB / "sensibilidad_A_global_resumen.json").read_text(encoding="utf-8"))
    for _, fila in oat.iterrows():
        V[f"oat_{fila['parametro']}"] = f1(fila["rango_pdef_pp"])
        V[f"oat_{fila['parametro']}_menos"] = f1(100 * (fila["pdef_menos"] - gres["pdef_base"]))
        V[f"oat_{fila['parametro']}_mas"] = f1(100 * (fila["pdef_mas"] - gres["pdef_base"]))
    for p_ in glob.index:
        V[f"src_{p_}"] = f2(glob.loc[p_, "SRC_pdef"])
        V[f"srcm_{p_}"] = f2(glob.loc[p_, "SRC_mfpt"])
    V["R2_pdef"] = f3(gres["R2_pdef"])
    V["R2_mfpt"] = f3(gres["R2_mfpt"])
    V["glob_pdef_lo"], V["glob_pdef_hi"] = f1(100 * gres["pdef_p2.5"]), f1(100 * gres["pdef_p97.5"])
    V["glob_mfpt_lo"], V["glob_mfpt_hi"] = f1(gres["mfpt_p2.5"]), f1(gres["mfpt_p97.5"])
    V["n_glob"] = f"{cfg['sensibilidad']['muestras_globales_A']:,}".replace(",", ".")

    # Reinicios de Baum-Welch
    rb = pd.read_csv(TAB / "reinicios_bw.csv")
    mejor = rb["logver_final"].max()
    glob_ok = (mejor - rb["logver_final"]) <= 1e-3 * abs(mejor)
    aleat = rb["inicializacion"] != "estructurada"
    V["n_reinicios"] = f"{int(aleat.sum())}"
    V["n_global"] = f"{int((glob_ok & aleat).sum())}"
    V["n_local"] = f"{int((~glob_ok & aleat).sum())}"
    V["n_perm"] = f"{int((rb['permutado'] & aleat).sum())}"
    V["estr_global"] = "sí" if bool(glob_ok[~aleat].iloc[0]) else "no"
    V["estr_iter"] = f"{int(rb.loc[~aleat, 'iteraciones'].iloc[0])}"
    V["acc_local_min"] = f1(100 * rb.loc[~glob_ok, "exactitud_bw"].min()) if (~glob_ok).any() else "—"
    V["acc_global"] = f1(100 * rb.loc[glob_ok, "exactitud_bw"].mean())

    # Reproducibilidad y entorno
    rp = json.loads((RES / "reproducibilidad.json").read_text(encoding="utf-8"))
    V["hash1"] = rp["sha256_corrida_1"][:16]
    V["hash2"] = rp["sha256_corrida_2"][:16]
    V["hash3"] = rp["sha256_semilla_alternativa"][:16]
    V["repro_R"] = f"{rp['R']}"
    V["repro_igual"] = "idénticas" if rp["identicas_misma_semilla"] else "distintas"
    V["repro_solapan"] = "se solapan" if all(v["ic_se_solapan"] for v in rp["comparacion_estadistica"].values()) else "no se solapan en todos los casos"
    ent = json.loads((RES / "entorno.json").read_text(encoding="utf-8"))
    V["py"] = ent["python"]
    for k_, v_ in ent["paquetes"].items():
        V[f"v_{k_}"] = v_
    V["so"] = ent["sistema"]
    esp = pd.read_csv(TAB / "resumen_transiciones_espurias.csv")
    for _, fila in esp.iterrows():
        V[f"{fila['escenario']}_{fila['metrica']}"] = f1(100 * fila["media"])
    info3 = json.loads((TAB / "figB1_tramo_ilustrativo.json").read_text(encoding="utf-8"))
    V["f3_dir"], V["f3_hmm"], V["f3_T"] = str(info3["aciertos_ingenua"]), str(info3["aciertos_viterbi"]), str(info3["T"])
    V["n_pruebas"] = "17"
    return V


# ---------------------------------------------------------------------- #
# Tablas
# ---------------------------------------------------------------------- #
def tabla_generica(doc, encabezado, filas, anchos_cm, titulo, nota=None, tam=8.5, alin_cols=None, filas_grupo=()):
    cap = doc.add_paragraph()
    formato_parrafo(cap, WD_ALIGN_PARAGRAPH.LEFT, 1.0, antes=6, despues=3, mantener=True)
    texto_enriquecido(cap, titulo, tam=9.5)
    t = doc.add_table(rows=1 + len(filas), cols=len(encabezado))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    margenes_celda(t)
    for j, ancho in enumerate(anchos_cm):
        for celda in t.columns[j].cells:
            celda.width = Cm(ancho)
    total = len(filas)
    for i, fila in enumerate([encabezado] + filas):
        for j, val in enumerate(fila):
            celda = t.cell(i, j)
            celda.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            celda.text = ""
            lineas = str(val).split("\n")
            for k, linea in enumerate(lineas):
                p = celda.paragraphs[0] if k == 0 else celda.add_paragraph()
                alin = WD_ALIGN_PARAGRAPH.LEFT
                if alin_cols and i > 0:
                    alin = {"c": WD_ALIGN_PARAGRAPH.CENTER, "l": WD_ALIGN_PARAGRAPH.LEFT, "r": WD_ALIGN_PARAGRAPH.RIGHT}[alin_cols[j]]
                if i == 0:
                    alin = WD_ALIGN_PARAGRAPH.CENTER if (alin_cols and alin_cols[j] != "l") else WD_ALIGN_PARAGRAPH.LEFT
                # keep_with_next en todas las filas salvo la última: Word mantiene la tabla en una página
                formato_parrafo(p, alin, 1.0, 0, 0, mantener=i < total)
                texto_enriquecido(p, linea, tam=tam, base_negrita=(i == 0) or (i - 1 in filas_grupo and j == 0))
            if i == 0:
                sombrear(celda, GRIS_CABECERA)
            arriba = (12, "0B0B0B") if i == 0 else (0, "auto")
            abajo = (6, "0B0B0B") if i == 0 else ((12, "0B0B0B") if i == total else (2, "C3C2B7"))
            borde_celda(celda, top=arriba, bottom=abajo, left=(0, "auto"), right=(0, "auto"))
        # evitar que la fila se divida entre páginas
        trPr = t.rows[i]._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
        if i == 0:
            th = OxmlElement("w:tblHeader")
            trPr.append(th)
    if nota:
        pn = doc.add_paragraph()
        formato_parrafo(pn, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.0, antes=2, despues=8)
        texto_enriquecido(pn, nota, tam=8.5, color="3A3936")
    else:
        espacio = doc.add_paragraph()
        formato_parrafo(espacio, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 4)
    return t


FILAS_ESTADO_ARTE = [
    ["Kobayashi *et al.* (2012) [@kobayashi2012]", "Infraestructura vial; datos de inspección", "HMM de deterioro",
     "Trata sesgos de selección y errores de las inspecciones como fenómeno latente", "Justifica modelar la condición real como estado oculto"],
    ["Lethanh y Adey (2013) [@lethanh2013]", "Pavimentos; caso real", "HMM exponencial (EHMM)",
     "La condición es latente y el indicador medido es su señal observable", "Base conceptual de la emisión B"],
    ["Alonso-Solorzano *et al.* (2023) [@alonso2023]", "Red estatal de España; IRI", "Cadena de Markov visible (TPM)",
     "Matrices anuales estimables con datos bianuales; sin mantenimiento", "Estructura de A por niveles de deterioro"],
    ["Tran *et al.* (2023) [@tran2023]", "Red de carreteras; fisuras, ahuellamiento, rugosidad", "Cadena de Markov visible",
     "Pronóstico de la red y del presupuesto de mantenimiento", "Uso de π₀Aᵗ para indicadores de red"],
    ["Isradi *et al.* (2024) [@isradi2024]", "Vías de Indonesia; IRI 2018–2019", "Cadena de Markov con M&R",
     "Incorpora mejoras por mantenimiento y rehabilitación en la TPM", "Transiciones de rehabilitación en A"],
    ["Shakya *et al.* (2025) [@shakya2025]", "Carreteras nacionales de Nepal", "Modelo de riesgo markoviano",
     "Agrega la lluvia monzónica como covariable del deterioro", "Motiva escenarios de A por condiciones externas"],
    ["Sasai *et al.* (2025) [@sasai2025]", "Gestión de pavimentos de autopistas", "Proceso de deterioro continuo multidimensional",
     "Supera la limitación de un único indicador de los modelos de Markov", "Contraste: alternativa continua al HMM discreto"],
    ["Yu *et al.* (2023) [@yu2023]", "Rugosidad con teléfonos inteligentes", "Marco de evaluación estadística",
     "La repetibilidad depende de velocidad, vehículo y montaje; tendencia a sobrestimar", "Justifica el ruido σ de los escenarios E4–E5"],
    ["Gámiz *et al.* (2023) [@gamiz2023a; @gamiz2023b]", "Fiabilidad y mantenimiento", "HMM y HMM con señal markoviana",
     "Medidas de fiabilidad a partir del estado oculto; sensibilidad a señales falsas", "Análogo metodológico en mantenimiento"],
    ["Aggab *et al.* (2022) [@aggab2022]", "Pronóstico de vida útil remanente", "HMM discreto + observador",
     "Vida remanente vía tiempo medio hasta absorción", "Uso de tiempos de primer paso"],
]


def tabla_estado_arte(doc, titulo, nota):
    enc = ["Trabajo", "Dominio y datos", "Enfoque probabilístico", "Aporte principal", "Relación con este estudio"]
    filas = [[citar(c) for c in f] for f in FILAS_ESTADO_ARTE]
    return tabla_generica(doc, enc, filas, [3.0, 3.1, 2.9, 4.0, 3.5], titulo, nota, tam=8)


def tabla_escenarios(doc, titulo, nota):
    cfg = cargar_config()
    enc = ["Escenario", "Qué representa", "Matriz de transición A", "Error de medición (define B)"]
    nombres = {"d_BR": "d_{BR}", "d_BM": "d_{BM}", "d_RM": "d_{RM}", "d_RC": "d_{RC}", "d_MC": "d_{MC}",
               "r_M": "r_M", "r_C": "r_C", "factor_transito": "f"}
    esc = escenarios(cfg)
    base = esc[0].parametros_A
    filas = []
    for e in esc:
        p = e.parametros_A
        if e.id == "E1":
            a_txt = (", ".join(f"${nombres[k]} = {p[k]:g}$" for k in ("d_BR", "d_BM", "d_RM", "d_RC")) + "\n"
                     + ", ".join(f"${nombres[k]} = {p.get(k, 0):g}$" for k in ("d_MC", "r_M", "r_C", "factor_transito")))
        else:
            cambios = [k for k in nombres if abs(p.get(k, 0) - base.get(k, 0)) > 1e-12]
            a_txt = (", ".join(f"${nombres[k]} = {p[k]:g}$" for k in cambios) + "; resto igual a E1") if cambios else "Igual a E1"
        er = e.error
        dist = "normal" if er["distribucion"] == "normal" else f"t de Student, ν = {er['gl']}"
        que = {"E1": "Tránsito medio, política correctiva, perfilómetro", "E2": "Mayor carga de vehículos pesados",
               "E3": "Intervención preventiva del 30 % de los tramos Malos", "E4": "Auscultación con sensor de bajo costo",
               "E5": "Sensor de bajo costo con valores atípicos"}[e.id]
        filas.append([f"{e.id} {e.nombre}", que, a_txt, f"{dist}, σ = {er['sigma']:.2f} m/km"])
    return tabla_generica(doc, enc, filas, [2.7, 4.2, 6.2, 3.4], titulo, nota, tam=8, alin_cols="llll")


def tabla_analitica(doc, titulo, nota):
    an = pd.read_csv(TAB / "propiedades_analiticas.csv")
    enc = ["Escenario", "π* deficiente (%)", "Años de Bueno a deficiente", "Permanencia media B / R / M / C (años)",
           "|λ₂|", "t_mez (años)", "Intervenciones por 100 km-año", "diag(B) media"]
    filas = []
    for _, a in an.iterrows():
        filas.append([f"{a['escenario']} {a['nombre']}", f1(100 * a["pdef_estacionaria"]), f1(a["mfpt_Bueno_a_deficiente"]),
                      " / ".join(f1(a[f"permanencia_{n}"]) for n in ["Bueno", "Regular", "Malo", "Crítico"]),
                      f3(a["lambda2"]), f"{int(a['t_mezcla_0.01'])}", f1(a["intervenciones_100km_anio"]), f2(a["diag_B_media"])])
    return tabla_generica(doc, enc, filas, [3.1, 1.7, 1.8, 3.4, 1.3, 1.4, 2.0, 1.6], titulo, nota, tam=8, alin_cols="lcccccccc")


def tabla_montecarlo(doc, titulo, nota):
    res = pd.read_csv(TAB / "resumen_escenarios.csv")
    ids = list(dict.fromkeys(res["escenario"]))
    metricas = [
        ("exactitud_ingenua", "Exactitud, clasificación directa (%)", 100, f1),
        ("exactitud_viterbi", "Exactitud, HMM con λ verdadero (%)", 100, f1),
        ("exactitud_bw", "Exactitud, HMM con λ̂ Baum–Welch (%)", 100, f1),
        ("sens_def_ingenua", "Sensibilidad tramos deficientes, directa (%)", 100, f1),
        ("sens_def_bw", "Sensibilidad tramos deficientes, HMM λ̂ (%)", 100, f1),
        ("mae_A", "MAE(Â) × 10³", 1000, f1),
        ("mae_B", "MAE(B̂) × 10³", 1000, f1),
    ]
    enc = ["Métrica"] + ids
    filas = []
    for col, nombre, esc, fmt in metricas:
        fila = [nombre]
        for e in ids:
            r = res[(res["escenario"] == e) & (res["metrica"] == col)].iloc[0]
            fila.append(f"{fmt(esc * r['media'])} ± {fmt(esc * r['de'])}\n[{fmt(esc * r['ic_inf'])}, {fmt(esc * r['ic_sup'])}]\n"
                        f"({fmt(esc * r['min'])}–{fmt(esc * r['max'])})")
        filas.append(fila)
    return tabla_generica(doc, enc, filas, [3.9, 2.5, 2.5, 2.5, 2.5, 2.5], titulo, nota, tam=7.8, alin_cols="lccccc")


def tabla_pruebas(doc, titulo, nota):
    pru = pd.read_csv(TAB / "pruebas_pareadas.csv")
    comp = pd.read_csv(TAB / "comparacion_escenarios.csv")
    enc = ["Escenario", "Δ exactitud HMM λ̂ − directa (pp) [IC 95 %]", "p (Wilcoxon, Holm)", "d_z",
           "Δ sensibilidad deficientes (pp)", "A₁₂ vs. E1 (exactitud HMM λ̂)"]
    filas = []
    for e in list(dict.fromkeys(pru["escenario"])):
        a = pru[(pru["escenario"] == e) & (pru["comparacion"] == "exactitud_bw - exactitud_ingenua")].iloc[0]
        b = pru[(pru["escenario"] == e) & (pru["comparacion"] == "sens_def_bw - sens_def_ingenua")].iloc[0]
        c = comp[(comp["metrica"] == "exactitud_bw") & (comp["escenario"] == e)]
        a12 = "—" if c.empty else f"{f2(c['A12'].iloc[0])} (p {p_txt(c['p_holm'].iloc[0])})"
        filas.append([e, f"{f1(100 * a['dif_media'])} [{f1(100 * a['dif_ic_inf'])}, {f1(100 * a['dif_ic_sup'])}]",
                      p_txt(a["p_holm"]), f1(a["d_z"]), f"{f1(100 * b['dif_media'])} [{f1(100 * b['dif_ic_inf'])}, {f1(100 * b['dif_ic_sup'])}]", a12])
    return tabla_generica(doc, enc, filas, [1.8, 3.6, 2.2, 1.4, 3.6, 3.6], titulo, nota, tam=8, alin_cols="cccccc")


def tabla_entorno(doc, titulo, nota, V):
    enc = ["Elemento", "Valor"]
    filas = [
        ["Lenguaje e intérprete", f"Python {V['py']} (CPython, 64 bits)"],
        ["Bibliotecas", f"NumPy {V['v_numpy']}, SciPy {V['v_scipy']}, pandas {V['v_pandas']}, Matplotlib {V['v_matplotlib']}, hmmlearn {V['v_hmmlearn']} (solo validación), pytest {V['v_pytest']}"],
        ["Sistema operativo", V["so"]],
        ["Generador aleatorio", f"PCG64 con SeedSequence([{V['semilla']}, experimento, índice]); un flujo independiente por réplica"],
        ["Verificación del código", f"{V['n_pruebas']} pruebas unitarias superadas (Forward y Viterbi = enumeración exhaustiva; coincidencia con hmmlearn; monotonía de EM)"],
        ["Reejecución con la misma semilla", f"SHA-256 {V['hash1']}… = {V['hash2']}… (resultados {V['repro_igual']})"],
        ["Reejecución con otra semilla", f"SHA-256 {V['hash3']}…; los IC 95 % de las exactitudes {V['repro_solapan']}"],
        ["Ejecución completa", "python ejecutar_todo.py  (parámetros en config/escenarios.json)"],
    ]
    return tabla_generica(doc, enc, filas, [4.2, 12.4], titulo, nota, tam=8.5, alin_cols="ll")


# ---------------------------------------------------------------------- #
# Construcción del documento
# ---------------------------------------------------------------------- #
def configurar(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)  # carta
    for lado in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, lado, Cm(2.5))
    st = doc.styles["Normal"]
    st.font.name = LETRA
    st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), LETRA)
    st.paragraph_format.line_spacing = 1.5
    for nombre, tam, cursiva in (("Heading 1", 12.5, False), ("Heading 2", 11.5, True), ("Heading 3", 11, True)):
        h = doc.styles[nombre]
        h.font.name, h.font.size, h.font.bold, h.font.italic = LETRA, Pt(tam), True, cursiva
        h.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)
        rpr = h.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.insert(0, rf)
        for att in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(att), LETRA)
        for att in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if rf.get(qn(att)) is not None:
                del rf.attrib[qn(att)]
        pf = h.paragraph_format
        pf.space_before, pf.space_after, pf.keep_with_next, pf.line_spacing = Pt(12 if nombre == "Heading 1" else 8), Pt(4), True, 1.15
    # Pie de página con número
    pie = sec.footer.paragraphs[0]
    formato_parrafo(pie, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 0, 0)
    campo_pagina(pie)
    enc = sec.header.paragraphs[0]
    formato_parrafo(enc, WD_ALIGN_PARAGRAPH.RIGHT, 1.0, 0, 0)
    r = enc.add_run("HMM para la condición oculta del pavimento a partir de mediciones ruidosas del IRI")
    fijar_fuente(r, 8, cursiva=True, color="898781")


def construir():
    V = valores()
    texto = FUENTE.read_text(encoding="utf-8")
    texto = re.sub(r"<!--.*?-->", "", texto, flags=re.S)  # comentarios de la fuente

    def sustituir(mo):
        clave = mo.group(1)
        if clave not in V:
            raise KeyError(f"Valor no definido: §{clave}§")
        return V[clave]
    texto = re.sub(r"§([^§]+)§", sustituir, texto)

    # Referencias: numeración IEEE por orden de primera aparición
    cuerpo, refs = texto.split("## Referencias", 1)
    numerar_citas(cuerpo)
    lineas_refs = refs.splitlines()
    definidas = {}
    resto_refs = []
    for ln in lineas_refs:
        mo = re.match(r"^\[@([^\]]+)\]\s*(.*)$", ln)
        if mo:
            definidas[mo.group(1)] = mo.group(2)
        else:
            resto_refs.append(ln)
    sin_definir = [k for k in CITAS if k not in definidas]
    if sin_definir:
        raise KeyError(f"Citas sin entrada en Referencias: {sin_definir}")
    no_citadas = [k for k in definidas if k not in CITAS]
    if no_citadas:
        warnings.warn(f"Referencias no citadas en el texto: {no_citadas}")
    lista = [f"[{n}] {definidas[k]}" for k, n in sorted(CITAS.items(), key=lambda kv: kv[1])]
    texto = citar(cuerpo) + "## Referencias\n" + "\n".join(lista) + "\n" + "\n".join(resto_refs)
    print(f"{len(CITAS)} referencias citadas en orden de aparición.")

    doc = Document()
    configurar(doc)
    n_fig, n_tab, n_ec = 0, 0, 0
    lineas = texto.splitlines()
    i = 0
    en_refs = False
    while i < len(lineas):
        linea = lineas[i].rstrip()
        if not linea.strip() or linea.startswith("<!--"):
            i += 1
            continue
        if linea.startswith("# "):  # título
            p = doc.add_paragraph()
            formato_parrafo(p, WD_ALIGN_PARAGRAPH.CENTER, 1.15, 0, 6)
            texto_enriquecido(p, linea[2:], tam=15, base_negrita=True)
        elif linea.startswith("%TITULO_EN "):
            p = doc.add_paragraph()
            formato_parrafo(p, WD_ALIGN_PARAGRAPH.CENTER, 1.15, 0, 8)
            texto_enriquecido(p, linea[len("%TITULO_EN "):], tam=12, base_cursiva=True)
        elif linea.startswith("%CENTRO "):
            p = doc.add_paragraph()
            formato_parrafo(p, WD_ALIGN_PARAGRAPH.CENTER, 1.15, 0, 2)
            texto_enriquecido(p, linea[len("%CENTRO "):], tam=10.5)
        elif linea.startswith("## "):
            en_refs = "Referencias" in linea
            doc.add_heading(level=1).add_run(linea[3:])
        elif linea.startswith("### "):
            doc.add_heading(level=2).add_run(linea[4:])
        elif linea.startswith("%SALTO"):
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        elif linea.startswith("$$"):
            n_ec += 1
            cuerpo = linea.strip()
            mo = re.match(r"\$\$(.+)\$\$\s*(\(\w+\))?\s*$", cuerpo)
            if not mo:
                raise ValueError(f"Ecuación mal formada: {linea}")
            latex, etiqueta = mo.group(1), mo.group(2) or f"({n_ec})"
            # Tabla invisible 1x3: [vacío | ecuación en modo presentación | (n)]
            t = doc.add_table(rows=1, cols=3)
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = False
            anchos = [1.2, ANCHO_TEXTO_CM - 2.4, 1.2]
            for j, ancho in enumerate(anchos):
                celda = t.cell(0, j)
                celda.width = Cm(ancho)
                celda.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                borde_celda(celda, top=(0, "auto"), bottom=(0, "auto"), left=(0, "auto"), right=(0, "auto"))
            margenes_celda(t, 20, 20, 0, 0)
            pe = t.cell(0, 1).paragraphs[0]
            formato_parrafo(pe, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 2, 2)
            elem = MATE.omml(latex, bloque=True)
            if elem is not None:
                pe._p.append(elem)
            else:
                fijar_fuente(pe.add_run(latex), 10, cursiva=True)
            pn = t.cell(0, 2).paragraphs[0]
            formato_parrafo(pn, WD_ALIGN_PARAGRAPH.RIGHT, 1.0, 0, 0)
            fijar_fuente(pn.add_run(etiqueta), 11)
            formato_parrafo(t.cell(0, 0).paragraphs[0], WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 0)
            t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        elif linea.startswith("!["):
            mo = re.match(r"!\[(.+)\]\((.+?)\)(\{ancho=([\d.]+)\})?", linea)
            leyenda, ruta, ancho = mo.group(1), mo.group(2), float(mo.group(4) or 16)
            n_fig += 1
            p = doc.add_paragraph()
            formato_parrafo(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 6, 2, mantener=True)
            p.add_run().add_picture(str((RAIZ / "articulo" / ruta).resolve()), width=Cm(ancho))
            c = doc.add_paragraph()
            formato_parrafo(c, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.0, 0, 10)
            texto_enriquecido(c, leyenda if leyenda.startswith("**") else f"**Fig. {n_fig}.** {leyenda}", tam=9.5)
        elif linea.startswith("[[TABLA:"):
            mo = re.match(r"\[\[TABLA:(\w+)\|(.+?)(\|(.+))?\]\]", linea)
            clave, titulo, nota = mo.group(1), mo.group(2), mo.group(4)
            if not titulo.startswith("**"):  # las tablas de anexo traen su propio rótulo (p. ej., Tabla B1)
                n_tab += 1
                titulo = f"**Tabla {n_tab}.** {titulo}"
            {"estado_arte": tabla_estado_arte, "escenarios": tabla_escenarios, "analitica": tabla_analitica,
             "montecarlo": tabla_montecarlo, "pruebas": tabla_pruebas}.get(clave, lambda d, t, n: tabla_entorno(d, t, n, V))(doc, titulo, nota)
        elif linea.startswith("```"):
            titulo = linea[3:].strip()
            cuerpo = []
            i += 1
            while not lineas[i].startswith("```"):
                cuerpo.append(lineas[i])
                i += 1
            t = doc.add_table(rows=1, cols=1)
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            celda = t.cell(0, 0)
            celda.width = Cm(ANCHO_TEXTO_CM)
            sombrear(celda, "F6F6F3")
            borde_celda(celda, top=(8, "0B0B0B"), bottom=(8, "0B0B0B"), left=(0, "auto"), right=(0, "auto"))
            p0 = celda.paragraphs[0]
            formato_parrafo(p0, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 2, 3)
            texto_enriquecido(p0, titulo, tam=9.5)
            for ln in cuerpo:
                pp = celda.add_paragraph()
                formato_parrafo(pp, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 0)
                sangria = len(ln) - len(ln.lstrip(" "))
                pp.paragraph_format.left_indent = Cm(0.25 + 0.22 * sangria)
                codigo, _, comentario = ln.strip().partition("▷")
                texto_enriquecido(pp, codigo.rstrip(), tam=8.8)
                if comentario:  # comentarios alineados con una tabulación fija
                    pp.paragraph_format.tab_stops.add_tab_stop(Cm(10.2))
                    texto_enriquecido(pp, "\t▷ " + comentario.strip(), tam=8.8, color="52514E")
            trPr = t.rows[0]._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:cantSplit"))
            esp = doc.add_paragraph()
            formato_parrafo(esp, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 6)
        elif re.match(r"^- ", linea):
            p = doc.add_paragraph(style="List Bullet")
            formato_parrafo(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.5, 0, 3)
            texto_enriquecido(p, linea[2:])
        elif en_refs and re.match(r"^\[\d+\]", linea):
            p = doc.add_paragraph()
            formato_parrafo(p, WD_ALIGN_PARAGRAPH.LEFT, 1.15, 0, 3)
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.9)
            num, resto = re.match(r"^(\[\d+\])\s*(.*)$", linea).groups()
            texto_enriquecido(p, f"{num}\t{resto}", tam=9.5)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(0.9))
        elif linea.startswith("%NOTA "):
            p = doc.add_paragraph()
            formato_parrafo(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.15, 0, 6)
            texto_enriquecido(p, linea[len("%NOTA "):], tam=9.5)
        else:
            p = doc.add_paragraph()
            formato_parrafo(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1.5, 0, 6)
            texto_enriquecido(p, linea)
        i += 1
    for p in doc.paragraphs:  # títulos de sección en la fuente del documento
        if p.style.name.startswith("Heading"):
            for r in p.runs:
                fijar_fuente(r, None, color="0B0B0B")
    doc.core_properties.title = "Modelo oculto de Markov para inferir la condición real del pavimento"
    doc.core_properties.author = "Johan, Juan y Jhojan"
    doc.core_properties.language = "es-CO"
    doc.save(SALIDA)
    print(f"Documento generado: {SALIDA}  ({n_fig} figuras, {n_tab} tablas, {n_ec} ecuaciones numeradas)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    construir()
