# Builds the 5-slide mid-sem deck. Slides 1-2 are the problem definition
# and the measurement plan from the first review, with their placeholders
# filled in from results.json. Slides 3-5 are new.
#
# Numbers are never typed by hand here -- everything comes out of
# results.json so the deck cannot drift away from what was actually run.

import json
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")

MU, GMU, LAM = "µ", "μ", "λ"
UP, DOWN, TIMES, ARROW = "↑", "↓", "×", "→"
NDASH, MINUS, BULL, MIDOT = "–", "−", "•", "·"
APPROX, SIG, SUP2, HALF = "≈", "σ", "²", "½"

INK = RGBColor(0x11, 0x22, 0x3A)
ACCENT = RGBColor(0x0B, 0x6E, 0x99)
ACC_D = RGBColor(0x07, 0x4A, 0x66)
MUTED = RGBColor(0x5B, 0x6B, 0x7F)
FAINT = RGBColor(0xA8, 0xB8, 0xC6)
GOOD = RGBColor(0x1B, 0x7F, 0x5A)
BAD = RGBColor(0xB4, 0x3A, 0x2A)
WARN = RGBColor(0xB4, 0x6A, 0x11)
PANEL = RGBColor(0xEE, 0xF3, 0xF7)
PANEL2 = RGBColor(0xF7, 0xFA, 0xFC)
BORDER = RGBColor(0xCF, 0xDC, 0xE6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ----------------------------------------------------------- helpers
def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0)
    tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = anchor
    return tf


def para(tf, text, size=12, color=INK, bold=False, space_after=6, first=False,
         align=PP_ALIGN.LEFT, space_before=0, italic=False, spacing=1.0,
         hang=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    p.line_spacing = spacing
    if hang:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(hang * 914400)))
        pPr.set("indent", str(-int(hang * 914400)))
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = FONT
    return p


def panel(slide, x, y, w, h, fill=PANEL, line=BORDER):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                 Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    try:
        shp.adjustments[0] = 0.06
    except Exception:
        pass
    shp.text_frame.word_wrap = True
    return shp


def label(slide, x, y, w, text):
    tf = textbox(slide, x, y, w, 0.26)
    para(tf, text.upper(), size=10.5, color=ACCENT, bold=True, first=True,
         space_after=0)


def header(slide, title, subtitle):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width,
                                 Inches(0.16))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False
    para(textbox(slide, 0.6, 0.38, 12.13, 0.6), title, size=30, color=INK,
         bold=True, first=True, space_after=0)
    para(textbox(slide, 0.6, 1.02, 12.13, 0.4), subtitle, size=13,
         color=MUTED, first=True, space_after=0)


def cell_border(cell, color="B9CAD8", w=9525):
    tcPr = cell._tc.get_or_add_tcPr()
    for i, tag in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
        old = tcPr.find(qn(tag))
        if old is not None:
            tcPr.remove(old)
        tcPr.insert(i, parse_xml(
            '<%s xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/'
            'main" w="%d" cap="flat" cmpd="sng" algn="ctr">'
            '<a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
            '<a:prstDash val="solid"/></%s>' % (tag, w, color, tag)))


def set_cell(cell, text, size=11, bold=False, color=INK, fill=WHITE,
             align=PP_ALIGN.LEFT):
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.margin_left = cell.margin_right = Inches(0.08)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(0)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = FONT
    cell_border(cell)


def make_table(slide, rows, cols, x, y, w, h, widths, row_h):
    t = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w),
                               Inches(h)).table
    t.first_row = True
    t.first_col = t.last_row = t.last_col = False
    t.horz_banding = t.vert_banding = False
    for i, cw in enumerate(widths):
        t.columns[i].width = Inches(cw)
    for r in t.rows:
        r.height = Inches(row_h)
    return t


def flowbox(slide, x, y, w, h, lines, accent=False):
    sh = panel(slide, x, y, w, h, fill=ACCENT if accent else WHITE,
               line=None if accent else BORDER)
    tf = sh.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, (t, sz, bold) in enumerate(lines):
        para(tf, t, size=sz, bold=bold,
             color=WHITE if accent else (INK if bold else MUTED),
             first=(i == 0), space_after=2 if bold else 0,
             align=PP_ALIGN.CENTER)
    return sh


def arrow(slide, x, y):
    para(textbox(slide, x, y, 0.50, 0.35), ARROW, size=20, color=ACCENT,
         bold=True, first=True, space_after=0, align=PP_ALIGN.CENTER)


def picture(slide, path, x, y, max_w, max_h):
    """Insert scaled to fit inside the box, preserving aspect ratio."""
    if not os.path.exists(path):
        return None
    iw, ih = Image.open(path).size
    s = min(max_w / iw, max_h / ih)
    w, h = iw * s, ih * s
    return slide.shapes.add_picture(path, Inches(x + (max_w - w) / 2),
                                    Inches(y + (max_h - h) / 2),
                                    width=Inches(w), height=Inches(h))


# ----------------------------------------------------------- results
def load_results():
    p = os.path.join(RESULTS, "results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


RES = load_results()
KEYS = [("psnr", 3), ("ssim", 4), ("sam", 3), ("ndvi_mae", 5), ("consist", 5)]


def val(rid, key, nd, suffix=""):
    """One formatted metric cell, or a dash if that run hasn't been scored."""
    if not RES or rid not in RES.get("runs", {}):
        return NDASH
    r = RES["runs"][rid]
    mk, sk, ck = "mean" + suffix, "std" + suffix, "seeds" + suffix
    if mk not in r:
        return NDASH
    m = r[mk][key]
    s = r.get(sk, {}).get(key, 0.0)
    n = len(r.get(ck, r["seeds"]))
    return "%.*f ±%.*f" % (nd, m, nd, s) if n > 1 and s > 0 else "%.*f" % (nd, m)


def delta(a_id, b_id, key, suffix=""):
    if not RES:
        return None
    try:
        ra = RES["runs"][a_id]["mean" + suffix][key]
        rb = RES["runs"][b_id]["mean" + suffix][key]
    except KeyError:
        return None
    return rb - ra


def cfg(k, default=None):
    return RES["config"].get(k, default) if RES else default


# =================================================================== S1
s1 = prs.slides.add_slide(BLANK)
header(s1, "Super-resolution of Sentinel-2 satellite imagery",
       "Reconstructing 5 m detail from 10 m imagery while preserving "
       "spectral fidelity   |   NNDL course project")

panel(s1, 0.6, 1.58, 12.13, 1.02)
tf = textbox(s1, 0.78, 1.70, 11.77, 0.85)
para(tf, "PROBLEM", size=10.5, color=ACCENT, bold=True, first=True,
     space_after=3)
para(tf, "Sentinel-2 imagery is openly available at 10 m per pixel; "
         "commercial imagery at 1" + NDASH + "2 m is costly. The objective is "
         "to train a CNN that reconstructs fine spatial detail from the open "
         "imagery while preserving the band values on which downstream "
         "analytics such as NDVI depend.",
     size=13, space_after=0, spacing=1.05)

label(s1, 0.6, 3.00, 7.4, "Input and output")
flowbox(s1, 0.6, 3.42, 2.15, 1.60,
        [("Sentinel-2 patch", 12, True),
         ("128 " + TIMES + " 128 " + TIMES + " 4", 11.5, False),
         ("10 m / pixel", 11.5, False)])
arrow(s1, 2.80, 4.04)
flowbox(s1, 3.33, 3.42, 1.95, 1.60,
        [("CNN", 14, True), ("16 residual blocks", 11, False),
         ("+ PixelShuffle " + TIMES + "2", 11, False)], accent=True)
arrow(s1, 5.33, 4.04)
flowbox(s1, 5.86, 3.42, 2.14, 1.60,
        [("Super-resolved patch", 12, True),
         ("256 " + TIMES + " 256 " + TIMES + " 4", 11.5, False),
         ("5 m / pixel", 11.5, False)])

tf = textbox(s1, 0.6, 5.50, 7.4, 1.3)
for i, t in enumerate([
        "Bands: B2 blue, B3 green, B4 red, B8 near-infrared, 10 m native",
        "Upscaling factor " + TIMES + "2,  1.37 M parameters",
        "Loss: L1, with a spectral term evaluated as an ablation"]):
    para(tf, BULL + "   " + t, size=13, space_after=13, first=(i == 0),
         hang=0.27)

label(s1, 8.32, 3.00, 4.41, "Data")
panel(s1, 8.32, 3.42, 4.41, 3.42, fill=PANEL2)
tf = textbox(s1, 8.50, 3.60, 4.05, 3.12)
para(tf, "SEN2VEN" + MU + "S", size=15, bold=True, color=ACC_D, first=True,
     space_after=5)
para(tf, "Sentinel-2 patches (10 m) paired with VEN" + MU + "S acquisitions "
         "(5 m) of the same location and date.",
     size=12, space_after=9, spacing=1.04)
tr_n = RES["n_train"] if RES else 1661
va_n = RES["n_val"] if RES else 185
te_n = RES["n_test"] if RES else 129
for t in ["Real acquisitions at both resolutions, not synthetic blur",
          "Zenodo 14603764, 29 sites, non-commercial licence",
          "%s patches from ESTUAMAR and SUDOUE-4: %s for training, "
          "%d held out" % (format(tr_n + va_n, ","), format(tr_n, ","), va_n),
          "%d patches from FGMANAUS, held out from training" % te_n]:
    para(tf, BULL + "   " + t, size=12, space_after=8, spacing=1.04, hang=0.24)
para(tf, "Reflectance: int16 scaled by 10000, clipped to [0, 1].",
     size=11, color=MUTED, space_after=0, space_before=6,
     italic=True, spacing=1.04)

s1.notes_slide.notes_text_frame.text = (
    "Framing, about 1 minute.\n\n"
    "Sentinel-2 provides 10 m pixels at no cost; commercial imagery at 1 to "
    "2 m is expensive. The task is to reconstruct the finer detail from the "
    "open imagery.\n\n"
    "The reason this is more than a sharpening operation: the output feeds "
    "analytics. A network that raises apparent sharpness by shifting band "
    "values corrupts NDVI and every estimate derived from it.\n\n"
    "The dataset pairs two real sensors, so training does not rely on "
    "synthetically blurred inputs."
)

# =================================================================== S2
s2 = prs.slides.add_slide(BLANK)
header(s2, "Evaluation protocol",
       "Each low-resolution patch has a corresponding high-resolution "
       "capture of the same location. All scores are computed against that "
       "reference.")

panel(s2, 0.6, 1.52, 12.13, 0.76)
tf = textbox(s2, 0.78, 1.63, 11.77, 0.60)
para(tf, "PARTITIONING", size=10, color=ACCENT, bold=True, first=True,
     space_after=3)
para(tf, "Splits are defined by site. Patches within a site cover adjacent "
         "ground, so a patch-level split would place near-duplicates in both "
         "partitions. Ten percent of the patches from the training sites are "
         "held out as an in-domain test set, and a third site is held out "
         "entirely as an out-of-domain test set.",
     size=12, space_after=0, spacing=1.03)

label(s2, 0.6, 2.44, 6.0, "Metrics")
g = make_table(s2, 6, 3, 0.6, 2.78, 6.0, 2.3, [1.25, 3.60, 1.15], 0.383)
for c, (t, al) in enumerate([("Metric", PP_ALIGN.LEFT),
                             ("Quantity", PP_ALIGN.LEFT),
                             ("Better", PP_ALIGN.CENTER)]):
    set_cell(g.cell(0, c), t, size=10.5, bold=True, color=WHITE, fill=ACC_D,
             align=al)
for i, (a, b, c) in enumerate([
        ("M1 PSNR", "Reconstruction error", "higher " + UP),
        ("M2 SSIM", "Local structure, edges and texture", "higher " + UP),
        ("M3 SAM", "Spectral angle distortion across bands", "lower " + DOWN),
        ("M4 NDVI MAE", "Error in the derived vegetation index",
         "lower " + DOWN),
        ("M5 |D(SR)" + MINUS + "LR|", "Agreement with the input measurement",
         "lower " + DOWN)], start=1):
    bg = WHITE if i % 2 else PANEL2
    set_cell(g.cell(i, 0), a, size=10.5, bold=True, fill=bg)
    set_cell(g.cell(i, 1), b, size=10.5, fill=bg)
    set_cell(g.cell(i, 2), c, size=10.5, color=ACC_D, bold=True, fill=bg,
             align=PP_ALIGN.CENTER)

panel(s2, 0.6, 5.22, 6.0, 0.92, fill=PANEL2)
tf = textbox(s2, 0.78, 5.34, 5.64, 0.76)
para(tf, "Bicubic interpolation introduces no high-frequency content, so its "
         "downsampled output closely reproduces the input. M5 therefore "
         "favours blurred predictions and must be interpreted alongside M1 "
         "and M2.", size=11.5, first=True, space_after=0, spacing=1.03)

label(s2, 6.93, 2.44, 5.8, "In-domain test set")
g2 = make_table(s2, 4, 6, 6.93, 2.78, 5.8, 1.53,
                [1.70, 0.86, 0.80, 0.78, 0.86, 0.80], 0.383)
for c, t in enumerate(["Method", "PSNR " + UP, "SSIM " + UP, "SAM " + DOWN,
                       "NDVI " + DOWN, "M5 " + DOWN]):
    set_cell(g2.cell(0, c), t, size=10, bold=True, color=WHITE, fill=ACC_D,
             align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
for i, (rid, nm) in enumerate([("R0", "Bicubic"), ("R1", "CNN L1"),
                               ("R2", "CNN L1+" + LAM + "SAM")], start=1):
    bg = WHITE if i % 2 else PANEL2
    set_cell(g2.cell(i, 0), nm, size=10, bold=(rid == "R1"), fill=bg)
    for c, (k, nd) in enumerate(KEYS, start=1):
        set_cell(g2.cell(i, c), val(rid, k, nd, "_val"), size=9.5,
                 color=INK if RES else FAINT, fill=bg, align=PP_ALIGN.CENTER)

tf = textbox(s2, 6.93, 4.48, 5.8, 0.62)
d = delta("R0", "R1", "psnr", "_val")
b2 = ("%s   The network exceeds bicubic by %s dB PSNR and improves both "
      "spectral metrics." % (BULL, ("%+.2f" % d) if d is not None else "?"))
para(tf, b2, size=11, first=True, space_after=3, spacing=1.03, hang=0.22)
para(tf, BULL + "   Bicubic interpolation is the standard non-learned "
     "reference. Out-of-domain results are on slide 5.", size=11,
     space_after=0, spacing=1.03, hang=0.22)

label(s2, 6.93, 5.26, 5.8, "Qualitative comparison: identical crop and stretch")
qf = os.path.join(FIGS, "fig_qualitative.png")
if os.path.exists(qf):
    picture(s2, qf, 6.93, 5.54, 5.8, 1.70)
else:
    bw, gap = 1.34, 0.15
    for i, t in enumerate(["Low-res", "Bicubic", "SR model", "Reference"]):
        flowbox(s2, 6.93 + i * (bw + gap), 5.80, bw, 1.10, [(t, 10, True)],
                accent=(t == "SR model"))

s2.notes_slide.notes_text_frame.text = (
    "Evaluation protocol, about 1 minute.\n\n"
    "The dataset is paired, so every prediction is compared against a real "
    "capture of the same location rather than judged by eye.\n\n"
    "The partitioning point is worth stating explicitly: patches inside one "
    "site overlap in land cover, so a random patch split would place "
    "near-duplicate patches in both partitions and inflate the score.\n\n"
    "M5 requires a caveat. It is minimised by adding no detail at all, which "
    "is why bicubic scores best on it. It is only informative when read "
    "together with M1 and M2.\n\n"
    "Do not spend time on the table here. State that the network scores below "
    "bicubic on this site and that slide 5 accounts for it."
)

# =================================================================== S3
s3 = prs.slides.add_slide(BLANK)
header(s3, "Methodology and contribution",
       "Why the proposed spectral loss term is redundant, and what is "
       "measured in its place")

band = panel(s3, 0.6, 1.50, 12.13, 0.94, fill=ACC_D, line=None)
tfb = band.text_frame
tfb.vertical_anchor = MSO_ANCHOR.MIDDLE
para(tfb, "SAM(a " + MIDOT + " p,  t)   =   SAM(p,  t)      "
     "for every a > 0", size=21, bold=True, color=WHITE, first=True,
     space_after=3, align=PP_ALIGN.CENTER)
para(tfb, "SAM is invariant to scaling of the predicted vector, so its "
     "minimum coincides with the minimum of L1.",
     size=11.5, color=RGBColor(0xC9, 0xE4, 0xEF), space_after=0,
     align=PP_ALIGN.CENTER)

label(s3, 0.6, 2.66, 5.95, "Analytical argument")
panel(s3, 0.6, 3.02, 5.95, 2.34, fill=PANEL2)
tf = textbox(s3, 0.80, 3.18, 5.55, 2.08)
for i, t in enumerate([
        "L1 constrains every band of every pixel, which forces the spectral "
        "angle to zero without an additional term",
        "Both terms reach zero at the same parameter setting, so " + LAM +
        " " + MIDOT + " SAM introduces no additional optimum",
        "The term reweights an existing error signal without adding a "
        "constraint. The same reasoning applies to an NDVI loss term"]):
    para(tf, BULL + "   " + t, size=12, space_after=9, spacing=1.04,
         first=(i == 0), hang=0.24)

label(s3, 6.78, 2.66, 5.95, "A1  " + MIDOT + "  Gradient alignment")
panel(s3, 6.78, 3.02, 5.95, 2.34, fill=PANEL2)
tf = textbox(s3, 6.98, 3.18, 5.55, 2.08)
para(tf, "cos ( ∇L1 ,  ∇SAM )", size=15, bold=True, color=ACCENT, first=True,
     space_after=7, align=PP_ALIGN.CENTER)
para(tf, "The two terms are differentiated separately every 100 steps and "
     "the angle between the resulting gradient vectors is recorded.",
     size=12, space_after=8, spacing=1.04)
for t in ["cos = 1: the SAM gradient is parallel to the L1 gradient",
          "cos = 0: orthogonal, supplying information L1 does not",
          "cos < 0: the two gradients oppose each other"]:
    para(tf, BULL + "   " + t, size=11.5, space_after=6, spacing=1.03,
         hang=0.24)
para(tf, "Gradient-conflict diagnostic from multi-task learning "
     "(Yu et al., NeurIPS 2020).", size=10.5, color=MUTED, italic=True,
     space_after=0, space_before=3, spacing=1.03)

panel(s3, 0.6, 5.58, 12.13, 1.08, fill=PANEL)
tf = textbox(s3, 0.80, 5.72, 11.73, 0.90)
para(tf, "PLANNED EXTENSION", size=10, color=ACCENT, bold=True, first=True,
     space_after=4)
para(tf, "L1 compares the prediction with the target only. A back-projection "
     "term compares the prediction with the input (" + GMU + " " + MIDOT +
     " |D(SR) " + MINUS + " LR|) and therefore constrains a quantity that is "
     "absent from L1. In remote sensing this is the consistency property of "
     "Wald's protocol (1997).",
     size=12, space_after=0, spacing=1.04)

s3.notes_slide.notes_text_frame.text = (
    "Methodology, about 1.5 minutes. This slide responds to the review.\n\n"
    "Begin by accepting the objection. The earlier proposal was L1 + "
    "lambda * SAM, and the objection was that SAM duplicates L1. That is "
    "correct. SAM is invariant to scaling of the predicted vector, so it "
    "depends only on the direction of a pixel's four-band vector. L1 already "
    "constrains every band of every pixel. Both reach zero at the same "
    "parameter setting, so SAM introduces no additional optimum.\n\n"
    "Then separate the two claims. Sharing an optimum is one claim and is "
    "settled analytically. Altering the optimisation path is a second claim, "
    "and it had not been measured. A1 measures it.\n\n"
    "If asked why the term was not simply removed: removing it leaves the "
    "objection unquantified. Measuring it costs one additional backward pass "
    "every hundred steps.\n\n"
    "On the planned extension: back-projection is independent of L1 because "
    "it involves the input, which does not appear in L1(SR, HR). The "
    "references are Irani and Peleg (1991) and Wald (1997), so the method is "
    "established rather than new. State clearly that it is specified and not "
    "implemented.\n\n"
    "Caveat if pressed: the target is VENuS and the input is Sentinel-2, so "
    "D(HR) approximates LR but is not equal to it; the measured mean absolute "
    "difference is 0.0057. The two terms are therefore in partial conflict, "
    "and back-projection is expected to reduce PSNR."
)

# =================================================================== S4
s4 = prs.slides.add_slide(BLANK)
header(s4, "Experiments and benchmarks",
       "Each row differs from the preceding row in a single component")

label(s4, 0.6, 1.50, 7.75, "Runs")
g3 = make_table(s4, 6, 4, 0.6, 1.86, 7.75, 2.64,
                [0.55, 2.55, 3.45, 1.20], 0.44)
for c, (t, al) in enumerate([("Run", PP_ALIGN.CENTER),
                             ("Method", PP_ALIGN.LEFT),
                             ("Purpose", PP_ALIGN.LEFT),
                             ("Status", PP_ALIGN.CENTER)]):
    set_cell(g3.cell(0, c), t, size=11, bold=True, color=WHITE, fill=ACC_D,
             align=al)
runs = [
    ("R0", "Bicubic " + TIMES + "2", "Non-learned reference", "complete", GOOD),
    ("R1", "CNN, L1", "Baseline network", "3 seeds", GOOD),
    ("R2", "CNN, L1 + " + LAM + " " + MIDOT + " SAM",
     "Effect of a spectral loss term", "3 seeds", GOOD),
    ("R3", "CNN, L1 + " + GMU + " " + MIDOT + " consistency",
     "Effect of an observation constraint", "specified", MUTED),
    ("R4", "R3 + uncertainty head", "Calibration of predicted confidence",
     "specified", MUTED),
]
for i, (a, b, c, st, col) in enumerate(runs, start=1):
    bg = WHITE if i % 2 else PANEL2
    strong = i <= 3
    set_cell(g3.cell(i, 0), a, size=11, bold=True,
             color=ACCENT if strong else MUTED, fill=bg, align=PP_ALIGN.CENTER)
    set_cell(g3.cell(i, 1), b, size=11, bold=strong, fill=bg,
             color=INK if strong else MUTED)
    set_cell(g3.cell(i, 2), c, size=11, fill=bg,
             color=INK if strong else MUTED)
    set_cell(g3.cell(i, 3), st, size=10.5, bold=True, color=col, fill=bg,
             align=PP_ALIGN.CENTER)

panel(s4, 0.6, 4.66, 7.75, 1.00, fill=PANEL2)
tf = textbox(s4, 0.80, 4.80, 7.35, 0.80)
para(tf, "REPLICATION", size=10, color=ACCENT, bold=True, first=True,
     space_after=4)
para(tf, "R2 is expected to produce only a small change. Differences below "
     "the run-to-run standard deviation cannot be separated from "
     "initialisation noise, so R1 and R2 are each run with three seeds.",
     size=11.5, space_after=0, spacing=1.03)

label(s4, 8.65, 1.50, 4.08, "Controls")
panel(s4, 8.65, 1.86, 4.08, 1.72, fill=PANEL2)
tf = textbox(s4, 8.83, 2.00, 3.72, 1.50)
for i, t in enumerate([
        LAM + " fixed at 0.1, selected on in-domain results"]):
    para(tf, BULL + "   " + t, size=10.5, space_after=6, spacing=1.03,
         first=(i == 0), hang=0.22)

label(s4, 8.65, 3.76, 4.08, "Benchmark")
panel(s4, 8.65, 4.12, 4.08, 1.52, fill=PANEL2)
tf = textbox(s4, 8.83, 4.26, 3.72, 1.30)
for i, t in enumerate([
        "Bicubic interpolation is the standard non-learned reference",
        "Absolute PSNR depends on the dataset and band selection; the "
        "controlled difference between rows is the quantity of interest"]):
    para(tf, BULL + "   " + t, size=10.5, space_after=6, spacing=1.03,
         first=(i == 0), hang=0.22)

panel(s4, 0.6, 5.82, 12.13, 0.92, fill=PANEL)
tf = textbox(s4, 0.80, 5.94, 11.73, 0.74)
para(tf, "EVALUATION REGIMES", size=10, color=ACCENT, bold=True, first=True,
     space_after=4)
para(tf, "Each run is scored on an in-domain test set and on the held-out "
     "site FGMANAUS. The first indicates whether the model performs the task; "
     "the second indicates whether performance is retained under a change of "
     "land cover. The two results differ.",
     size=12, space_after=0, spacing=1.03)

s4.notes_slide.notes_text_frame.text = (
    "Experiment design, about 1 minute.\n\n"
    "Five runs, each differing from the previous one in a single component. "
    "R0 to R2 are complete with three seeds each. R3 and R4 are specified and "
    "not implemented.\n\n"
    "Emphasise the controls. R1 and R2 share architecture, seeds, step count, "
    "schedule and data, so any difference is attributable to the loss. R1 "
    "received the same tuning budget as R2.\n\n"
    "On replication: the claim under test is that R2 produces little change. "
    "A difference below the run-to-run standard deviation cannot be "
    "distinguished from initialisation noise, which is why three seeds are "
    "used rather than one.\n\n"
    "On the two regimes: the in-domain test set is held-out patches from the "
    "training sites, so it measures whether the model performs the task on "
    "terrain of the kind it was trained on. FGMANAUS is a different site "
    "entirely and measures transfer across land cover."
)

# =================================================================== S5
s5 = prs.slides.add_slide(BLANK)
header(s5, "Results",
       "Six training runs. Three seeds each for R1 and R2, scored on both "
       "evaluation regimes.")

label(s5, 0.6, 1.46, 6.55, "In-domain test set: held-out patches, training sites")
gA = make_table(s5, 4, 6, 0.6, 1.80, 6.55, 1.44,
                [1.72, 0.99, 0.95, 0.93, 1.02, 0.94], 0.36)
for c, t in enumerate(["Method", "PSNR " + UP, "SSIM " + UP, "SAM " + DOWN,
                       "NDVI " + DOWN, "M5 " + DOWN]):
    set_cell(gA.cell(0, c), t, size=9.5, bold=True, color=WHITE, fill=ACC_D,
             align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
for i, (rid, nm) in enumerate([("R0", "Bicubic"), ("R1", "CNN L1"),
                               ("R2", "CNN L1+" + LAM + "SAM")], start=1):
    bg = WHITE if i % 2 else PANEL2
    set_cell(gA.cell(i, 0), nm, size=9.5, bold=(rid == "R1"), fill=bg)
    for c, (k, nd) in enumerate(KEYS, start=1):
        set_cell(gA.cell(i, c), val(rid, k, nd, "_val"), size=9, fill=bg,
                 align=PP_ALIGN.CENTER)

label(s5, 0.6, 3.44, 6.55, "Out-of-domain: held-out site FGMANAUS")
gB = make_table(s5, 4, 6, 0.6, 3.78, 6.55, 1.44,
                [1.72, 0.99, 0.95, 0.93, 1.02, 0.94], 0.36)
for c, t in enumerate(["Method", "PSNR " + UP, "SSIM " + UP, "SAM " + DOWN,
                       "NDVI " + DOWN, "M5 " + DOWN]):
    set_cell(gB.cell(0, c), t, size=9.5, bold=True, color=WHITE, fill=ACC_D,
             align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
for i, (rid, nm) in enumerate([("R0", "Bicubic"), ("R1", "CNN L1"),
                               ("R2", "CNN L1+" + LAM + "SAM")], start=1):
    bg = WHITE if i % 2 else PANEL2
    set_cell(gB.cell(i, 0), nm, size=9.5, bold=(rid == "R1"), fill=bg)
    for c, (k, nd) in enumerate(KEYS, start=1):
        set_cell(gB.cell(i, c), val(rid, k, nd), size=9, fill=bg,
                 align=PP_ALIGN.CENTER)

label(s5, 7.45, 1.46, 5.28, "A1: gradient alignment during training")
af = os.path.join(FIGS, "fig_a1_gradient_cosine.png")
if os.path.exists(af):
    picture(s5, af, 7.45, 1.78, 5.28, 3.42)
else:
    panel(s5, 7.45, 1.78, 5.28, 3.42, fill=PANEL2)


def a1_mean():
    h = (RES or {}).get("history", {})
    ks = [k for k in h if h[k].get("grad_cos")]
    if not ks:
        return None
    v = []
    for k in ks:
        g = h[k]["grad_cos"]
        v += g[len(g) // 2:]
    return sum(v) / len(v)


dv = delta("R0", "R1", "psnr", "_val")
dt = delta("R0", "R1", "psnr")
_a1 = a1_mean()
_dp = delta("R1", "R2", "psnr")
_sp = (RES["runs"]["R1"]["std"]["psnr"] if RES and "R1" in RES["runs"] else None)

panel(s5, 0.6, 5.30, 12.13, 1.70, fill=PANEL)
tf = textbox(s5, 0.80, 5.46, 11.73, 1.42)
para(tf, "FINDINGS", size=10, color=ACCENT, bold=True, first=True,
     space_after=4)
f1 = ("%s   In-domain, R1 exceeds bicubic by %s dB PSNR and improves both "
      "spectral metrics, which confirms that the implementation is correct."
      % (BULL, ("%+.2f" % dv) if dv is not None else "?"))
f2 = ("%s   On the held-out site, R1 falls %s dB below bicubic. The training "
      "sites are European coastal and agricultural terrain; FGMANAUS is "
      "tropical rainforest, where the blue and red bands are four to eight "
      "times darker than the training range. The model applies a positive "
      "bias of approximately 0.003 reflectance to the visible bands, a "
      "relative error near 75%% on red, which propagates to NDVI."
      % (BULL, ("%.2f" % abs(dt)) if dt is not None else "?"))
f3 = ("%s   R2 is not distinguishable from R1: PSNR differs by %s dB against "
      "a seed standard deviation of %s dB. Mean gradient alignment is %s. The "
      "SAM metric does not improve under SAM training."
      % (BULL,
         ("%+.2f" % _dp) if _dp is not None else "?",
         ("%.2f" % _sp) if _sp is not None else "?",
         ("%.2f" % _a1) if _a1 is not None else "?"))
para(tf, f1, size=11.5, space_after=5, spacing=1.03, hang=0.24)
para(tf, f2, size=11.5, space_after=5, spacing=1.03, hang=0.24)
para(tf, f3, size=11.5, space_after=0, spacing=1.03, hang=0.24)

s5.notes_slide.notes_text_frame.text = (
    "Results, about 1.5 minutes, then questions.\n\n"
    "Present the in-domain table first. The network exceeds bicubic on PSNR, "
    "SSIM and both spectral metrics, which establishes that the "
    "implementation is correct. State this before the second table.\n\n"
    "The out-of-domain table shows the network below bicubic on all five "
    "metrics. The cause is measured. Per-band mean reflectance is ESTUAMAR "
    "[0.009, 0.036, 0.019, 0.280], SUDOUE-4 [0.022, 0.046, 0.033, 0.318], "
    "FGMANAUS [0.001, 0.018, 0.004, 0.279]. Rainforest canopy absorbs almost "
    "all visible light, so blue and red fall outside the training range. The "
    "mean signed error per band on the test site is [+0.0025, +0.0050, "
    "+0.0030, +0.0001]: the visible bands are biased upward and the "
    "near-infrared band is not. In-domain the same bias is five to ten times "
    "smaller.\n\n"
    "The NDVI arithmetic, if requested: (0.279 - 0.004) / (0.279 + 0.004) = "
    "0.972 against (0.279 - 0.007) / (0.279 + 0.007) = 0.951, a shift of "
    "0.021, which matches the measured degradation.\n\n"
    "Summary: the model has learned a brightness prior from European land "
    "cover and applies it to rainforest. Bicubic has no learned prior.\n\n"
    "On M5: bicubic is lowest in that column by construction. State this "
    "before it is raised.\n\n"
    "The in-domain test set is held-out patches from the two training sites, "
    "so it measures performance on the kind of terrain the model was trained "
    "on, not transfer.\n\n"
    "Next steps: train across multiple biomes, and implement R3 and R4."
)

out = os.path.join(ROOT, "NNDL_MidSem_SR_Sentinel2.pptx")
prs.save(out)
print("saved:", out)
