"""Small reportlab layer shared by the two report builders: fonts, styles,
and helpers that turn plain Python into paragraphs, bullet lists, tables and
scaled images. reportlab lives in the scratch pylib (see README in reports/)."""
import os
import sys

import matplotlib

_SP = os.environ.get("REPORTLAB_PYLIB")
if _SP and _SP not in sys.path:
    sys.path.insert(0, _SP)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

_FONTS = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(_FONTS, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(_FONTS, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(_FONTS, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-M", os.path.join(_FONTS, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

PAGE_W, PAGE_H = A4
MARGIN = 1.9 * cm
TEXT_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#1c1c1c")
ACCENT = colors.HexColor("#1f4e79")
MUTED = colors.HexColor("#555555")
RULE = colors.HexColor("#c9d3df")
HEAD_BG = colors.HexColor("#e8eef5")
ZEBRA = colors.HexColor("#f6f8fb")
CALL_BG = colors.HexColor("#fff7e6")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=22, leading=27, textColor=ACCENT, alignment=TA_LEFT, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=12.5, leading=17, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15.5, leading=20, textColor=ACCENT, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="DV-B", fontSize=10.3, leading=14, textColor=INK, spaceBefore=7, spaceAfter=2),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8.4, leading=11.5, textColor=MUTED, spaceAfter=4),
    "caption": ParagraphStyle("caption", fontName="DV-I", fontSize=8.4, leading=11.2, textColor=MUTED, alignment=TA_CENTER, spaceBefore=2, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.2, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.2, leading=10.6, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="DV-M", fontSize=8.0, leading=11.6, textColor=INK, leftIndent=14, spaceBefore=2, spaceAfter=6, backColor=colors.HexColor("#f3f4f6"), borderPadding=(4, 6, 4, 6)),
    "callout": ParagraphStyle("callout", fontName="DV", fontSize=9.4, leading=13.2, textColor=INK, backColor=CALL_BG, borderPadding=(6, 8, 6, 8), borderColor=colors.HexColor("#f0c36d"), borderWidth=0.6, spaceBefore=4, spaceAfter=9),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(text):
    return P(text, "h1")


def H2(text):
    return P(text, "h2")


def H3(text):
    return P(text, "h3")


def Small(text):
    return P(text, "small")


def Mono(text):
    return Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), S["mono"])


def Call(text):
    return P(text, "callout")


def Bul(items, style="body"):
    return ListFlowable([ListItem(Paragraph(t, S[style]), leftIndent=12, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=14, bulletFontName="DV", bulletFontSize=9)


def Num(items, style="body"):
    return ListFlowable([ListItem(Paragraph(t, S[style]), leftIndent=14) for t in items],
                        bulletType="1", leftIndent=16, bulletFontName="DV", bulletFontSize=9)


def Img(path, width=TEXT_W, caption=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    width = min(width, TEXT_W)
    im = Image(path, width=width, height=width * h / w)
    im.hAlign = "CENTER"
    out = [im]
    if caption:
        out.append(P(caption, "caption"))
    return KeepTogether(out)


def Tbl(rows, widths=None, header=True, font=8.2, zebra=True, align_first_left=True, bold_first_col=False):
    """rows: list of lists of strings (may contain <b>..</b> markup)."""
    data = []
    for i, r in enumerate(rows):
        cells = []
        for j, c in enumerate(r):
            st = "cellb" if (header and i == 0) or (bold_first_col and j == 0) else "cell"
            cells.append(Paragraph(str(c), S[st]))
        data.append(cells)
    if widths is None:
        n = len(rows[0])
        widths = [TEXT_W / n] * n
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.9, ACCENT)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if (i - (1 if header else 0)) % 2 == 1:
                style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def Sp(h=6):
    return Spacer(1, h)


def build(story, path, title, subtitle=""):
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 1.1 * cm, title)
        canvas.drawRightString(PAGE_W - MARGIN, 1.1 * cm, "page %d" % doc.page)
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN, 1.4 * cm, PAGE_W - MARGIN, 1.4 * cm)
        canvas.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=1.7 * cm, bottomMargin=1.9 * cm, title=title, author="mesh pipeline analysis")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return path


def title_page(title, subtitle, lines):
    out = [Sp(120), Paragraph(title, S["title"]), Paragraph(subtitle, S["subtitle"]), Sp(10)]
    for ln in lines:
        out.append(P(ln, "small"))
    out.append(PageBreak())
    return out
