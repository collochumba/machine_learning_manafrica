"""Audit Engagement Programs: export builders (Word, Excel, PDF, HTML). Pure functions, no Streamlit, no database.

Input is the dict returned by aep.get_program(). Every export carries the same identification block and the same five
columns in the same order: No. | Audit Area | Requirements | Procedure / Test | Control. Nothing outside the AEP scope
(no work performed, results, findings, or "Prepared by") is ever written."""
import base64
import html
import io
import re
from datetime import date

import theme as th

COLUMNS = ["No.", "Audit Area", "Requirements", "Procedure / Test", "Control"]
FORMATS = {  # key -> (label, extension, mime)
    "docx": ("Word (.docx)", "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "xlsx": ("Excel (.xlsx)", "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "pdf": ("PDF (.pdf)", "pdf", "application/pdf"),
    "html": ("HTML (.html)", "html", "text/html"),
}
TITLE = "AUDIT ENGAGEMENT PROGRAMS"
GREEN, GOLD, CREAM_GOLD = "004D2C", "D4AF37", "F3E7B8"


def letter(i):
    return chr(ord("A") + i) if i < 26 else str(i + 1)


def _meta(p):
    return [("Program Reference", p["program_ref"]), ("Program Title", p["title"]),
            ("Financial Year", p["financial_year"]), ("Audit Period", p.get("audit_period") or ""),
            ("Campus / Location", p.get("campus") or "N/A"), ("Status", p["status"])]


def slug(text, n=40):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(text or "")).strip("-")[:n].strip("-")
    return s or "Program"


def filename(p, fmt, on=None):
    """AEP-2026-001_Title-slug_v3_20261019.docx - reference, title, program version, date, extension."""
    on = on or date.today()
    return f"{slug(p['program_ref'], 20)}_{slug(p['title'])}_v{p.get('version', 1)}_{on:%Y%m%d}.{FORMATS[fmt][1]}"


# ---------------------------------------------------------------- HTML
DOC_CSS = """
.aep-doc{background:#fff;border:1px solid #bbb;padding:28px 34px;color:#111;font-family:Georgia,'Times New Roman',serif}
.aep-doc h1{font-size:1.1rem;margin:.1rem 0;text-align:center;color:#111 !important}
.aep-doc table{border-collapse:collapse;width:100%;font-size:.82rem;margin-bottom:.6rem}
.aep-doc th,.aep-doc td{border:1px solid #444;padding:5px 7px;vertical-align:top;text-align:left;white-space:pre-wrap;word-break:break-word}
.aep-doc th{background:#004D2C;color:#fff}.aep-doc td.k{background:#f4f4f4;font-weight:700;width:16%}
.aep-doc .sec{background:#F3E7B8;border:1px solid #444;padding:5px 8px;font-weight:700;margin-top:.8rem}
@media print{.aep-doc{border:none;padding:0}.aep-doc thead{display:table-header-group}.aep-doc tr{page-break-inside:avoid}.aep-doc .sec{page-break-after:avoid}}
"""


def _e(v):
    return html.escape(str(v if v is not None else ""))


def logo_uri():
    return ("data:image/png;base64," + base64.b64encode(th.LOGO_PATH.read_bytes()).decode()) if th.logo_available() else ""


def document_html(p, standalone=False):
    """Formal document. standalone=True returns a complete HTML page; otherwise a fragment for embedding in Streamlit."""
    logo = logo_uri()
    img = f'<div style="text-align:center"><img src="{logo}" alt="KSG logo" style="height:70px"></div>' if logo else ""
    m = _meta(p)
    out = [f'<div class="aep-doc">{img}<h1>KENYA SCHOOL OF GOVERNMENT (KSG)</h1><h1>INTERNAL AUDIT DEPARTMENT</h1>'
           f'<h1 style="margin-bottom:.8rem">{TITLE}</h1><table>']
    for i in range(0, len(m), 2):
        (a, b), (c, d) = m[i], m[i + 1]
        out.append(f'<tr><td class="k">{_e(a)}:</td><td>{_e(b)}</td><td class="k">{_e(c)}:</td><td>{_e(d)}</td></tr>')
    out.append("</table>")
    if not p["sections"]:
        out.append("<p><i>No audit area headings have been added to this program yet.</i></p>")
    for i, s in enumerate(p["sections"]):
        out.append(f'<div class="sec">{letter(i)}. {_e(s["heading"])}</div><table><thead><tr><th style="width:5%">{COLUMNS[0]}</th>'
                   + "".join(f"<th>{_e(c)}</th>" for c in COLUMNS[1:]) + "</tr></thead><tbody>")
        for n, r in enumerate(s["rows"], 1):
            out.append(f'<tr><td>{n}</td><td>{_e(r["area"])}</td><td>{_e(r["requirements"])}</td>'
                       f'<td>{_e(r["procedure"])}</td><td>{_e(r["control"])}</td></tr>')
        if not s["rows"]:
            out.append('<tr><td colspan="5"><i>No procedures recorded.</i></td></tr>')
        out.append("</tbody></table>")
    out.append("</div>")
    body = "".join(out)
    if not standalone:
        return body
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{_e(p['program_ref'])} - {TITLE.title()}</title>"
            f"<style>body{{margin:16px;background:#fff}}{DOC_CSS}</style></head><body>{body}</body></html>")


def build_html(p):
    return document_html(p, standalone=True).encode("utf-8")


# ---------------------------------------------------------------- Word
def build_docx(p):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(1.6))
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(9.5)

    def shade(cell, hex_):
        pr = cell._tc.get_or_add_tcPr()
        sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hex_)
        pr.append(sh)

    def centred(text, size, bold=True, color=None):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(2)
        run = para.add_run(text); run.bold = bold; run.font.size = Pt(size)
        if color:
            run.font.color.rgb = RGBColor.from_string(color)

    if th.logo_available():
        doc.add_paragraph().alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].add_run().add_picture(str(th.LOGO_PATH), height=Cm(1.8))
    centred("KENYA SCHOOL OF GOVERNMENT (KSG)", 14, color=GREEN)
    centred("INTERNAL AUDIT DEPARTMENT", 11)
    centred(TITLE, 13, color=GREEN)

    m = _meta(p)
    t = doc.add_table(rows=len(m) // 2, cols=4)
    t.style = "Table Grid"
    for i in range(0, len(m), 2):
        cells = t.rows[i // 2].cells
        for j, (k, v) in enumerate((m[i], m[i + 1])):
            cells[j * 2].text, cells[j * 2 + 1].text = f"{k}:", str(v)
            cells[j * 2].paragraphs[0].runs[0].bold = True
            shade(cells[j * 2], "F6F1E0")
    doc.add_paragraph()

    widths = [Cm(1.3), Cm(4.6), Cm(6.6), Cm(8.4), Cm(5.6)]
    if not p["sections"]:
        doc.add_paragraph("No audit area headings have been added to this program yet.")
    for i, s in enumerate(p["sections"]):
        h = doc.add_paragraph()
        h.paragraph_format.keep_with_next = True
        r = h.add_run(f"{letter(i)}. {s['heading']}"); r.bold = True; r.font.size = Pt(11); r.font.color.rgb = RGBColor.from_string(GREEN)
        tbl = doc.add_table(rows=1, cols=5)
        tbl.style = "Table Grid"
        tbl.autofit = False
        for c, (cell, name) in enumerate(zip(tbl.rows[0].cells, COLUMNS)):
            cell.text = name; cell.width = widths[c]
            run = cell.paragraphs[0].runs[0]; run.bold = True; run.font.color.rgb = RGBColor(255, 255, 255)
            shade(cell, GREEN)
        trpr = tbl.rows[0]._tr.get_or_add_trPr()
        rep = OxmlElement("w:tblHeader"); rep.set(qn("w:val"), "true"); trpr.append(rep)  # repeat header on each page
        for n, row in enumerate(s["rows"], 1):
            cells = tbl.add_row().cells
            for c, val in enumerate([str(n), row["area"], row["requirements"], row["procedure"], row["control"]]):
                cells[c].text = val; cells[c].width = widths[c]
        if not s["rows"]:
            cells = tbl.add_row().cells
            cells[0].merge(cells[4]).text = "No procedures recorded."
        doc.add_paragraph()
    buf = io.BytesIO(); doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- Excel
def _put(ws, row, col, value):
    """Write a cell as TEXT. A value starting with = + - @ would otherwise be stored as a formula (formula injection)."""
    c = ws.cell(row=row, column=col, value=value)
    if isinstance(value, str):
        c.data_type = "s"
    return c


def build_xlsx(p):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    wb = Workbook(); ws = wb.active; ws.title = "Audit Engagement Program"
    thin = Side(style="thin", color="888888"); box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")
    for col, w in zip("ABCDE", (7, 30, 48, 58, 40)):
        ws.column_dimensions[col].width = w
    r = 1
    for text, size in (("KENYA SCHOOL OF GOVERNMENT (KSG) - INTERNAL AUDIT DEPARTMENT", 12), (TITLE, 14)):
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=5)
        _put(ws, r, 1, text).font = Font(bold=True, size=size, color=GREEN)
        ws.cell(row=r, column=1).alignment = Alignment(horizontal="center")
        r += 1
    r += 1
    m = _meta(p)
    for i in range(0, len(m), 2):
        for j, (k, v) in enumerate((m[i], m[i + 1])):
            kc = _put(ws, r, 1 + j * 3 if j == 0 else 4, f"{k}:")
            kc.font = Font(bold=True); kc.fill = PatternFill("solid", fgColor="F6F1E0")
            vc_col = 2 if j == 0 else 5
            _put(ws, r, vc_col, str(v)).alignment = wrap
        r += 1
    r += 1
    hdr_row = None
    for i, s in enumerate(p["sections"]):
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=5)
        c = _put(ws, r, 1, f"{letter(i)}. {s['heading']}")
        c.font = Font(bold=True, color=GREEN); c.fill = PatternFill("solid", fgColor=CREAM_GOLD)
        r += 1
        for col, name in enumerate(COLUMNS, 1):
            c = _put(ws, r, col, name)
            c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor=GREEN); c.border = box
            c.alignment = Alignment(wrap_text=True, vertical="center")
        hdr_row = hdr_row or r
        r += 1
        for n, row in enumerate(s["rows"], 1):
            for col, val in enumerate([n, row["area"], row["requirements"], row["procedure"], row["control"]], 1):
                c = _put(ws, r, col, val); c.alignment = wrap; c.border = box
            r += 1
        if not s["rows"]:
            _put(ws, r, 1, "No procedures recorded.").font = Font(italic=True)
            r += 1
        r += 1
    ws.page_setup.orientation = "landscape"; ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    buf = io.BytesIO(); wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- PDF
def build_pdf(p):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    green, gold = colors.HexColor("#" + GREEN), colors.HexColor("#" + CREAM_GOLD)
    ss = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8, leading=10)
    head = ParagraphStyle("head", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    ctr = ParagraphStyle("ctr", parent=ss["Normal"], alignment=1, fontName="Helvetica-Bold", textColor=green, fontSize=12, leading=15)
    sec = ParagraphStyle("sec", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=10, textColor=green, backColor=gold, borderPadding=3, spaceBefore=8, spaceAfter=4)
    P = lambda t, st=cell: Paragraph(html.escape(str(t if t is not None else "")).replace("\n", "<br/>"), st)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title=f"{p['program_ref']} - Audit Engagement Programs", author="Kenya School of Government")
    story = []
    if th.logo_available():
        im = Image(str(th.LOGO_PATH)); ratio = im.imageWidth / im.imageHeight
        im.drawHeight, im.drawWidth = 1.6 * cm, 1.6 * cm * ratio; story.append(im)
    story += [Paragraph("KENYA SCHOOL OF GOVERNMENT (KSG)", ctr), Paragraph("INTERNAL AUDIT DEPARTMENT", ctr), Paragraph(TITLE, ctr), Spacer(1, 6)]
    m = _meta(p)
    bold = ParagraphStyle("b", parent=cell, fontName="Helvetica-Bold")
    mt = Table([[P(f"{a}:", bold), P(b), P(f"{c}:", bold), P(d)] for (a, b), (c, d) in zip(m[0::2], m[1::2])],
               colWidths=[3.8 * cm, 9.2 * cm, 3.8 * cm, 9.2 * cm])
    mt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F6F1E0")),
                            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F6F1E0")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [mt, Spacer(1, 6)]
    if not p["sections"]:
        story.append(P("No audit area headings have been added to this program yet."))
    widths = [1.2 * cm, 4.2 * cm, 6.6 * cm, 8.4 * cm, 6.3 * cm]
    for i, s in enumerate(p["sections"]):
        data = [[P(c, head) for c in COLUMNS]]
        data += [[P(n), P(r["area"]), P(r["requirements"]), P(r["procedure"]), P(r["control"])] for n, r in enumerate(s["rows"], 1)]
        if not s["rows"]:
            data.append([P("No procedures recorded."), "", "", "", ""])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), green),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")] + ([("SPAN", (0, 1), (-1, 1))] if not s["rows"] else [])))
        story += [Paragraph(html.escape(f"{letter(i)}. {s['heading']}"), sec), t]
    doc.build(story)
    return buf.getvalue()


BUILDERS = {"docx": build_docx, "xlsx": build_xlsx, "pdf": build_pdf, "html": build_html}


def build(p, fmt):
    """Returns (bytes, filename, mime). Raises ValueError for an unknown format."""
    if fmt not in BUILDERS:
        raise ValueError("Unsupported export format.")
    return BUILDERS[fmt](p), filename(p, fmt), FORMATS[fmt][2]
