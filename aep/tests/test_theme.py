import io, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import openpyxl
import theme as th
import test_core as core
from export_excel import build_workbook


def test_palette_is_ksg():
    assert (th.GREEN, th.DARK_GREEN, th.GOLD, th.MAROON) == ("#006B3F", "#004D2C", "#D4AF37", "#6B1F2B")


def test_risk_and_status_mapping():
    assert th.risk_style("Low")[1] == th.GREEN and th.risk_style("HIGH")[1] == th.MAROON
    assert th.risk_style("Critical")[0] == th.DARK_MAROON and th.risk_style("Medium")[2] == th.GOLD
    assert th.status_style("Approved")[0] == th.GREEN and th.status_style("Returned")[1] == th.MAROON
    assert th.status_style("Awaiting Review")[2] == th.GOLD and th.status_style("unknown")[0] == th.NEUTRAL_BG


def test_badge_and_table_escape_html():
    assert "<script>" not in th.badge("<script>alert(1)</script>")
    t = th.html_table([{"a": "<b>x</b>", "risk_level": "High"}], [("a", "A"), ("risk_level", "Risk")])
    assert "<b>x</b>" not in t and "&lt;b&gt;" in t and "ksg-badge" in t


def test_missing_logo_is_harmless(monkeypatch, tmp_path):
    monkeypatch.setattr(th, "LOGO_PATH", tmp_path / "nope.png")
    assert not th.logo_available() and th.page_icon() == "📋"


def test_excel_uses_ksg_green_header():
    core.make.__globals__["d"].init_db()
    eid = core.make(); e = core.d.get_engagement(eid); r = core.d.get_rows(eid)
    ws = openpyxl.load_workbook(io.BytesIO(build_workbook(e, r, [])))["Working Plan"]
    assert ws["A1"].fill.fgColor.rgb.endswith("006B3F") and ws.auto_filter.ref
