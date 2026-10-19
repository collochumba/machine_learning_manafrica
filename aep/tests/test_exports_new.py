import io, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest, openpyxl, docx
from conftest_helpers import AUD, REV, use_tmp_db, d
import plan_register as pr, working_papers as wp, followup as fu, export_new as ex, theme as th


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    use_tmp_db(tmp_path, monkeypatch)


def doc_text(b):
    D = docx.Document(io.BytesIO(b))
    return "\n".join([p.text for p in D.paragraphs] + [c.text for t in D.tables for r in t.rows for c in r.cells])


def test_working_paper_exports_open_and_show_blank_results_honestly():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T-1", "title": "Fleet"}, AUD)
    w = wp.create_working_paper(eid, AUD, {"procedure": "Agree register"})
    wp.save_working_paper(w, {}, AUD, [{"item_ref": "KAA 1", "result": "Exception - for investigation", "exception_note": "n/a"}])
    got, ann = wp.get_working_paper(w), pr.get_engagement(eid)
    t = doc_text(ex.working_paper_docx(got, ann))
    assert "T-1/WP-001" in t and "[Not yet completed]" in t and "KAA 1" in t and "Sign-off" in t
    wb = openpyxl.load_workbook(io.BytesIO(ex.working_paper_xlsx(got, ann)))
    assert wb.sheetnames == ["Working Paper", "Sample Testing"] and wb["Sample Testing"]["A2"].value == "KAA 1"
    assert wb["Sample Testing"]["A1"].fill.fgColor.rgb.endswith(th.GREEN.lstrip("#"))      # KSG green headers


def test_followup_and_register_exports():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T-1", "title": "Fleet"}, AUD, is_test=True)
    f = fu.create_followup({"annual_engagement_id": eid, "original_recommendation": "Do X"}, AUD)
    t = doc_text(ex.followup_docx(fu.get_followup(f), pr.get_engagement(eid)))
    assert "MANAGEMENT-REPORTED status" in t and "AUDITOR-VERIFIED status" in t and "Do X" in t
    ws = openpyxl.load_workbook(io.BytesIO(ex.followup_xlsx(fu.list_followups(), {eid: pr.get_engagement(eid)}))).active
    assert ws["C2"].value == "Do X" and ws.max_row == 2
    reg = openpyxl.load_workbook(io.BytesIO(ex.register_xlsx(pr.list_engagements()))).active
    assert "TEST" in reg["P1"].value and "not an official" in reg["P2"].value


def test_file_names_are_sensible():
    assert ex.safe_name("AWP/ICT/2026", "WP 001", "working paper", ext="docx") == "AWP-ICT-2026_WP-001_working-paper.docx"
