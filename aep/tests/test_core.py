import io, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest, openpyxl, docx
import database as d
from audit_templates import TEMPLATES
from awp_generator import generate_rows, count_missing
from export_excel import build_workbook
from export_word import build_document

AUD = {"username": "auditor", "role": "Auditor"}
REV = {"username": "reviewer", "role": "Reviewer"}


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    for n, v in (("DATA_DIR", tmp_path), ("DB_PATH", tmp_path / "t.db"), ("EVIDENCE_DIR", tmp_path / "ev"), ("BACKUP_DIR", tmp_path / "bk")):
        monkeypatch.setattr(d, n, v)
    d.init_db()


def make():
    eid = d.create_engagement({"title": "T", "org": "O", "category": "ICT", "objective": "obj"}, AUD)
    e = d.get_engagement(eid)
    d.replace_rows(eid, generate_rows(e, TEMPLATES["ICT"][:3]), AUD)
    return eid


def test_create_validate_and_unique_ref():
    with pytest.raises(ValueError): d.create_engagement({"title": "", "org": "O", "category": "ICT"}, AUD)
    d.create_engagement({"title": "T", "org": "O", "category": "ICT", "ref": "X1"}, AUD)
    with pytest.raises(ValueError): d.create_engagement({"title": "T", "org": "O", "category": "ICT", "ref": "X1"}, AUD)


def test_rows_persist_and_missing_flagged():
    eid = make(); rows = d.get_rows(eid)
    assert len(rows) == 3 and count_missing(rows) > 0 and "[VERIFY" in rows[0]["criteria"]


def test_workflow_and_permissions():
    eid = make()
    with pytest.raises(PermissionError): d.set_status(eid, "Approved", AUD)
    d.set_status(eid, "Awaiting Review", AUD)
    with pytest.raises(PermissionError): d.replace_rows(eid, [], AUD)  # locked
    d.set_status(eid, "Approved", REV)
    assert d.get_engagement(eid)["approved_by"] == "reviewer"
    d.new_version(eid, "rev", AUD)
    e = d.get_engagement(eid)
    assert e["version"] == 2 and e["status"] == "Draft" and len(d.list_versions(eid)) == 1


def test_exports_open():
    eid = make(); e = d.get_engagement(eid); r = d.get_rows(eid)
    ws = openpyxl.load_workbook(io.BytesIO(build_workbook(e, r, [])))["Working Plan"]
    assert ws.max_row == 4 and ws.freeze_panes == "B2"
    assert "NOT APPROVED" in "\n".join(p.text for t in docx.Document(io.BytesIO(build_document(e, r, []))).tables for c in t._cells for p in c.paragraphs)


def test_backup_restore_and_stats():
    eid = make(); p = d.backup_db()
    d.delete_engagement(eid, {"username": "admin", "role": "Administrator"})
    assert d.stats()["total"] == 0
    d.restore_db(p.read_bytes(), {"username": "admin", "role": "Administrator"})
    assert d.stats()["total"] == 1
    with pytest.raises(ValueError): d.restore_db(b"junk", {"username": "admin", "role": "Administrator"})
