import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from conftest_helpers import AUD, REV, use_tmp_db, d
import aep, migrations


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    use_tmp_db(tmp_path, monkeypatch)


def meta(**kw):
    return {"title": "Review of ICT Policies", "financial_year": "2026/2027", "campus": "Nairobi",
            "audit_period": "FY", "status": "In Progress"} | kw


SECS = [{"heading": aep.AUDIT_AREAS[0], "rows": [
            {"area": "ICT Governance", "requirements": "R1", "procedure": "P1", "control": "C1"},
            {"area": "Access", "requirements": "R2", "procedure": "P2", "control": "C2"}, {"area": "", "procedure": ""}]},
        {"heading": aep.AUDIT_AREAS[1], "rows": [{"area": "Assets", "procedure": "P3"}]}]


def test_standalone_create_view_edit_and_counts():
    pid = aep.save_program(meta(), SECS, AUD)  # no plan or engagement needed
    p = aep.get_program(pid)
    assert p["program_ref"] == "AEP-2026-001" and [len(s["rows"]) for s in p["sections"]] == [2, 1]
    assert p["main_audit_area"] == "; ".join(aep.AUDIT_AREAS[:2])
    assert aep.stats() == {"total": 1, "draft": 0, "in_progress": 1, "completed": 0}
    s0 = p["sections"][0]
    secs = [{"id": s0["id"], "heading": "Custom heading", "rows": [{**s0["rows"][1], "procedure": "P2-edited"},
                                                                   {**s0["rows"][0]}, {"area": "New", "procedure": "P9"}]}]
    assert aep.save_program(meta(title="Renamed", status="Completed"), secs, AUD, pid) == pid
    p2 = aep.get_program(pid)
    assert p2["program_ref"] == "AEP-2026-001" and p2["title"] == "Renamed" and len(aep.list_programs()) == 1
    assert [s["heading"] for s in p2["sections"]] == ["Custom heading"]
    assert [r["procedure"] for r in p2["sections"][0]["rows"]] == ["P2-edited", "P1", "P9"]  # reordered + edited + added
    assert p2["sections"][0]["rows"][0]["id"] == s0["rows"][1]["id"]
    assert aep.stats() == {"total": 1, "draft": 0, "in_progress": 0, "completed": 1}


def test_validation_search_filter_unique_ref_permissions():
    with pytest.raises(ValueError): aep.save_program(meta(title=""), [], AUD)
    with pytest.raises(ValueError): aep.save_program(meta(status="Bogus"), [], AUD)
    with pytest.raises(ValueError): aep.save_program(meta(), [{"heading": "", "rows": []}], AUD)
    a = aep.save_program(meta(status="Draft"), [], AUD)
    b = aep.save_program(meta(title="Second"), [{"heading": aep.AUDIT_AREAS[2], "rows": []}], AUD)
    assert aep.get_program(b)["program_ref"] == "AEP-2026-002"
    with pytest.raises(ValueError): aep.save_program(meta() | {"program_ref": "aep-2026-001"}, [], AUD)
    assert [r["id"] for r in aep.list_programs(search="second")] == [b]
    assert [r["id"] for r in aep.list_programs(search="AEP-2026-001")] == [a]
    assert [r["id"] for r in aep.list_programs(status="Draft")] == [a]
    assert [r["id"] for r in aep.list_programs(area=aep.AUDIT_AREAS[2])] == [b]
    assert aep.list_programs(search="nothing") == []
    assert aep.stats()["total"] == 2 and aep.stats()["draft"] == 1
    with pytest.raises(PermissionError): aep.save_program(meta(), [], REV)


def test_migration_7_keeps_programs_made_under_migration_6(tmp_path):
    path = str(tmp_path / "old.db")
    c = sqlite3.connect(path)
    c.executescript("CREATE TABLE annual_engagements(id INTEGER PRIMARY KEY);" + migrations.M6_AEP)
    c.execute("INSERT INTO annual_engagements(id) VALUES(1)")
    c.execute("INSERT INTO aep_programs(program_ref,title,annual_engagement_id,financial_year,status) VALUES('AEP-1','T',1,'2026/2027','Pending Review')")
    c.execute("INSERT INTO aep_sections(program_id,heading) VALUES(1,'H')")
    c.execute("INSERT INTO aep_procedures(section_id,area,procedure) VALUES(1,'A','P')")
    c.execute("PRAGMA user_version=6"); c.commit(); c.close()
    assert migrations.apply(path, lambda: str(tmp_path / "bak.db")) == [7, 8, 9]  # 8 adds version/uid/cascade, 9 adds export + deletion-manifest tables
    c = sqlite3.connect(path)
    assert c.execute("SELECT title,status FROM aep_programs").fetchone() == ("T", "In Progress")
    assert c.execute("SELECT count(*) FROM aep_sections").fetchone()[0] == 1 == c.execute("SELECT count(*) FROM aep_procedures").fetchone()[0]
    c.execute("INSERT INTO aep_programs(uid,program_ref,title,financial_year) VALUES('u2','AEP-2','T2','2026/2027')")  # engagement now optional
