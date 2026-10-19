import sqlite3, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from conftest_helpers import AUD, REV, ADM, use_tmp_db, d
import plan_register as pr, template_lib as tl, awp_builder as ab, working_papers as wp, followup as fu, migrations


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    monkeypatch.setenv("AWP_DEFAULT_PASSWORD", "Test-Password-123")
    use_tmp_db(tmp_path, monkeypatch)


def test_existing_authentication_and_password_hashing_unchanged():
    assert d.authenticate("admin", "Test-Password-123")["role"] == "Administrator"
    assert d.authenticate("admin", "wrong") is None and d.authenticate("nobody", "x") is None
    with d.connect() as c:
        row = c.execute("SELECT salt,pw_hash FROM users WHERE username='admin'").fetchone()
    assert "Test-Password-123" not in (row["salt"] + row["pw_hash"]) and len(row["pw_hash"]) == 64
    with pytest.raises(PermissionError): d.save_user("x", "longenough-12", "Auditor", AUD)
    d.save_user("newbie", "longenough-12", "Auditor", ADM)
    assert d.authenticate("newbie", "longenough-12")["username"] == "newbie"


def test_new_module_permissions_enforced_at_data_layer():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T", "title": "T"}, AUD)
    tid = tl.create_template("T", {"procedures": [{"procedure": "p"}]}, AUD)
    awp = ab.save_awp(eid, tid, AUD)
    d.set_status(awp, "Awaiting Review", AUD); d.set_status(awp, "Approved", REV)
    w = wp.create_from_awp(awp, AUD)[0]
    for call in (lambda: pr.create_engagement({"financial_year": "x", "ref": "y", "title": "z"}, REV),
                 lambda: tl.create_template("N", {}, REV), lambda: ab.save_awp(eid, tid, REV),
                 lambda: wp.create_working_paper(eid, REV, {"procedure": "p"}), lambda: wp.save_working_paper(w, {}, REV),
                 lambda: fu.create_followup({"original_recommendation": "r"}, REV), lambda: pr.save_area({"name": "A"}, AUD)):
        with pytest.raises(PermissionError): call()


def test_backup_restore_roundtrip_keeps_new_module_data():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T", "title": "T"}, AUD)
    wid = wp.create_working_paper(eid, AUD, {"procedure": "p"})
    bak = d.backup_db()
    wp_ver = migrations.current_version(bak)
    assert wp_ver == migrations.LATEST
    with d.connect() as c: c.execute("DELETE FROM working_papers"); c.execute("DELETE FROM annual_engagements")
    d.restore_db(bak.read_bytes(), ADM)
    assert pr.get_engagement(eid)["ref"] == "T" and wp.get_working_paper(wid)["procedure"] == "p"


def test_restoring_an_old_backup_upgrades_it_and_keeps_data():
    old = d.DATA_DIR / "old.db"
    c = sqlite3.connect(old); c.executescript(d.SCHEMA)
    c.execute("INSERT INTO engagements(ref,title,org,category) VALUES('OLD','Old','O','ICT')"); c.commit(); c.close()
    d.restore_db(old.read_bytes(), ADM)
    assert migrations.current_version(d.DB_PATH) == migrations.LATEST and d.list_engagements()[0]["ref"] == "OLD"


def test_audit_log_records_material_changes():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T", "title": "T"}, AUD)
    acts = [r["action"] for r in d.get_log()]
    assert "create_annual_engagement" in acts and "migrate" in acts


def test_error_messages_do_not_leak_paths_or_sql():
    with pytest.raises(ValueError) as e: wp.get_working_paper(999)
    assert "sqlite" not in str(e.value).lower() and str(d.DATA_DIR) not in str(e.value)
