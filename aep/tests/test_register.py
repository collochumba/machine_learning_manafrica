import io, sqlite3, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from conftest_helpers import AUD, REV, ADM, use_tmp_db, d
import migrations, plan_register as pr


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    use_tmp_db(tmp_path, monkeypatch)


def test_fresh_db_is_at_latest_version_and_seeds_candidates():
    assert migrations.current_version(d.DB_PATH) == migrations.LATEST
    areas = pr.list_areas()
    assert len(areas) == 8 and all(a["is_candidate"] for a in areas)


def test_migration_of_legacy_db_preserves_data_and_backs_up(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.db"
    c = sqlite3.connect(legacy); c.executescript(d.SCHEMA)
    c.execute("INSERT INTO engagements(ref,title,org,category) VALUES('OLD/1','Old plan','O','ICT')")
    c.execute("INSERT INTO users(username,salt,pw_hash,role) VALUES('admin','00ff','h','Administrator')")
    c.commit(); c.close()
    assert migrations.current_version(legacy) == 0
    monkeypatch.setattr(d, "DB_PATH", legacy)
    d.init_db()
    assert migrations.current_version(legacy) == migrations.LATEST
    assert [e["ref"] for e in d.list_engagements()] == ["OLD/1"]          # existing record preserved
    assert d.authenticate("admin", "x") is None                          # existing user row untouched
    assert list(d.BACKUP_DIR.glob("*pre-migration*.db"))                 # backup taken before migrating
    with d.connect() as cn:
        assert cn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_failed_migration_rolls_back_and_reports(tmp_path, monkeypatch):
    p = tmp_path / "x.db"
    c = sqlite3.connect(p); c.execute("CREATE TABLE audit_areas(id INTEGER)"); c.commit(); c.close()  # collides with migration 1
    with pytest.raises(migrations.MigrationError) as e:
        migrations.apply(p, lambda: "backup-path")
    assert "rolled back" in str(e.value) and migrations.current_version(p) == 0
    c = sqlite3.connect(p)
    assert not c.execute("SELECT 1 FROM sqlite_master WHERE name='annual_engagements'").fetchone()  # nothing half-applied
    c.close()


def test_engagement_crud_validation_and_duplicates():
    base = {"financial_year": "2099/2100", "ref": "T-1", "title": "Test"}
    eid = pr.create_engagement(base, AUD, is_test=True)
    with pytest.raises(ValueError): pr.create_engagement({**base, "ref": "t-1"}, AUD)       # duplicate (case-insensitive)
    with pytest.raises(ValueError): pr.create_engagement({**base, "ref": "T-2", "planned_start": "2099-05-02", "planned_end": "2099-05-01"}, AUD)
    with pytest.raises(ValueError): pr.create_engagement({"financial_year": "", "ref": "T-3", "title": "x"}, AUD)
    pr.create_engagement({**base, "financial_year": "2100/2101"}, AUD)                      # same ref, other year is fine
    pr.update_engagement(eid, {"title": "Renamed", "status": "In Progress"}, AUD)
    assert pr.get_engagement(eid)["title"] == "Renamed"
    with pytest.raises(PermissionError): pr.create_engagement({**base, "ref": "T-9"}, REV)  # reviewer cannot edit


def test_import_preview_flags_duplicates_incomplete_invalid_and_never_overwrites():
    pr.create_engagement({"financial_year": "2099/2100", "ref": "TEST-001", "title": "Pre-existing"}, AUD)
    data = pr.make_test_workbook()
    assert pr.sheet_names("t.xlsx", data)[0] == pr.TEST_SHEET
    headers, rows = pr.read_sheet("t.xlsx", data, pr.TEST_SHEET)
    m = pr.suggest_mapping(headers)
    assert m["ref"] == 0 and m["title"] == 1 and m["department"] == 8
    prev = pr.build_preview(headers, rows, m, "2099/2100", "t.xlsx", pr.TEST_SHEET, is_test=True)
    st = [p["status"] for p in prev]
    assert st == ["Duplicate", "Incomplete", "Duplicate", "Invalid"]
    res = pr.commit_import(prev, [p["row"] for p in prev], AUD)
    assert res["added"] == 1 and len(res["skipped"]) == 3
    assert [e["title"] for e in pr.list_engagements() if e["ref"] == "TEST-001"] == ["Pre-existing"]   # not overwritten
    e = [x for x in pr.list_engagements() if x["ref"] == "TEST-002"][0]
    assert e["is_test_data"] == 1 and e["source_sheet"] == pr.TEST_SHEET and e["source_row"] == 3     # traceable source


def test_import_rejects_bad_files():
    for name, data in (("a.csv", b"x"), ("a.xlsx", b""), ("a.xlsx", b"not a zip"), ("a.xlsx", b"x" * (pr.MAX_UPLOAD + 1))):
        with pytest.raises(ValueError): pr.sheet_names(name, data)


def test_delete_blocked_when_linked_records_exist():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T-1", "title": "T"}, AUD)
    with d.connect() as c:
        c.execute("INSERT INTO followups(annual_engagement_id,original_recommendation) VALUES(?,?)", (eid, "r"))
    with pytest.raises(ValueError) as e: pr.delete_engagement(eid, ADM)
    assert "linked records" in str(e.value)
    with pytest.raises(sqlite3.IntegrityError):                       # DB-level guard, not just app logic
        with d.connect() as c: c.execute("DELETE FROM annual_engagements WHERE id=?", (eid,))
    with pytest.raises(PermissionError): pr.delete_engagement(eid, AUD)
    e2 = pr.create_engagement({"financial_year": "2099/2100", "ref": "T-2", "title": "T"}, AUD)
    pr.delete_engagement(e2, ADM)
    assert pr.get_engagement(e2) is None


def test_area_catalogue_and_linking():
    a = pr.save_area({"name": "Fleet"}, ADM)
    with pytest.raises(ValueError): pr.save_area({"name": "fleet"}, ADM)
    with pytest.raises(PermissionError): pr.save_area({"name": "X"}, AUD)
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T-1", "title": "T"}, AUD)
    ids = [a["id"] if isinstance(a, dict) else a, pr.list_areas()[0]["id"]]
    pr.set_engagement_areas(eid, ids, AUD)
    assert len(pr.get_engagement_areas(eid)) == 2


def test_library_area_links_and_verification_rules():
    d.add_lib({"kind": "Audit criterion", "category": "ICT", "title": "Backups are tested"}, AUD)
    lid = d.list_lib()[0]["id"]
    a = pr.list_areas()[0]["id"]
    pr.set_library_links(lid, [a, a], AUD)
    assert pr.library_links(lid) == [a] and pr.library_for_area(a)[0]["id"] == lid
    assert pr.get_library_meta(lid)["verification_status"] == "Unverified"
    with pytest.raises(ValueError): pr.set_library_meta(lid, "Verified", "", "", AUD)        # no document/clause recorded
    d.update_lib(lid, {"kind": "Audit criterion", "category": "ICT", "title": "Backups are tested", "doc_title": "Doc", "clause": "4.1"}, AUD, False)
    pr.set_library_meta(lid, "Verified", "checked", "Restore succeeds", AUD)
    assert d.list_lib()[0]["verified"] == 1 and pr.get_library_meta(lid)["expected_result"] == "Restore succeeds"
    pr.set_library_meta(lid, "Unverified", "", "", AUD)
    assert d.list_lib()[0]["verified"] == 0
