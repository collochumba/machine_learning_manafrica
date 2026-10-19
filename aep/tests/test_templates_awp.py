import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from conftest_helpers import AUD, REV, ADM, use_tmp_db, d
import plan_register as pr, template_lib as tl, awp_builder as ab


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    use_tmp_db(tmp_path, monkeypatch)


CONTENT = {"objective": "Template objective", "scope": "Template scope", "sampling_methodology": "Judgemental (auditor to confirm)",
           "criteria": [{"text": "Fleet records are complete", "source": "Test source", "clause": "", "verification": "bogus"}],
           "procedures": [{"title": "Fleet register", "procedure": "Agree register to physical vehicles", "evidence_required": "Register"},
                          {"title": "Fuel", "procedure": "Reconcile fuel", "evidence_required": "Fuel records"}]}


def setup_eng(**kw):
    return pr.create_engagement({"financial_year": "2099/2100", "ref": "T-1", "title": "Fleet", "department": "Dept", **kw}, AUD)


def test_template_create_edit_versions_duplicate_archive():
    tid = tl.create_template("Fleet", CONTENT, AUD, family="Transport")
    assert tl.get_template(tid)["content"]["criteria"][0]["verification"] == "Unverified"   # unknown -> never "Verified"
    with pytest.raises(ValueError): tl.create_template("fleet", CONTENT, AUD)
    v = tl.update_template(tid, {**CONTENT, "objective": "Changed"}, AUD, note="edit")
    assert v == 2 and tl.get_template(tid)["content"]["objective"] == "Changed"
    assert tl.get_template(tid, 1)["content"]["objective"] == "Template objective"           # old version intact
    assert [x["version"] for x in tl.list_versions(tid)] == [2, 1]
    dup = tl.duplicate_template(tid, "Fleet copy", AUD)
    assert tl.get_template(dup)["version"] == 1 and tl.get_template(dup)["content"]["objective"] == "Changed"
    tl.set_status(tid, "Archived", AUD)
    assert tid not in [t["id"] for t in tl.list_templates()] and tid in [t["id"] for t in tl.list_templates(status=None)]
    with pytest.raises(ValueError): tl.update_template(tid, CONTENT, AUD)
    with pytest.raises(PermissionError): tl.create_template("X", CONTENT, REV)


def test_generate_awp_is_draft_linked_and_flags_unverified():
    eid, tid = setup_eng(objective="Engagement objective"), tl.create_template("Fleet", CONTENT, AUD, family="Transport")
    awp = ab.save_awp(eid, tid, AUD)
    e, rows = d.get_engagement(awp), d.get_rows(awp)
    assert e["status"] == "Draft" and e["approved_by"] is None and e["period"] == "2099/2100"
    assert e["objective"] == "Engagement objective"                                         # engagement wins over template
    assert len(rows) == 2 and rows[0]["status"] == "Not Started" and rows[0]["actual"] == "" and rows[0]["conclusion"] == ""
    assert "UNVERIFIED" in rows[0]["criteria"] and rows[0]["risk_level"] == "Not assessed"   # nothing invented
    assert ab.source_of(awp)["annual_engagement_id"] == eid and ab.awps_for_engagement(eid)[0]["id"] == awp
    d.set_status(awp, "Awaiting Review", AUD)                                              # existing workflow reused
    with pytest.raises(PermissionError): d.set_status(awp, "Approved", AUD)
    d.set_status(awp, "Approved", REV)


def test_template_changes_do_not_alter_existing_awp():
    eid, tid = setup_eng(), tl.create_template("Fleet", CONTENT, AUD, family="Transport")
    awp = ab.save_awp(eid, tid, AUD)
    before = d.get_rows(awp)
    tl.update_template(tid, {**CONTENT, "procedures": [{"title": "New", "procedure": "Completely different"}]}, AUD)
    tl.set_status(tid, "Archived", AUD)
    assert d.get_rows(awp) == before
    snap = ab.source_of(awp)["template_snapshot"]
    assert snap["version"] == 1 and len(snap["content"]["procedures"]) == 2
    with pytest.raises(ValueError): ab.save_awp(eid, tid, AUD)                              # archived template cannot generate


def test_generation_requires_edit_permission_and_unique_refs():
    eid, tid = setup_eng(), tl.create_template("Fleet", CONTENT, AUD)
    with pytest.raises(PermissionError): ab.save_awp(eid, tid, REV)
    a, b = ab.save_awp(eid, tid, AUD), ab.save_awp(eid, tid, AUD)
    assert d.get_engagement(a)["ref"] != d.get_engagement(b)["ref"]


def test_awp_with_links_cannot_be_deleted_silently():
    eid, tid = setup_eng(), tl.create_template("Fleet", CONTENT, AUD)
    awp = ab.save_awp(eid, tid, AUD)
    with pytest.raises(ValueError): d.delete_engagement(awp, ADM)   # awp_source RESTRICT guards it
    assert d.get_engagement(awp) is not None


def test_seed_templates_idempotent_flagged_and_unattributed():
    import seed_templates as st
    made = st.seed(AUD)
    assert len(made) == 18 and st.seed(AUD) == []
    t = [x for x in tl.list_templates() if x["name"].startswith("Transport")][0]
    assert t["source_status"] == st.PENDING and t["area_id"] is not None
    c = tl.get_template(t["id"])["content"]
    assert len(c["procedures"]) == 19 and all("to be mapped" in p["source_ref"] for p in c["procedures"])
    assert c["criteria"][0]["verification"] == "Unverified" and c["objective"] == ""
