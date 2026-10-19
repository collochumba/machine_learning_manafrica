import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from conftest_helpers import AUD, REV, ADM, use_tmp_db, d
import plan_register as pr, template_lib as tl, awp_builder as ab, working_papers as wp

PROC = {"procedures": [{"title": "Fleet register", "procedure": "Agree register", "evidence_required": "Register"},
                       {"title": "Fuel", "procedure": "Reconcile fuel", "evidence_required": "Fuel records"}]}


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    use_tmp_db(tmp_path, monkeypatch)


def approved_awp():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T-1", "title": "Fleet"}, AUD)
    tid = tl.create_template("Fleet", PROC, AUD)
    awp = ab.save_awp(eid, tid, AUD)
    d.set_status(awp, "Awaiting Review", AUD); d.set_status(awp, "Approved", REV)
    return eid, awp


def test_generate_from_approved_awp_only_and_no_prefilled_results():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T-2", "title": "X"}, AUD)
    awp = ab.save_awp(eid, tl.create_template("F2", PROC, AUD), AUD)
    with pytest.raises(ValueError): wp.create_from_awp(awp, AUD)                    # still Draft
    eid, awp = approved_awp()
    ids = wp.create_from_awp(awp, AUD)
    assert len(ids) == 2
    w = wp.get_working_paper(ids[0])
    assert w["procedure"] == "Agree register" and w["documents_required"] == "Register" and w["awp_id"] == awp
    for f in ("work_performed", "evidence_refs", "test_results", "exceptions", "analysis", "conclusion"):
        assert w[f] == ""                                                           # never pre-populated
    assert wp.create_from_awp(awp, AUD) == []                                       # no duplicates on re-run


def test_unique_reference_per_engagement_not_global():
    e1 = pr.create_engagement({"financial_year": "2099/2100", "ref": "A", "title": "A"}, AUD)
    e2 = pr.create_engagement({"financial_year": "2099/2100", "ref": "B", "title": "B"}, AUD)
    wp.create_working_paper(e1, AUD, {"wp_ref": "WP-1", "procedure": "p"})
    with pytest.raises(ValueError): wp.create_working_paper(e1, AUD, {"wp_ref": "WP-1", "procedure": "p"})
    wp.create_working_paper(e2, AUD, {"wp_ref": "WP-1", "procedure": "p"})        # same ref, other engagement: allowed
    auto = wp.create_working_paper(e1, AUD, {"procedure": "p"})
    assert wp.get_working_paper(auto)["wp_ref"] == "A/WP-002"
    with pytest.raises(ValueError): wp.save_working_paper(auto, {"wp_ref": "WP-1"}, AUD)


def test_save_reopen_samples_versions_submit_review_cycle():
    eid, awp = approved_awp()
    w = wp.create_from_awp(awp, AUD)[0]
    with pytest.raises(ValueError): wp.submit_for_review(w, AUD)                    # nothing performed yet
    samples = [{"item_ref": "KAA 001A", "result": "No exception", "evidence_ref": "E1", "attributes": {"logbook": "yes"}},
               {"item_ref": "KAA 002B", "result": "Exception - for investigation", "exception_note": "No logbook entry"}]
    wp.save_working_paper(w, {"work_performed": "Tested 2 vehicles", "conclusion": "See exceptions"}, AUD, samples)
    got = wp.get_working_paper(w)
    assert got["version"] == 2 and len(got["samples"]) == 2 and got["samples"][0]["attributes"] == {"logbook": "yes"}
    with pytest.raises(ValueError): wp.save_working_paper(w, {}, AUD, [{"result": "Exception - for investigation"}])  # needs a note
    with pytest.raises(ValueError): wp.save_working_paper(w, {}, AUD, [{"result": "Bad"}])
    wp.submit_for_review(w, AUD)
    assert wp.get_working_paper(w)["status"] == "Awaiting Review"
    with pytest.raises(PermissionError): wp.save_working_paper(w, {"conclusion": "x"}, AUD)   # locked
    with pytest.raises(PermissionError): wp.review(w, "Reviewed", "", AUD)                    # auditor cannot review
    with pytest.raises(ValueError): wp.review(w, "Returned", "", REV)
    wp.review(w, "Returned", "Add evidence references", REV)
    r = wp.get_working_paper(w)
    assert r["status"] == "Returned" and r["reviewer"] == "reviewer" and r["review_date"] and "evidence" in r["reviewer_comments"]
    wp.save_working_paper(w, {"evidence_refs": "E1,E2"}, AUD)                                  # editable again after return
    wp.submit_for_review(w, AUD); wp.review(w, "Reviewed", "OK", REV)
    assert wp.get_working_paper(w)["status"] == "Reviewed" and len(wp.list_versions(w)) >= 5


def test_engagement_with_working_papers_cannot_be_deleted():
    eid, awp = approved_awp(); wp.create_from_awp(awp, AUD)
    with pytest.raises(ValueError): pr.delete_engagement(eid, ADM)
    with pytest.raises(ValueError): d.delete_engagement(awp, ADM)
