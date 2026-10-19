import io, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest, openpyxl
from conftest_helpers import AUD, REV, ADM, use_tmp_db, d
import plan_register as pr, followup as fu
import transport_analysis as ta


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    use_tmp_db(tmp_path, monkeypatch)


def test_fuel_efficiency_values_and_bad_input():
    assert ta.fuel_efficiency(450, 50)["value"] == 9.0
    assert ta.fuel_efficiency("1,000", "80")["value"] == 12.5
    for k, l, st in ((None, 5, ta.INSUFFICIENT), ("", "", ta.INSUFFICIENT), ("abc", 4, ta.INSUFFICIENT),
                     (100, 0, ta.INVALID), (-1, 5, ta.INVALID), (True, 5, ta.INSUFFICIENT), (float("nan"), 5, ta.INSUFFICIENT)):
        r = ta.fuel_efficiency(k, l); assert r["value"] is None and r["status"] == st


def test_fuel_analysis_flags_for_investigation_not_findings():
    rows = ta.analyse_fuel([{"vehicle": "A", "km": 100, "litres": 10}, {"vehicle": "B", "km": 100, "litres": 40}, {"vehicle": "C", "km": None, "litres": 5}],
                           expected_kmpl=10, tolerance_pct=20)
    assert [r["status"] for r in rows] == [ta.OK, ta.FLAG, ta.INSUFFICIENT]
    assert "explanation" in rows[1]["note"] and "finding" not in rows[1]["note"].lower()


def test_fleet_utilisation():
    assert ta.fleet_utilisation(8, 10)["value"] == 80.0
    assert ta.fleet_utilisation(1, 3)["value"] == 33.3
    assert ta.fleet_utilisation(5, 0)["status"] == ta.INVALID and ta.fleet_utilisation(11, 10)["status"] == ta.INVALID
    assert ta.fleet_utilisation(None, 10)["status"] == ta.INSUFFICIENT and ta.fleet_utilisation(2.5, 10)["status"] == ta.INVALID


def test_maintenance_cost_period_and_rejects():
    recs = [{"vehicle": "A", "date": "2099-01-10", "amount": 100}, {"vehicle": "A", "date": "2099-02-10", "amount": "50.5"},
            {"vehicle": "B", "date": "2098-12-31", "amount": 999}, {"vehicle": "", "date": "2099-01-01", "amount": 5},
            {"vehicle": "B", "date": "2099-01-15", "amount": "x"}]
    r = ta.maintenance_cost(recs, "2099-01-01", "2099-06-30")
    assert r["per_vehicle"] == {"A": 150.5} and r["total"] == 150.5 and len(r["rejected"]) == 2 and "2099-01-01" in r["period"]
    with pytest.raises(ValueError): ta.maintenance_cost(recs, "", "2099-06-30")
    with pytest.raises(ValueError): ta.maintenance_cost(recs, "2099-07-01", "2099-06-30")


def test_multiple_fuel_transactions_flagged():
    rows = [{"vehicle": "A", "date": "2099-01-01"}, {"vehicle": "A", "date": "2099-01-01"}, {"vehicle": "B", "date": "2099-01-01"}]
    f = ta.flag_multiple_fuel(rows)
    assert len(f) == 1 and f[0]["vehicle"] == "A" and f[0]["status"] == ta.FLAG


def test_analysis_workbook_has_formulas_and_blank_inputs():
    ws = openpyxl.load_workbook(io.BytesIO(ta.build_analysis_workbook()))
    assert ws["Fuel efficiency"]["E2"].value.startswith("=IF") and ws["Fuel efficiency"]["C2"].value is None
    assert "Fleet utilisation" in ws.sheetnames and "Maintenance cost" in ws.sheetnames


def test_followup_separate_statuses_and_workflow():
    eid = pr.create_engagement({"financial_year": "2099/2100", "ref": "T-1", "title": "T"}, AUD)
    fid = fu.create_followup({"annual_engagement_id": eid, "original_recommendation": "Reconcile fuel monthly"}, AUD)   # no finding id needed
    r = fu.get_followup(fid)
    assert r["mgmt_status"] == "Not Started" and r["auditor_status"] == "Unable to Verify" and r["finding_ref"] == ""
    fu.update_followup(fid, {"mgmt_status": "Implemented", "mgmt_update_date": "2099-03-01"}, AUD)
    r = fu.get_followup(fid)
    assert r["mgmt_status"] == "Implemented" and r["auditor_status"] == "Unable to Verify"      # management claim does not set auditor status
    with pytest.raises(ValueError): fu.update_followup(fid, {"auditor_status": "Done"}, AUD)
    with pytest.raises(ValueError): fu.update_followup(fid, {"mgmt_update_date": "01/03/2099"}, AUD)
    with pytest.raises(ValueError): fu.create_followup({"original_recommendation": " "}, AUD)
    with pytest.raises(ValueError): fu.submit(fid, AUD)                                          # needs conclusion + date
    fu.update_followup(fid, {"auditor_status": "Partially Implemented", "followup_conclusion": "Partly", "followup_date": "2099-03-05"}, AUD)
    fu.submit(fid, AUD)
    with pytest.raises(PermissionError): fu.update_followup(fid, {"agreed_action": "x"}, AUD)
    with pytest.raises(PermissionError): fu.review(fid, "Reviewed", "ok", AUD)
    fu.review(fid, "Reviewed", "ok", REV)
    assert fu.get_followup(fid)["reviewer"] == "reviewer" and len(fu.list_followups(eid)) == 1
