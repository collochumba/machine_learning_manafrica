"""Headless Streamlit smoke tests (streamlit.testing AppTest). This drives the real app.py script without a browser, so it
checks that screens render and the main flows work; it is NOT a substitute for a manual browser check (see README)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from streamlit.testing.v1 import AppTest
from conftest_helpers import ADM, AUD, d, use_tmp_db
import aep

APP = str(pathlib.Path(__file__).resolve().parents[1] / "app.py")
SECS = [{"heading": "HR & Payroll Management", "rows": [{"area": "Payroll", "requirements": "R", "procedure": "P", "control": "C"}]}]


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AEP_DRIVE_AUTH", "none")  # Drive not configured -> manual fallback path
    monkeypatch.delenv("AWP_DEFAULT_PASSWORD", raising=False)
    use_tmp_db(tmp_path, monkeypatch)
    import streamlit as st
    st.cache_resource.clear()


def run(user=None):
    at = AppTest.from_file(APP, default_timeout=30)
    if user: at.session_state["user"] = dict(user)
    return at.run()


def texts(at):
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info])


def test_first_run_shows_setup_not_default_password():
    at = run()
    assert not at.exception and "no default password" in texts(at).lower() and "ChangeMe" not in texts(at)


def test_login_screen_after_setup_and_wrong_password():
    d.create_first_admin("boss", "a-long-unique-pass")
    at = run(); assert not at.exception and at.text_input[0].label == "Username"
    at.text_input[0].set_value("boss"); at.text_input[1].set_value("bad"); at.button[0].click(); at.run()
    assert "Invalid username or password" in texts(at)


def test_list_export_and_two_step_delete_flow():
    d.create_first_admin("boss", "a-long-unique-pass")
    pid = aep.save_program({"title": "Payroll review", "financial_year": "2026/2027", "status": "Draft"}, SECS, AUD)
    at = run(ADM)
    assert not at.exception and "AUDIT ENGAGEMENT PROGRAMS" in texts(at) and "AEP-2026-001" in texts(at)
    assert {"View", "Edit", "Export", "Delete"} <= {b.label for b in at.button}
    assert "Prepared" not in texts(at)
    # export screen: Drive unavailable -> honest manual-upload message, download still produced
    next(b for b in at.button if b.label == "Export").click(); at.run()
    assert not at.exception and "NOT available" in texts(at)
    next(b for b in at.button if b.label == "Generate export").click(); at.run()
    assert not at.exception and "not configured" in texts(at).lower() and "uploaded to Google Drive" not in texts(at)
    next(b for b in at.button if b.label.startswith("← Back")).click(); at.run()
    # delete step A: disabled until confirmed; cancel changes nothing
    next(b for b in at.button if b.label == "Delete").click(); at.run()
    assert "cannot be undone" in texts(at) and "Payroll review" in texts(at)
    assert next(b for b in at.button if b.label == "Permanently delete program").disabled
    next(b for b in at.button if b.label == "Cancel").click(); at.run()
    assert len(aep.list_programs()) == 1
    next(b for b in at.button if b.label == "Delete").click(); at.run()
    at.text_input[0].set_value("AEP-2026-001"); at.checkbox[0].check(); at.run()
    btn = next(b for b in at.button if b.label == "Permanently delete program"); assert not btn.disabled
    btn.click(); at.run()
    assert not at.exception and aep.list_programs() == [] and "permanently deleted from the application database" in texts(at)


def test_auditor_cannot_see_enabled_delete():
    d.create_first_admin("boss", "a-long-unique-pass")
    aep.save_program({"title": "T", "financial_year": "2026/2027", "status": "Draft"}, SECS, AUD)
    at = run(AUD)
    assert next(b for b in at.button if b.label == "Delete").disabled and not next(b for b in at.button if b.label == "Export").disabled
