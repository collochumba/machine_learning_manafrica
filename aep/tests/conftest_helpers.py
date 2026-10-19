"""Shared helpers for the new test modules (temp database per test)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import database as d

AUD = {"username": "auditor", "role": "Auditor"}
REV = {"username": "reviewer", "role": "Reviewer"}
ADM = {"username": "admin", "role": "Administrator"}


def use_tmp_db(tmp_path, monkeypatch):
    for n, v in (("DATA_DIR", tmp_path), ("DB_PATH", tmp_path / "t.db"), ("EVIDENCE_DIR", tmp_path / "ev"),
                 ("BACKUP_DIR", tmp_path / "bk")):
        monkeypatch.setattr(d, n, v)
    d.init_db()
