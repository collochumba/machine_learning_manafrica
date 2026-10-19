"""Audit Engagement Program (AEP): standalone data layer; needs no annual plan or engagement.

Tables (migrations 6-7): aep_programs -> aep_sections (main audit area headings) -> aep_procedures (rows).
Rows keep their ids on edit; removed sections/rows are deleted explicitly inside one transaction.
Concurrency: every program has a `version`; saving with a stale expected_version raises ConflictError (optimistic lock).
Deletion: delete_program() removes the program + dependent rows in one transaction and writes a manifest of its Drive
file ids to aep_deletions / aep_deletion_files so Drive clean-up can be done (or retried) separately."""
import sqlite3
import uuid

import database as db

AUDIT_AREAS = ["Information & Communication Technology", "Administration, Asset & Project Management",
               "HR & Payroll Management", "Training, Research, & Consultancy",
               "Finance & Accounts - Receivables", "Hospitality Management", "Supply Chain Management",
               "Finance & Accounts - Payables"]
STATUSES = ["Draft", "In Progress", "Completed"]
ROW_FIELDS = ["area", "requirements", "procedure", "control"]
META = ["title", "financial_year", "campus", "audit_period", "main_audit_area", "status"]


class ConflictError(ValueError):
    """The program was changed by someone else after the editor opened it."""


def _t(v):
    return "" if v is None else str(v).strip()


def letter(i):
    return chr(ord("A") + i) if i < 26 else str(i + 1)


def next_ref(fy):
    """AEP-<year>-NNN, where year is the start year of the financial year (e.g. 2026/2027 -> 2026).
    References of deleted programs (kept in the deletion manifest) are never reused."""
    yr = (_t(fy)[:4] or "0000")
    with db.connect() as c:
        n = c.execute("SELECT COUNT(*) FROM aep_programs WHERE program_ref LIKE ?", (f"AEP-{yr}-%",)).fetchone()[0] + 1
        while (c.execute("SELECT 1 FROM aep_programs WHERE program_ref=?", (f"AEP-{yr}-{n:03d}",)).fetchone()
               or c.execute("SELECT 1 FROM aep_deletions WHERE program_ref=? COLLATE NOCASE", (f"AEP-{yr}-{n:03d}",)).fetchone()):
            n += 1
        return f"AEP-{yr}-{n:03d}"


def stats(financial_year=None):
    q, p = "SELECT status, COUNT(*) FROM aep_programs", []
    if financial_year:
        q += " WHERE financial_year=?"; p.append(financial_year)
    with db.connect() as c:
        got = dict(c.execute(q + " GROUP BY status", p).fetchall())
    return {"total": sum(got.values()), "draft": got.get("Draft", 0), "in_progress": got.get("In Progress", 0),
            "completed": got.get("Completed", 0)}


def financial_years():
    with db.connect() as c:
        ys = {r[0] for r in c.execute("SELECT DISTINCT financial_year FROM aep_programs")}
    return sorted(ys, reverse=True)


def heading_options():
    """Official main audit areas plus any other headings already used in saved programs."""
    with db.connect() as c:
        used = [r[0] for r in c.execute("SELECT DISTINCT heading FROM aep_sections ORDER BY heading")]
    return AUDIT_AREAS + [h for h in used if h not in AUDIT_AREAS]


def delete_section(section_id, user):
    """Delete one section and its rows in a single transaction."""
    db._need(user, "edit")
    with db.connect() as c:
        c.execute("DELETE FROM aep_sections WHERE id=?", (section_id,))  # rows cascade
        db._log(c, user, "delete_aep_section", str(section_id))


def list_programs(financial_year=None, area=None, status=None, search=""):
    q, pr = "SELECT p.* FROM aep_programs p WHERE 1=1", []
    for col, v in (("p.financial_year", financial_year), ("p.status", status)):
        if v:
            q += f" AND {col}=?"; pr.append(v)
    if area:
        q += " AND EXISTS(SELECT 1 FROM aep_sections s WHERE s.program_id=p.id AND s.heading=?)"; pr.append(area)
    if _t(search):
        q += " AND (p.program_ref LIKE ? OR p.title LIKE ?)"; pr += [f"%{_t(search)}%"] * 2
    with db.connect() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY p.program_ref", pr)]


def get_program(pid):
    with db.connect() as c:
        p = c.execute("SELECT * FROM aep_programs WHERE id=?", (pid,)).fetchone()
        if not p:
            raise ValueError("Audit Engagement Program not found.")
        p = dict(p)
        p["sections"] = []
        for s in c.execute("SELECT * FROM aep_sections WHERE program_id=? ORDER BY position,id", (pid,)):
            s = dict(s)
            s["rows"] = [dict(r) for r in c.execute("SELECT * FROM aep_procedures WHERE section_id=? ORDER BY position,id", (s["id"],))]
            p["sections"].append(s)
        return p


def _validate(data, sections):
    d = {f: _t(data.get(f)) for f in META}
    for f, lab in (("title", "Program title"), ("financial_year", "Financial year")):
        if not d[f]:
            raise ValueError(f"{lab} is required.")
    if d["status"] not in STATUSES:
        raise ValueError("Select a valid status.")
    clean = []
    for s in sections or []:
        heading = _t(s.get("heading"))
        if not heading:
            raise ValueError("Every audit area section needs a heading.")
        rows = [{f: _t(r.get(f)) for f in ROW_FIELDS} | {"id": r.get("id")} for r in s.get("rows", [])]
        rows = [r for r in rows if any(r[f] for f in ROW_FIELDS)]  # blank rows are dropped
        clean.append({"id": s.get("id"), "heading": heading, "rows": rows})
    return d, clean


def save_program(data, sections, user, pid=None, expected_version=None):
    """Create (pid=None) or update a program with its sections and rows in one transaction; returns the id.
    For updates pass expected_version (the version the editor loaded): a mismatch raises ConflictError and nothing is written."""
    db._need(user, "edit")
    d, secs = _validate(data, sections)
    d["main_audit_area"] = "; ".join(s["heading"] for s in secs)  # register column, derived from the headings
    try:
        with db.connect(immediate=True) as c:  # write lock first: the version check and the update are atomic
            if pid is None:
                ref = _t(data.get("program_ref")) or next_ref(d["financial_year"])
                cur = c.execute("INSERT INTO aep_programs(uid,program_ref," + ",".join(META)
                                + ",version,created_by,created_at,updated_at,updated_by) VALUES(?,?," + ",".join("?" * len(META)) + ",1,?,?,?,?)",
                                [uuid.uuid4().hex, ref] + [d[f] for f in META] + [user["username"], db.now(), db.now(), user["username"]])
                pid = cur.lastrowid
            else:
                cur_row = c.execute("SELECT program_ref, version, updated_by, updated_at FROM aep_programs WHERE id=?", (pid,)).fetchone()
                if not cur_row:
                    raise ValueError("Audit Engagement Program not found. It may have been deleted by another user.")
                if expected_version is not None and cur_row["version"] != expected_version:
                    raise ConflictError(
                        f"This program was changed by {cur_row['updated_by'] or 'another user'} at {cur_row['updated_at']} "
                        "after you opened it. Your changes were NOT saved, so nothing was overwritten. "
                        "Reload the latest version, then re-apply your edits.")
                ref = cur_row["program_ref"]
                c.execute("UPDATE aep_programs SET " + ",".join(f + "=?" for f in META)
                          + ",updated_at=?,updated_by=?,version=version+1 WHERE id=?",
                          [d[f] for f in META] + [db.now(), user["username"], pid])
            keep_s = {s["id"] for s in secs if s["id"]}
            for sid, in c.execute("SELECT id FROM aep_sections WHERE program_id=?", (pid,)).fetchall():
                if sid not in keep_s:
                    c.execute("DELETE FROM aep_sections WHERE id=?", (sid,))  # rows go with it (ON DELETE CASCADE)
            for i, s in enumerate(secs):
                if s["id"]:
                    if not c.execute("SELECT 1 FROM aep_sections WHERE id=? AND program_id=?", (s["id"], pid)).fetchone():
                        raise ValueError("A section does not belong to this program.")
                    c.execute("UPDATE aep_sections SET heading=?,position=? WHERE id=?", (s["heading"], i, s["id"]))
                    sid = s["id"]
                else:
                    sid = c.execute("INSERT INTO aep_sections(program_id,heading,position) VALUES(?,?,?)", (pid, s["heading"], i)).lastrowid
                keep_r = {r["id"] for r in s["rows"] if r["id"]}
                for rid, in c.execute("SELECT id FROM aep_procedures WHERE section_id=?", (sid,)).fetchall():
                    if rid not in keep_r:
                        c.execute("DELETE FROM aep_procedures WHERE id=?", (rid,))
                for j, r in enumerate(s["rows"]):
                    vals = [r[f] for f in ROW_FIELDS]
                    if r["id"]:
                        n = c.execute("UPDATE aep_procedures SET position=?," + ",".join(f + "=?" for f in ROW_FIELDS)
                                      + " WHERE id=? AND section_id=?", [j] + vals + [r["id"], sid]).rowcount
                        if not n:
                            raise ValueError("A procedure row does not belong to this section.")
                    else:
                        c.execute("INSERT INTO aep_procedures(section_id,position," + ",".join(ROW_FIELDS) + ") VALUES(?,?,?,?,?,?)",
                                  [sid, j] + vals)
            db._log(c, user, "save_aep", ref)
            return pid
    except sqlite3.IntegrityError as e:
        if "program_ref" in str(e):
            raise ValueError("That program reference already exists.") from None
        raise ValueError("The program could not be saved (database constraint).") from None


def delete_program(pid, user, confirmed=False):
    """STEP A of deletion. Permanently removes the program, its sections and procedure rows (one transaction).
    Never touches Google Drive. Records a manifest (aep_deletions + aep_deletion_files) of the program's Drive file ids so
    the separate, separately-confirmed Drive clean-up can run or be retried. Returns {"deletion_id","program_ref","drive_files"}."""
    db._need(user, "delete")
    if confirmed is not True:
        raise ValueError("Deletion was not confirmed. Nothing was deleted.")
    with db.connect(immediate=True) as c:
        p = c.execute("SELECT id, uid, program_ref, title FROM aep_programs WHERE id=?", (pid,)).fetchone()
        if not p:
            raise ValueError("Audit Engagement Program not found. It may already have been deleted.")
        files = c.execute("SELECT drive_file_id, filename FROM aep_exports WHERE program_uid=? AND drive_file_id IS NOT NULL "
                          "AND drive_status='uploaded'", (p["uid"],)).fetchall()
        did = c.execute("INSERT INTO aep_deletions(program_uid,program_ref,program_title,deleted_by,deleted_at,drive_decision) VALUES(?,?,?,?,?,?)",
                        (p["uid"], p["program_ref"], p["title"], user["username"], db.now(), "pending" if files else "none")).lastrowid
        for f in files:
            c.execute("INSERT INTO aep_deletion_files(deletion_id,drive_file_id,filename,status,updated_at) VALUES(?,?,?,?,?)",
                      (did, f["drive_file_id"], f["filename"], "retained", db.now()))
        c.execute("DELETE FROM aep_exports WHERE program_uid=?", (p["uid"],))
        c.execute("DELETE FROM aep_programs WHERE id=?", (pid,))  # sections + rows cascade; other programs untouched
        db._log(c, user, "delete_aep", p["program_ref"])  # audit log holds the reference only, no audit content
    return {"deletion_id": did, "program_ref": p["program_ref"], "drive_files": [dict(f) for f in files]}


def count_rows(pid):
    with db.connect() as c:
        return c.execute("SELECT COUNT(*) FROM aep_procedures r JOIN aep_sections s ON s.id=r.section_id WHERE s.program_id=?",
                         (pid,)).fetchone()[0]
