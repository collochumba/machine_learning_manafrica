"""Module F: individual working papers (own tables; not stored in AWP plan rows).

Planning fields can be pre-filled from an approved AWP procedure. Work performed, results, exceptions,
analysis and conclusions are NEVER pre-populated - they must be entered by the auditor."""
import json
import sqlite3

import database as db
import plan_register as pr

WP_STATUSES = ["Draft", "Awaiting Review", "Returned", "Reviewed"]
WP_LOCKED = ("Awaiting Review", "Reviewed")
SAMPLE_RESULTS = ["Not tested", "No exception", "Exception - for investigation", "Not applicable"]
PLANNING = ["area", "audit_objective", "control_objective", "risk", "criteria", "procedure", "population",
            "sample_selection", "documents_required"]
AUDITOR_FIELDS = ["work_performed", "evidence_refs", "test_results", "exceptions", "analysis", "conclusion",
                  "prepared_by", "prepared_date", "reviewer"]
EDIT_FIELDS = PLANNING + AUDITOR_FIELDS + ["wp_ref"]
REVIEW_FIELDS = ["reviewer", "review_date", "reviewer_comments"]


def _t(v):
    return "" if v is None else str(v).strip()


def _get(c, wid):
    r = c.execute("SELECT * FROM working_papers WHERE id=?", (wid,)).fetchone()
    if not r:
        raise ValueError("Working paper not found.")
    return dict(r)


def next_ref(annual_id):
    with db.connect() as c:
        ann = c.execute("SELECT ref FROM annual_engagements WHERE id=?", (annual_id,)).fetchone()
        if not ann:
            raise ValueError("Engagement not found.")
        n = c.execute("SELECT COUNT(*) FROM working_papers WHERE annual_engagement_id=?", (annual_id,)).fetchone()[0] + 1
        while c.execute("SELECT 1 FROM working_papers WHERE annual_engagement_id=? AND wp_ref=?", (annual_id, f"{ann['ref']}/WP-{n:03d}")).fetchone():
            n += 1
        return f"{ann['ref']}/WP-{n:03d}"


def create_working_paper(annual_id, user, data=None, awp_id=None, awp_wp_ref=""):
    db._need(user, "edit")
    data = data or {}
    ref = _t(data.get("wp_ref")) or next_ref(annual_id)
    vals = {f: _t(data.get(f)) for f in EDIT_FIELDS if f != "wp_ref"}
    vals["prepared_by"] = vals["prepared_by"] or user["username"]
    try:
        with db.connect() as c:
            if not c.execute("SELECT 1 FROM annual_engagements WHERE id=?", (annual_id,)).fetchone():
                raise ValueError("Engagement not found.")
            cols = ["annual_engagement_id", "awp_id", "awp_wp_ref", "wp_ref"] + list(vals) + ["created_by", "created_at", "updated_at"]
            cur = c.execute(f"INSERT INTO working_papers({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                            [annual_id, awp_id, awp_wp_ref, ref] + list(vals.values()) + [user["username"], db.now(), db.now()])
            db._log(c, user, "create_working_paper", ref)
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError(f"Working paper reference '{ref}' already exists for this engagement.") from None


def create_from_awp(awp_id, user, wp_refs=None):
    """Create one working paper per selected procedure of an APPROVED AWP (planning fields only)."""
    db._need(user, "edit")
    awp = db.get_engagement(awp_id)
    if not awp:
        raise ValueError("AWP not found.")
    if awp["status"] != "Approved":
        raise ValueError("Working papers can only be generated from an Approved AWP.")
    with db.connect() as c:
        src = c.execute("SELECT annual_engagement_id FROM awp_source WHERE awp_id=?", (awp_id,)).fetchone()
    if not src or not src["annual_engagement_id"]:
        raise ValueError("This AWP is not linked to an annual engagement; link it before creating working papers.")
    ann_id, made = src["annual_engagement_id"], []
    for r in db.get_rows(awp_id):
        if wp_refs is not None and r["wp_ref"] not in wp_refs:
            continue
        with db.connect() as c:
            if c.execute("SELECT 1 FROM working_papers WHERE awp_id=? AND awp_wp_ref=?", (awp_id, r["wp_ref"])).fetchone():
                continue  # already created for this procedure
        made.append(create_working_paper(ann_id, user, {
            "area": r["area"], "audit_objective": r["objective"], "control_objective": r["control_objective"],
            "risk": r["risk"], "criteria": r["criteria"], "procedure": r["procedure"],
            "sample_selection": r["sampling_method"], "documents_required": r["evidence_required"]},
            awp_id=awp_id, awp_wp_ref=r["wp_ref"]))
    return made


def get_working_paper(wid):
    with db.connect() as c:
        d = _get(c, wid)
        d["samples"] = samples_of(wid, c)
        return d


def list_working_papers(annual_id=None, status=None):
    q, p = "SELECT * FROM working_papers WHERE 1=1", []
    if annual_id:
        q += " AND annual_engagement_id=?"; p.append(annual_id)
    if status:
        q += " AND status=?"; p.append(status)
    with db.connect() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY annual_engagement_id, wp_ref", p)]


def samples_of(wid, c=None):
    def q(cn):
        return [{**dict(r), "attributes": json.loads(r["attributes"] or "{}")} for r in
                cn.execute("SELECT * FROM working_paper_samples WHERE wp_id=? ORDER BY position,id", (wid,))]
    if c:
        return q(c)
    with db.connect() as cn:
        return q(cn)


def _snapshot(c, wid, note, user):
    wp = _get(c, wid)
    wp["samples"] = samples_of(wid, c)
    c.execute("INSERT INTO working_paper_versions(wp_id,version,snapshot,note,created_by,created_at) VALUES(?,?,?,?,?,?)",
              (wid, wp["version"], json.dumps(wp), note, user["username"], db.now()))


def _editable(c, wid, user):
    db._need(user, "edit")
    wp = _get(c, wid)
    if wp["status"] in WP_LOCKED:
        raise PermissionError(f"Working paper is '{wp['status']}' and locked.")
    return wp


def save_working_paper(wid, data, user, samples=None):
    """Save a draft. Previous state is kept in version history; version increments on every save."""
    with db.connect() as c:
        wp = _editable(c, wid, user)
        new_ref = _t(data.get("wp_ref")) or wp["wp_ref"]
        if not _t(data.get("procedure", wp["procedure"])):
            raise ValueError("The procedure or test being performed is required.")
        _snapshot(c, wid, "Saved", user)
        vals = {f: _t(data[f]) if f in data else wp[f] for f in EDIT_FIELDS}
        vals["wp_ref"] = new_ref
        try:
            c.execute(f"UPDATE working_papers SET {','.join(f + '=?' for f in vals)},version=version+1,updated_at=? WHERE id=?",
                      list(vals.values()) + [db.now(), wid])
        except sqlite3.IntegrityError:
            raise ValueError(f"Working paper reference '{new_ref}' already exists for this engagement.") from None
        if samples is not None:
            _write_samples(c, wid, samples)
        db._log(c, user, "save_working_paper", new_ref)


def _write_samples(c, wid, samples):
    c.execute("DELETE FROM working_paper_samples WHERE wp_id=?", (wid,))
    for i, s in enumerate(samples):
        res = _t(s.get("result")) or "Not tested"
        if res not in SAMPLE_RESULTS:
            raise ValueError(f"Unknown sample result '{res}'.")
        if res.startswith("Exception") and not _t(s.get("exception_note")):
            raise ValueError(f"Sample '{_t(s.get('item_ref')) or i + 1}': describe the exception for investigation.")
        c.execute("INSERT INTO working_paper_samples(wp_id,item_ref,description,attributes,evidence_ref,result,exception_note,position) VALUES(?,?,?,?,?,?,?,?)",
                  (wid, _t(s.get("item_ref")), _t(s.get("description")), json.dumps(s.get("attributes") or {}),
                   _t(s.get("evidence_ref")), res, _t(s.get("exception_note")), i))


def submit_for_review(wid, user):
    with db.connect() as c:
        wp = _editable(c, wid, user)
        missing = [lab for f, lab in (("work_performed", "Work performed"), ("conclusion", "Conclusion")) if not wp[f]]
        if missing:
            raise ValueError("Complete before submitting: " + ", ".join(missing) + ".")
        _snapshot(c, wid, "Submitted for review", user)
        c.execute("UPDATE working_papers SET status='Awaiting Review',prepared_by=COALESCE(NULLIF(prepared_by,''),?),"
                  "prepared_date=COALESCE(NULLIF(prepared_date,''),?),version=version+1,updated_at=? WHERE id=?",
                  (user["username"], db.now()[:10], db.now(), wid))
        db._log(c, user, "submit_working_paper", wp["wp_ref"])


def review(wid, decision, comments, user):
    """Reviewer approves ('Reviewed') or returns a working paper. Reviewer details are recorded."""
    db._need(user, "approve")
    if decision not in ("Reviewed", "Returned"):
        raise ValueError("Decision must be Reviewed or Returned.")
    if decision == "Returned" and not _t(comments):
        raise ValueError("Add reviewer comments explaining what must be corrected.")
    with db.connect() as c:
        wp = _get(c, wid)
        if wp["status"] != "Awaiting Review":
            raise ValueError("Only working papers awaiting review can be reviewed.")
        _snapshot(c, wid, f"Review: {decision}", user)
        c.execute("UPDATE working_papers SET status=?,reviewer=?,review_date=?,reviewer_comments=?,version=version+1,updated_at=? WHERE id=?",
                  (decision, user["username"], db.now()[:10], _t(comments), db.now(), wid))
        db._log(c, user, "review_working_paper", f"{wp['wp_ref']}: {decision}")


def list_versions(wid):
    with db.connect() as c:
        return [dict(r) for r in c.execute("SELECT id,version,note,created_by,created_at FROM working_paper_versions WHERE wp_id=? ORDER BY id DESC", (wid,))]
