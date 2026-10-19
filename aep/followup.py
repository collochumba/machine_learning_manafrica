"""Module I: basic post-audit follow-up. Management-reported and auditor-verified statuses are separate fields.
No findings register is required: finding_ref is optional free text. Schema allows later issue tracking."""
import database as db

MGMT_STATUSES = ["Not Started", "In Progress", "Implemented", "Partially Implemented", "Not Implemented"]
AUDITOR_STATUSES = ["Unable to Verify", "In Progress", "Implemented", "Partially Implemented", "Not Implemented"]
REVIEW_STATUSES = ["Draft", "Awaiting Review", "Reviewed", "Returned"]
FIELDS = ["annual_engagement_id", "awp_id", "finding_ref", "original_recommendation", "agreed_action", "action_owner",
          "original_due_date", "mgmt_status", "mgmt_update_date", "mgmt_evidence", "auditor_procedures", "auditor_status",
          "followup_conclusion", "followup_date"]
_LOCKED = ("Awaiting Review", "Reviewed")


def _t(v):
    return "" if v is None else str(v).strip()


def _clean(data):
    d = {f: (data.get(f) if f in ("annual_engagement_id", "awp_id") else _t(data.get(f))) for f in FIELDS if f in data}
    if "original_recommendation" in d and not d["original_recommendation"]:
        raise ValueError("The original recommendation is required.")
    if d.get("mgmt_status", "Not Started") not in MGMT_STATUSES:
        raise ValueError("Unknown management-reported status.")
    if d.get("auditor_status", "Unable to Verify") not in AUDITOR_STATUSES:
        raise ValueError("Unknown auditor-verified status.")
    for k in ("original_due_date", "mgmt_update_date", "followup_date"):
        v = d.get(k)
        if v:
            try:
                __import__("datetime").date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"{k.replace('_', ' ').capitalize()} must be a date (YYYY-MM-DD).") from None
    return d


def create_followup(data, user):
    db._need(user, "edit")
    d = _clean({"mgmt_status": "Not Started", "auditor_status": "Unable to Verify", **data})
    if "original_recommendation" not in d:
        raise ValueError("The original recommendation is required.")
    cols = list(d) + ["created_by", "created_at", "updated_at"]
    with db.connect() as c:
        cur = c.execute(f"INSERT INTO followups({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                        list(d.values()) + [user["username"], db.now(), db.now()])
        db._log(c, user, "create_followup", d.get("finding_ref") or f"id {cur.lastrowid}")
        return cur.lastrowid


def update_followup(fid, data, user):
    db._need(user, "edit")
    with db.connect() as c:
        cur = c.execute("SELECT * FROM followups WHERE id=?", (fid,)).fetchone()
        if not cur:
            raise ValueError("Follow-up record not found.")
        if cur["review_status"] in _LOCKED:
            raise PermissionError(f"Follow-up is '{cur['review_status']}' and locked.")
        d = _clean({**dict(cur), **data})
        d = {k: v for k, v in d.items() if k in FIELDS}
        c.execute(f"UPDATE followups SET {','.join(k + '=?' for k in d)},updated_at=? WHERE id=?", list(d.values()) + [db.now(), fid])
        db._log(c, user, "update_followup", str(fid))


def submit(fid, user):
    db._need(user, "edit")
    with db.connect() as c:
        r = c.execute("SELECT * FROM followups WHERE id=?", (fid,)).fetchone()
        if not r:
            raise ValueError("Follow-up record not found.")
        if r["review_status"] in _LOCKED:
            raise PermissionError("Already submitted or reviewed.")
        if not r["followup_conclusion"] or not r["followup_date"]:
            raise ValueError("Enter the follow-up conclusion and date before submitting for review.")
        c.execute("UPDATE followups SET review_status='Awaiting Review',updated_at=? WHERE id=?", (db.now(), fid))
        db._log(c, user, "submit_followup", str(fid))


def review(fid, decision, comments, user):
    db._need(user, "approve")
    if decision not in ("Reviewed", "Returned"):
        raise ValueError("Decision must be Reviewed or Returned.")
    with db.connect() as c:
        r = c.execute("SELECT review_status FROM followups WHERE id=?", (fid,)).fetchone()
        if not r or r["review_status"] != "Awaiting Review":
            raise ValueError("Only follow-ups awaiting review can be reviewed.")
        c.execute("UPDATE followups SET review_status=?,reviewer=?,reviewer_comments=?,updated_at=? WHERE id=?",
                  (decision, user["username"], _t(comments), db.now(), fid))
        db._log(c, user, "review_followup", f"{fid}: {decision}")


def get_followup(fid):
    with db.connect() as c:
        r = c.execute("SELECT * FROM followups WHERE id=?", (fid,)).fetchone()
        return dict(r) if r else None


def list_followups(annual_id=None):
    with db.connect() as c:
        q, p = "SELECT * FROM followups", ()
        if annual_id:
            q, p = q + " WHERE annual_engagement_id=?", (annual_id,)
        return [dict(r) for r in c.execute(q + " ORDER BY id DESC", p)]
