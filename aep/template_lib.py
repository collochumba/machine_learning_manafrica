"""Module D: reusable, versioned AWP templates. Every save creates an immutable version row."""
import json

import database as db

TEMPLATE_STATUSES = ["Active", "Archived"]
SOURCE_STATUSES = ["Auditor-defined", "Derived from source document", "Draft structure - source to be confirmed"]
VERIFICATION = ["Verified", "Unverified"]
PROC_FIELDS = ["ref", "title", "procedure", "evidence_required", "expected", "sampling", "risk", "risk_level",
               "control_objective", "source_ref"]
CONTENT_TEXT = ["description", "objective", "scope", "risk_assessment", "sampling_methodology", "expected_results",
                "resource_requirements", "planned_timelines", "prepared_by", "reviewer"]


def blank_content():
    return {**{k: "" for k in CONTENT_TEXT}, "control_objectives": [], "criteria": [], "procedures": []}


def normalise(content):
    """Coerce a content dict to the canonical shape; criteria default to Unverified (never assumed authoritative)."""
    c = blank_content()
    for k in CONTENT_TEXT:
        c[k] = str(content.get(k) or "").strip()
    c["control_objectives"] = [str(x).strip() for x in content.get("control_objectives", []) if str(x).strip()]
    for cr in content.get("criteria", []):
        if str(cr.get("text") or "").strip():
            v = cr.get("verification") if cr.get("verification") in VERIFICATION else "Unverified"
            c["criteria"].append({"text": str(cr["text"]).strip(), "source": str(cr.get("source") or "").strip(),
                                  "clause": str(cr.get("clause") or "").strip(), "verification": v,
                                  "kind": str(cr.get("kind") or "").strip()})
    for i, p in enumerate(content.get("procedures", []), 1):
        row = {f: str(p.get(f) or "").strip() for f in PROC_FIELDS}
        if not row["procedure"] and not row["title"]:
            continue
        row["ref"] = row["ref"] or f"P{i:02d}"
        c["procedures"].append(row)
    return c


def _name(n):
    n = (n or "").strip()
    if not n:
        raise ValueError("Template name is required.")
    return n


def create_template(name, content, user, family="", area_id=None, source_status="Auditor-defined", note="Initial version"):
    db._need(user, "edit")
    name = _name(name)
    if source_status not in SOURCE_STATUSES:
        raise ValueError("Unknown source status.")
    content = normalise(content)
    with db.connect() as c:
        if c.execute("SELECT 1 FROM awp_templates WHERE lower(name)=lower(?)", (name,)).fetchone():
            raise ValueError(f"A template named '{name}' already exists.")
        cur = c.execute("INSERT INTO awp_templates(name,family,area_id,source_status,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                        (name, family.strip(), area_id, source_status, user["username"], db.now(), db.now()))
        tid = cur.lastrowid
        c.execute("INSERT INTO awp_template_versions(template_id,version,content,change_note,created_by,created_at) VALUES(?,?,?,?,?,?)",
                  (tid, 1, json.dumps(content), note, user["username"], db.now()))
        db._log(c, user, "create_template", f"{name} v1")
        return tid


def update_template(tid, content, user, note="", name=None, family=None, area_id=-1, source_status=None):
    """Saving an edit never changes old versions: it appends version n+1 and makes it current."""
    db._need(user, "edit")
    content = normalise(content)
    with db.connect() as c:
        t = c.execute("SELECT * FROM awp_templates WHERE id=?", (tid,)).fetchone()
        if not t:
            raise ValueError("Template not found.")
        if t["status"] == "Archived":
            raise ValueError("Archived templates cannot be edited. Activate the template first.")
        new_name = _name(name) if name is not None else t["name"]
        if c.execute("SELECT 1 FROM awp_templates WHERE lower(name)=lower(?) AND id<>?", (new_name, tid)).fetchone():
            raise ValueError(f"A template named '{new_name}' already exists.")
        ver = t["current_version"] + 1
        c.execute("INSERT INTO awp_template_versions(template_id,version,content,change_note,created_by,created_at) VALUES(?,?,?,?,?,?)",
                  (tid, ver, json.dumps(content), note.strip(), user["username"], db.now()))
        c.execute("UPDATE awp_templates SET name=?,family=?,area_id=?,source_status=?,current_version=?,updated_at=? WHERE id=?",
                  (new_name, t["family"] if family is None else family.strip(), t["area_id"] if area_id == -1 else area_id,
                   t["source_status"] if source_status is None else source_status, ver, db.now(), tid))
        db._log(c, user, "update_template", f"{new_name} v{ver}")
        return ver


def duplicate_template(tid, new_name, user):
    db._need(user, "edit")
    t = get_template(tid)
    if not t:
        raise ValueError("Template not found.")
    return create_template(new_name, t["content"], user, family=t["family"], area_id=t["area_id"],
                           source_status=t["source_status"], note=f"Duplicated from '{t['name']}' v{t['current_version']}")


def set_status(tid, status, user):
    db._need(user, "edit")
    if status not in TEMPLATE_STATUSES:
        raise ValueError("Unknown template status.")
    with db.connect() as c:
        if not c.execute("SELECT 1 FROM awp_templates WHERE id=?", (tid,)).fetchone():
            raise ValueError("Template not found.")
        c.execute("UPDATE awp_templates SET status=?,updated_at=? WHERE id=?", (status, db.now(), tid))
        db._log(c, user, "template_status", f"{tid}: {status}")


def get_template(tid, version=None):
    with db.connect() as c:
        t = c.execute("SELECT * FROM awp_templates WHERE id=?", (tid,)).fetchone()
        if not t:
            return None
        v = c.execute("SELECT * FROM awp_template_versions WHERE template_id=? AND version=?",
                      (tid, version or t["current_version"])).fetchone()
        if not v:
            raise ValueError("Template version not found.")
        return {**dict(t), "version": v["version"], "version_id": v["id"], "content": json.loads(v["content"]),
                "change_note": v["change_note"], "version_by": v["created_by"], "version_at": v["created_at"]}


def list_templates(status="Active", family=None, area_id=None, search=""):
    q, p = "SELECT * FROM awp_templates WHERE 1=1", []
    if status:
        q += " AND status=?"; p.append(status)
    if family:
        q += " AND family=?"; p.append(family)
    if area_id:
        q += " AND area_id=?"; p.append(area_id)
    if search:
        q += " AND (name LIKE ? OR family LIKE ?)"; p += [f"%{search}%"] * 2
    with db.connect() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY family, name", p)]


def list_versions(tid):
    with db.connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT id,version,change_note,created_by,created_at FROM awp_template_versions WHERE template_id=? ORDER BY version DESC", (tid,))]
