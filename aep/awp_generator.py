"""Draft working-plan generation. Missing information is flagged, never invented."""
from database import ROW_FIELDS, now

MISSING = "[TO BE COMPLETED BY AUDITOR]"
LABELS = {"wp_ref": "WP Ref", "area": "Audit Area", "objective": "Audit Objective", "risk": "Risk Identified",
          "risk_level": "Risk Level", "control_objective": "Control Objective", "criteria": "Audit Criteria",
          "procedure": "Audit Procedure / Test", "evidence_required": "Evidence Required",
          "sampling_method": "Sampling Method", "sample_size": "Sample Size / Population",
          "auditor": "Responsible Auditor", "expected": "Expected Results", "actual": "Actual Results",
          "evidence_ref": "Evidence Reference", "conclusion": "Auditor's Conclusion",
          "review_comments": "Review Comments", "status": "Status"}


def _criteria(eng, area, library):
    parts = []
    for it in library:
        if it["kind"] == "Audit criterion" and area.lower() in it["title"].lower():
            tag = "VERIFIED" if it["verified"] else "UNVERIFIED - confirm against source"
            parts.append(f"{it['title']} ({it['doc_title'] or 'no document'}, {it['clause'] or 'no clause'}) [{tag}]")
    if not parts:
        parts.append(f"[VERIFY: identify applicable criteria for '{area}'"
                     + (f"; engagement lists (unverified): {eng['policies']}" if eng.get("policies") else "") + "]")
    return "\n".join(parts)


def generate_rows(eng, items, library=()):
    rows = []
    for i, it in enumerate(items, 1):
        a = it["area"]
        rows.append({
            "wp_ref": f"{eng['ref']}/WP{i:02d}", "area": a, "objective": eng.get("objective") or MISSING,
            "risk": it.get("risk") or MISSING, "risk_level": eng.get("risk_level") or "Not assessed",
            "control_objective": it.get("control_objective") or f"[VERIFY: control objective for {a}]",
            "criteria": _criteria(eng, a, library), "procedure": it.get("procedure") or MISSING,
            "evidence_required": it.get("evidence_required") or MISSING,
            "sampling_method": it.get("sampling_method") or "[AUDITOR TO DETERMINE]", "sample_size": MISSING,
            "auditor": eng.get("team") or MISSING, "expected": it.get("expected") or MISSING,
            "actual": "", "evidence_ref": "", "conclusion": "", "review_comments": "", "status": "Not Started"})
    return rows


def rows_from_saved(eng, saved_rows):
    out = []
    for i, r in enumerate(saved_rows, 1):
        r = {**r, "wp_ref": f"{eng['ref']}/WP{i:02d}", "actual": "", "evidence_ref": "", "conclusion": "",
             "review_comments": "", "status": "Not Started", "extra": {}}
        out.append(r)
    return out


def count_missing(rows):
    marks = ("[TO BE COMPLETED", "[VERIFY", "[UNVERIFIED", "[AUDITOR TO")
    return sum(any(m in str(r.get(f, "")) for m in marks) for r in rows for f in ROW_FIELDS)


def seed_demo(user):
    import database as db
    from audit_templates import TEMPLATES
    eid = db.create_engagement({
        "title": "DEMO (FICTIONAL) - ICT Review", "org": "Demo Department (fictional)", "category": "ICT",
        "area": "ICT general controls", "period": "FY 2099/2100", "objective": "Demonstration objective (fictional).",
        "scope": "Fictional scope for training only.", "team": user["username"], "risk_level": "Medium"}, user, is_demo=True)
    eng = db.get_engagement(eid)
    db.replace_rows(eid, generate_rows(eng, TEMPLATES["ICT"][:4]), user)
    return eid
