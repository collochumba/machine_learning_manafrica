"""Module E: generate an AWP from an annual engagement + a reusable template.

The AWP is a COPY: procedures are written into the existing plan_rows/engagements tables and the template
version used is snapshotted in awp_source, so later template edits never change an existing AWP.
Generation always produces a Draft; approval stays with the existing reviewer workflow."""
import json

import awp_generator as gen
import database as db
import plan_register as pr
import template_lib as tl


def _criteria_text(criteria):
    if not criteria:
        return None
    out = []
    for c in criteria:
        tag = "VERIFIED" if c["verification"] == "Verified" else "UNVERIFIED - confirm against source"
        out.append(f"{c['text']} ({c['source'] or 'no source recorded'}"
                   f"{', ' + c['clause'] if c['clause'] else ', no clause recorded'}) [{tag}]")
    return "\n".join(out)


def build_draft(annual_id, template_id, version=None):
    """Return (engagement_dict, rows, template) WITHOUT saving, so the auditor can review and edit first."""
    ann = pr.get_engagement(annual_id)
    tpl = tl.get_template(template_id, version)
    if not ann or not tpl:
        raise ValueError("Select an existing engagement and template.")
    if tpl["status"] != "Active":
        raise ValueError("Only Active templates can be used to generate an AWP.")
    c = tpl["content"]
    areas = pr.get_engagement_areas(annual_id)
    method = [f"{lab}: {c[k]}" for lab, k in (("Risk assessment", "risk_assessment"),
                                              ("Sampling methodology", "sampling_methodology"),
                                              ("Resource requirements", "resource_requirements"),
                                              ("Planned timelines", "planned_timelines")) if c[k]]
    eng = {"ref": "", "title": f"AWP - {ann['title']}", "org": ann["department"] or "Kenya School of Government",
           "category": tpl["family"] or "General Operations", "area": (areas[0]["name"] if areas else tpl["family"]),
           "sub_area": tpl["name"], "period": ann["financial_year"], "start_date": ann["planned_start"] if pr._is_iso(ann["planned_start"]) else "",
           "end_date": ann["planned_end"] if pr._is_iso(ann["planned_end"]) else "",
           "objective": ann["objective"] or c["objective"], "scope": ann["scope"] or c["scope"],
           "policies": "; ".join(f"{x['text']} [{x['verification']}]" for x in c["criteria"]),
           "team": ann["team"] or c["prepared_by"], "reviewer": c["reviewer"],
           "risk_level": ann["risk_rating"] if ann["risk_rating"] in ("Low", "Medium", "High", "Critical") else "Not assessed",
           "instructions": "\n".join(method)}
    items = [{"area": p["title"] or p["procedure"][:60], "risk": p["risk"], "procedure": p["procedure"],
              "evidence_required": p["evidence_required"], "control_objective": p["control_objective"],
              "sampling_method": p["sampling"] or c["sampling_methodology"], "expected": p["expected"] or c["expected_results"]}
             for p in c["procedures"]]
    return eng, items, tpl


def _unique_ref(base):
    with db.connect() as cn:
        ref, n = base, 2
        while cn.execute("SELECT 1 FROM engagements WHERE ref=?", (ref,)).fetchone():
            ref, n = f"{base}-{n}", n + 1
        return ref


def save_awp(annual_id, template_id, user, eng_overrides=None, version=None, rows_override=None, items_override=None):
    """Create the AWP (status Draft) from the template version. Returns the new AWP id."""
    db._need(user, "edit")
    eng, items, tpl = build_draft(annual_id, template_id, version)
    items = items if items_override is None else items_override
    ann = pr.get_engagement(annual_id)
    eng.update(eng_overrides or {})
    eng["ref"] = eng.get("ref") or _unique_ref(f"AWP/{ann['ref']}")
    crit = _criteria_text(tpl["content"]["criteria"])
    awp_id = db.create_engagement(eng, user)           # existing validation, uniqueness and audit log
    awp = db.get_engagement(awp_id)
    rows = rows_override if rows_override is not None else gen.generate_rows(awp, items)
    if rows_override is None and crit:
        for r in rows:
            r["criteria"] = crit
    db.replace_rows(awp_id, rows, user)
    with db.connect() as c:
        c.execute("INSERT INTO awp_source(awp_id,annual_engagement_id,template_version_id,template_snapshot,created_by,created_at) VALUES(?,?,?,?,?,?)",
                  (awp_id, annual_id, tpl["version_id"], json.dumps({"name": tpl["name"], "version": tpl["version"],
                                                                     "content": tpl["content"]}), user["username"], db.now()))
        db._log(c, user, "generate_awp", f"{eng['ref']} from '{tpl['name']}' v{tpl['version']}")
    return awp_id


def source_of(awp_id):
    with db.connect() as c:
        r = c.execute("SELECT * FROM awp_source WHERE awp_id=?", (awp_id,)).fetchone()
        if not r:
            return None
        d = dict(r); d["template_snapshot"] = json.loads(d["template_snapshot"] or "{}")
        return d


def awps_for_engagement(annual_id):
    with db.connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT e.id,e.ref,e.title,e.status,e.version FROM engagements e JOIN awp_source s ON s.awp_id=e.id "
            "WHERE s.annual_engagement_id=? ORDER BY e.id", (annual_id,))]
