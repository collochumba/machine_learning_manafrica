"""Streamlit pages for the new modules (A-K). Business logic lives in the data modules; this file is UI only."""
import pandas as pd
import streamlit as st

import awp_builder as ab
import database as dbm
import export_new as ex
import followup as fu
import plan_register as pr
import seed_templates as seeds
import template_lib as tl
import theme as th
import transport_analysis as ta
import working_papers as wpm
from awp_generator import count_missing
from export_excel import build_workbook
from export_word import build_document

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def run(fn, ok=None):
    try:
        r = fn()
        if ok: st.success(ok)
        return True if r is None else r
    except (ValueError, PermissionError) as e:
        st.error(str(e)); return False


def header(t, sub=None):
    st.markdown(th.page_header(t, sub), unsafe_allow_html=True)


def _can(user): return dbm.can(user["role"], "edit")
def _label(e): return f"{e['financial_year']} | {e['ref']} - {e['title']}" + ("  [TEST DATA]" if e.get("is_test_data") else "")


def pick_engagement(key, fy_filter=True):
    engs = pr.list_engagements()
    if not engs:
        st.info("No engagements yet. Add one or import the Annual Audit Plan under 'Annual Audit Plan'."); return None
    fys = pr.financial_years()
    fy = st.selectbox("Financial year", fys, key=key + "fy") if fy_filter else None
    pool = [e for e in engs if not fy or e["financial_year"] == fy]
    return st.selectbox("Engagement", pool, format_func=_label, key=key + "e") if pool else None


# =========================== A. Annual Audit Plan ===========================
def page_annual_plan(user):
    header("Annual Audit Plan", "Financial-year engagement register (separate from the audit-area catalogue and from individual AWPs)")
    t = st.tabs(["Register", "Add / edit", "Import from Excel", "Export"])
    engs = pr.list_engagements()
    with t[0]:
        if not engs: st.info("The register is empty. Use 'Import from Excel' to load the approved Annual Audit Plan, or add engagements manually.")
        else:
            fy = st.multiselect("Financial year", pr.financial_years())
            df = pd.DataFrame([e for e in engs if not fy or e["financial_year"] in fy])
            st.dataframe(df[["id", "financial_year", "ref", "title", "risk_rating", "audit_type", "planned_start", "planned_end", "department", "status", "is_test_data"]]
                         .rename(columns={"is_test_data": "TEST data"}), hide_index=True, use_container_width=True)
            if any(e["is_test_data"] for e in engs): st.warning("Rows marked TEST data are demonstration records, not official KSG engagements.")
            e = st.selectbox("Linked records for", engs, format_func=_label, key="lr")
            st.write(f"Areas: {', '.join(a['name'] for a in pr.get_engagement_areas(e['id'])) or 'none'}  |  Linked: {pr.linked_records(e['id'])}")
            if user["role"] == "Administrator":
                with st.expander("Delete engagement (administrator)"):
                    ok = st.checkbox("I understand this permanently deletes the engagement record", key="delok")
                    if st.button("Delete engagement", disabled=not ok) and run(lambda: pr.delete_engagement(e["id"], user), "Engagement deleted."): st.rerun()
    with t[1]:
        if not _can(user): st.info("Your role cannot edit the register."); 
        else:
            cur = st.selectbox("Edit existing (or leave as 'New')", [None] + engs, format_func=lambda x: "New engagement" if x is None else _label(x))
            g = lambda k: (cur or {}).get(k, "") or ""
            with st.form("eng_form"):
                c1, c2 = st.columns(2)
                d = {"financial_year": c1.text_input("Financial year * (e.g. 2026/2027)", g("financial_year")), "ref": c2.text_input("Engagement reference *", g("ref")),
                     "title": st.text_input("Engagement title *", g("title")), "objective": st.text_area("Audit objective", g("objective")),
                     "scope": st.text_area("Scope", g("scope")), "risk_rating": c1.text_input("Risk rating (as per approved source; leave blank if none)", g("risk_rating")),
                     "audit_type": c2.text_input("Nature / type of audit", g("audit_type")), "planned_start": c1.text_input("Planned start (YYYY-MM-DD or period)", g("planned_start")),
                     "planned_end": c2.text_input("Planned end (YYYY-MM-DD or period)", g("planned_end")), "department": c1.text_input("Responsible department / auditee", g("department")),
                     "team": c2.text_input("Assigned audit team", g("team"))}
                if cur: d["status"] = st.selectbox("Status", pr.ENG_STATUSES, index=pr.ENG_STATUSES.index(cur["status"]) if cur["status"] in pr.ENG_STATUSES else 0)
                areas = pr.list_areas()
                chosen = st.multiselect("Linked audit areas", [a["id"] for a in areas], format_func=lambda i: next(a["name"] for a in areas if a["id"] == i),
                                        default=[a["id"] for a in pr.get_engagement_areas(cur["id"])] if cur else [])
                if st.form_submit_button("Save engagement", type="primary"):
                    def save():
                        eid = pr.update_engagement(cur["id"], d, user) or cur["id"] if cur else pr.create_engagement(d, user)
                        pr.set_engagement_areas(cur["id"] if cur else eid, chosen, user)
                    if run(save, "Engagement saved."): st.rerun()
    with t[2]: _import_tab(user)
    with t[3]:
        if engs: st.download_button("Download register (Excel)", pr_xlsx(engs), "annual-audit-plan-register.xlsx", XLSX)
        else: st.info("Nothing to export yet.")


def pr_xlsx(engs): return ex.register_xlsx(engs)


def _import_tab(user):
    st.markdown("Upload an Excel workbook (.xlsx, max 5 MB). Nothing is saved until you review the preview and confirm. "
                "**Existing engagements are never overwritten.**")
    st.download_button("Download a clearly labelled TEST workbook", pr.make_test_workbook(), "TEST-annual-plan-import.xlsx", XLSX)
    if not _can(user): st.info("Your role cannot import."); return
    up = st.file_uploader("Annual Audit Plan workbook", type=["xlsx"], key="imp")
    if not up: return
    try:
        data, names = up.getvalue(), pr.sheet_names(up.name, up.getvalue())
        c1, c2, c3 = st.columns(3)
        sheet = c1.selectbox("Sheet", names); hdr = int(c2.number_input("Header row", 1, 50, 1)); fy = c3.text_input("Financial year *", "")
        headers, rows = pr.read_sheet(up.name, data, sheet, hdr)
    except ValueError as e:
        st.error(str(e)); return
    sug, opts = pr.suggest_mapping(headers), [None] + list(range(len(headers)))
    st.markdown("##### Field mapping (change if the suggestion is wrong)")
    mapping, cols = {}, st.columns(3)
    for i, f in enumerate(pr.FIELDS):
        mapping[f] = cols[i % 3].selectbox(pr.FIELD_LABELS[f] + (" *" if f in pr.REQUIRED else ""), opts, index=opts.index(sug.get(f)),
                                           format_func=lambda x: "(not mapped)" if x is None else headers[x] or f"Column {x + 1}", key=f"map{f}")
    is_test = sheet == pr.TEST_SHEET or st.checkbox("Mark imported rows as TEST data", sheet == pr.TEST_SHEET)
    prev = pr.build_preview(headers, rows, mapping, fy, up.name, sheet, is_test)
    df = pd.DataFrame(prev)
    st.dataframe(df[["row", "status", "issues", "ref", "title", "department", "planned_start", "planned_end"]], hide_index=True, use_container_width=True)
    cnt = df.status.value_counts().to_dict(); st.write("Preview: " + ", ".join(f"{k}: {v}" for k, v in cnt.items()))
    inc = st.checkbox("Also import 'Incomplete' rows (missing recommended fields)", False)
    ok_rows = [p["row"] for p in prev if p["status"] == "New" or (inc and p["status"] == "Incomplete")]
    confirm = st.checkbox(f"I have reviewed the preview and confirm importing {len(ok_rows)} row(s)")
    if st.button("Import confirmed rows", type="primary", disabled=not (confirm and ok_rows)):
        r = run(lambda: pr.commit_import(prev, ok_rows, user))
        if r: st.success(f"Imported {r['added']} engagement(s); skipped {len(r['skipped'])}."); 


# =========================== B. Audit areas catalogue / C. library links ===========================
def areas_catalogue_ui(user):
    st.subheader("Audit area catalogue (reusable)")
    st.caption("Candidate categories - not proof that each is an approved engagement. Reconcile with the approved Annual Audit Plan.")
    areas = pr.list_areas(include_archived=True)
    st.dataframe(pd.DataFrame(areas)[["id", "name", "description", "status", "is_candidate"]], hide_index=True, use_container_width=True)
    if user["role"] != "Administrator": st.info("Only administrators manage the catalogue."); return
    cur = st.selectbox("Edit area (or New)", [None] + areas, format_func=lambda a: "New area" if a is None else a["name"], key="areasel")
    with st.form("catalogue"):
        n, ds = st.text_input("Name *", (cur or {}).get("name", "")), st.text_area("Description", (cur or {}).get("description", ""))
        stt = st.selectbox("Status", ["Active", "Archived"], index=0 if not cur or cur["status"] == "Active" else 1)
        cand = st.checkbox("Candidate category (not yet reconciled to the approved plan)", bool((cur or {}).get("is_candidate", 1)))
        if st.form_submit_button("Save area", type="primary"):
            if run(lambda: pr.save_area({"name": n, "description": ds, "status": stt, "is_candidate": cand}, user, (cur or {}).get("id")), "Area saved."): st.rerun()


def library_links_ui(user):
    st.subheader("Verification and audit-area links")
    items = dbm.list_lib()
    if not items: st.info("Add library items first."); return
    it = st.selectbox("Library item", items, format_func=lambda x: f"#{x['id']} {x['kind']}: {x['title']}", key="lnk")
    meta, areas = pr.get_library_meta(it["id"]), pr.list_areas()
    cur = [a for a in pr.library_links(it["id"]) if a in [x["id"] for x in areas]]
    if meta["verification_status"] == "Unverified": st.warning("UNVERIFIED - a suggested procedure/criterion is not a legal or KSG policy requirement until verified against its source.")
    with st.form(f"meta{it['id']}"):
        stt = st.selectbox("Verification status", pr.VERIFY_STATES, index=pr.VERIFY_STATES.index(meta["verification_status"]))
        exp, nt = st.text_area("Expected test result", meta["expected_result"]), st.text_area("Notes", meta["notes"])
        sel = st.multiselect("Linked audit areas", [a["id"] for a in areas], default=cur, format_func=lambda i: next(a["name"] for a in areas if a["id"] == i))
        if st.form_submit_button("Save", type="primary", disabled=not _can(user)):
            def save(): pr.set_library_meta(it["id"], stt, nt, exp, user); pr.set_library_links(it["id"], sel, user)
            run(save, "Saved.")


# =========================== D. AWP templates ===========================
def page_awp_templates(user):
    header("AWP Templates", "Reusable, versioned templates. Editing creates a new version; existing AWPs are never changed.")
    t = st.tabs(["Library", "Create / edit", "Version history"])
    areas = {a["id"]: a["name"] for a in pr.list_areas()}
    with t[0]:
        c1, c2, c3 = st.columns(3)
        stt = c1.selectbox("Status", ["Active", "Archived", None], format_func=lambda x: x or "All")
        fam, q = c2.text_input("Family (e.g. Transport, ICT)"), c3.text_input("Search")
        lst = tl.list_templates(stt, fam or None, None, q)
        if not lst: st.info("No templates yet. Create one, or ask an administrator to load the starter structures in Settings."); 
        else:
            st.dataframe(pd.DataFrame(lst)[["id", "name", "family", "status", "current_version", "source_status"]], hide_index=True, use_container_width=True)
            sel = st.selectbox("Template", lst, format_func=lambda x: f"{x['name']} (v{x['current_version']})", key="tplsel")
            full = tl.get_template(sel["id"])
            if full["source_status"] != "Auditor-defined": st.warning(f"Source status: {full['source_status']}. Criteria and procedures need auditor verification.")
            with st.expander("Contents"):
                st.json(full["content"])
            if _can(user):
                a, b, c = st.columns(3)
                nm = a.text_input("Name for duplicate", f"{full['name']} (copy)")
                if b.button("Duplicate") and run(lambda: tl.duplicate_template(sel["id"], nm, user), "Duplicated."): st.rerun()
                tgt = "Archived" if full["status"] == "Active" else "Active"
                if c.button(f"Set {tgt}") and run(lambda: tl.set_status(sel["id"], tgt, user), f"Template is now {tgt}."): st.rerun()
    with t[1]: _template_editor(user, areas)
    with t[2]:
        lst = tl.list_templates(None)
        if lst:
            s = st.selectbox("Template", lst, format_func=lambda x: x["name"], key="vh")
            st.dataframe(pd.DataFrame(tl.list_versions(s["id"])), hide_index=True, use_container_width=True)


def _template_editor(user, areas):
    if not _can(user): st.info("Your role cannot edit templates."); return
    lst = tl.list_templates("Active")
    cur = st.selectbox("Edit an existing template (or New)", [None] + lst, format_func=lambda x: "New template" if x is None else f"{x['name']} (v{x['current_version']})", key="edsel")
    full = tl.get_template(cur["id"]) if cur else None
    c = full["content"] if full else tl.blank_content()
    with st.form(f"tpled{cur['id'] if cur else 'new'}"):
        c1, c2 = st.columns(2)
        name, fam = c1.text_input("Template name *", full["name"] if full else ""), c2.text_input("Family", full["family"] if full else "")
        aid = c1.selectbox("Audit area", [None] + list(areas), format_func=lambda x: "(none)" if x is None else areas[x],
                           index=([None] + list(areas)).index(full["area_id"]) if full and full["area_id"] in areas else 0)
        src = c2.selectbox("Source status", tl.SOURCE_STATUSES, index=tl.SOURCE_STATUSES.index(full["source_status"]) if full else 0)
        v = {k: st.text_area(k.replace("_", " ").capitalize(), c[k]) for k in tl.CONTENT_TEXT if k not in ("prepared_by", "reviewer")}
        v["prepared_by"], v["reviewer"] = c1.text_input("Prepared by", c["prepared_by"]), c2.text_input("Reviewer", c["reviewer"])
        co = st.text_area("Control objectives (one per line)", "\n".join(c["control_objectives"]))
        st.markdown("**Audit criteria and sources** (unverified unless a source document and clause are confirmed)")
        crit = st.data_editor(pd.DataFrame(c["criteria"] or [{"text": "", "source": "", "clause": "", "verification": "Unverified", "kind": ""}],
                                           columns=["text", "source", "clause", "verification", "kind"]), num_rows="dynamic", key=f"crit{cur['id'] if cur else 'n'}",
                              column_config={"verification": st.column_config.SelectboxColumn(options=tl.VERIFICATION)})
        st.markdown("**Audit procedures**")
        procs = st.data_editor(pd.DataFrame(c["procedures"] or [{}], columns=tl.PROC_FIELDS), num_rows="dynamic", key=f"proc{cur['id'] if cur else 'n'}")
        note = st.text_input("Change note (saved with the new version)")
        if st.form_submit_button("Save template (creates a new version)", type="primary"):
            content = {**v, "control_objectives": [x for x in co.splitlines() if x.strip()],
                       "criteria": crit.fillna("").to_dict("records"), "procedures": procs.fillna("").to_dict("records")}
            fn = (lambda: tl.update_template(cur["id"], content, user, note, name, fam, aid, src)) if cur else (lambda: tl.create_template(name, content, user, fam, aid, src))
            if run(fn, "Template saved."): st.rerun()


# =========================== E. Generate AWP ===========================
def page_generate_awp(user):
    header("Generate AWP", "Engagement -> template -> review -> edit -> save -> submit -> export")
    if not _can(user): st.warning("Your role cannot generate AWPs."); return
    eng = pick_engagement("gen")
    if not eng: return
    areas = pr.list_areas(); linked = pr.get_engagement_areas(eng["id"])
    st.caption("Linked audit areas: " + (", ".join(a["name"] for a in linked) or "none (link areas under Annual Audit Plan > Add / edit)"))
    tpls = [t for t in tl.list_templates("Active") if not linked or not t["area_id"] or t["area_id"] in [a["id"] for a in linked]]
    other = st.checkbox("Show templates for all areas", not tpls)
    if other: tpls = tl.list_templates("Active")
    if not tpls: st.info("No active templates available. Create one under AWP Templates."); return
    tpl = st.selectbox("Template", tpls, format_func=lambda x: f"{x['name']} (v{x['current_version']})")
    draft_eng, items, full = ab.build_draft(eng["id"], tpl["id"])
    if full["source_status"] != "Auditor-defined": st.warning(f"Template source status: {full['source_status']}.")
    st.markdown("##### 1. Engagement information (editable before saving)")
    c1, c2 = st.columns(2)
    ov = {k: (c1 if i % 2 == 0 else c2).text_input(k.replace("_", " ").capitalize(), draft_eng[k], key=f"g{k}")
          for i, k in enumerate(["title", "org", "period", "team", "reviewer", "risk_level"])}
    ov["objective"], ov["scope"] = st.text_area("Objective", draft_eng["objective"], key="gobj"), st.text_area("Scope", draft_eng["scope"], key="gscope")
    ov["instructions"] = st.text_area("Risk assessment, sampling, resources, timelines", draft_eng["instructions"], key="gins")
    st.markdown("##### 2. Procedures, criteria and evidence")
    cols = ["area", "risk", "control_objective", "procedure", "evidence_required", "sampling_method", "expected"]
    ed = st.data_editor(pd.DataFrame(items, columns=cols), num_rows="dynamic", key=f"gitems{tpl['id']}{eng['id']}")
    crit = ab._criteria_text(full["content"]["criteria"])
    if crit: st.text_area("Criteria carried into the AWP (UNVERIFIED unless confirmed)", crit, disabled=True)
    if st.button("Save AWP as Draft", type="primary"):
        r = run(lambda: ab.save_awp(eng["id"], tpl["id"], user, ov, items_override=ed.fillna("").to_dict("records")))
        if r: st.session_state["gen_awp"] = r; st.success("AWP saved as Draft. It is NOT approved.")
    awp_id = st.session_state.get("gen_awp")
    if awp_id and dbm.get_engagement(awp_id):
        a = dbm.get_engagement(awp_id); rows = dbm.get_rows(awp_id)
        st.markdown(f"**{a['ref']}** - status {th.badge(a['status'])}", unsafe_allow_html=True)
        n = count_missing(rows)
        if n: st.warning(f"{n} cell(s) still contain placeholders to complete or verify.")
        b1, b2, b3 = st.columns(3)
        if a["status"] in ("Draft", "In Progress", "Returned") and b1.button("Submit for review") and run(lambda: dbm.set_status(awp_id, "Awaiting Review", user), "Submitted. A reviewer must approve it."): st.rerun()
        safe = ex.safe_name(a["ref"], "AWP", ext="{}")
        b2.download_button("Excel", build_workbook(a, rows, dbm.get_comments(awp_id)), safe.format("xlsx"), XLSX)
        b3.download_button("Word", build_document(a, rows, dbm.get_comments(awp_id)), safe.format("docx"), DOCX)
        st.caption("Further edits and the review/approval steps are on 'My Working Plans'.")


# =========================== F. Working papers (+ G transport analysis) ===========================
def page_working_papers(user):
    header("Working Papers", "Individual working papers linked to an engagement and, where applicable, an approved AWP procedure")
    t = st.tabs(["Working papers", "Create", "Transport analysis tools"])
    with t[0]: _wp_list(user)
    with t[1]: _wp_create(user)
    with t[2]: _transport_tools()


def _wp_create(user):
    if not _can(user): st.info("Your role cannot create working papers."); return
    eng = pick_engagement("wpc")
    if not eng: return
    awps = [a for a in ab.awps_for_engagement(eng["id"]) if a["status"] == "Approved"]
    st.markdown("##### From an approved AWP")
    if awps:
        a = st.selectbox("Approved AWP", awps, format_func=lambda x: f"{x['ref']} v{x['version']}")
        rows = dbm.get_rows(a["id"])
        pick = st.multiselect("Procedures", [r["wp_ref"] for r in rows], default=[r["wp_ref"] for r in rows], format_func=lambda x: f"{x} - {next(r['area'] for r in rows if r['wp_ref'] == x)}")
        if st.button("Generate working papers", type="primary"):
            r = run(lambda: wpm.create_from_awp(a["id"], user, pick))
            if r is not False: st.success(f"Created {len(r)} working paper(s) (planning fields only; results left blank)." if r else "Nothing new to create.")
    else: st.info("No Approved AWP is linked to this engagement yet. Approve an AWP first, or create a manual working paper below.")
    st.markdown("##### Manual working paper")
    with st.form("wpmanual"):
        ref = st.text_input("Working paper reference (blank = auto)"); proc = st.text_area("Procedure / test *")
        if st.form_submit_button("Create"): run(lambda: wpm.create_working_paper(eng["id"], user, {"wp_ref": ref, "procedure": proc}), "Working paper created.")


def _wp_list(user):
    eng = pick_engagement("wpl")
    if not eng: return
    lst = wpm.list_working_papers(eng["id"])
    if not lst: st.info("No working papers for this engagement yet."); return
    st.dataframe(pd.DataFrame(lst)[["id", "wp_ref", "area", "procedure", "status", "version", "prepared_by"]], hide_index=True, use_container_width=True)
    sel = st.selectbox("Open working paper", lst, format_func=lambda x: f"{x['wp_ref']} [{x['status']}]")
    w = wpm.get_working_paper(sel["id"])
    editable = _can(user) and w["status"] not in wpm.WP_LOCKED
    st.markdown(f"{th.badge(w['status'])} &nbsp; v{w['version']}", unsafe_allow_html=True)
    if not editable: st.caption("Read-only: submitted/reviewed, or your role cannot edit.")
    k = f"wp{w['id']}v{w['version']}"
    with st.form(k):
        d = {"wp_ref": st.text_input("Working paper reference", w["wp_ref"], disabled=not editable)}
        for f, lab in (("area", "Audit area"), ("audit_objective", "Audit objective"), ("control_objective", "Control objective"), ("risk", "Risk"), ("criteria", "Audit criteria (unverified unless confirmed)"),
                       ("procedure", "Procedure / test being performed"), ("population", "Population"), ("sample_selection", "Sample selection"), ("documents_required", "Documents and evidence required"),
                       ("work_performed", "Work performed (auditor to enter)"), ("evidence_refs", "Evidence references"), ("test_results", "Test results"),
                       ("exceptions", "Exceptions / deviations (for investigation, not yet findings)"), ("analysis", "Auditor analysis"), ("conclusion", "Conclusion")):
            d[f] = st.text_area(lab, w[f], disabled=not editable)
        c1, c2 = st.columns(2)
        d["prepared_by"], d["prepared_date"] = c1.text_input("Prepared by", w["prepared_by"], disabled=not editable), c2.text_input("Prepared date (YYYY-MM-DD)", w["prepared_date"], disabled=not editable)
        st.markdown("**Sample-level testing**")
        sdf = pd.DataFrame(w["samples"], columns=["item_ref", "description", "evidence_ref", "result", "exception_note"])
        samp = st.data_editor(sdf, num_rows="dynamic", disabled=not editable, key=k + "s",
                              column_config={"result": st.column_config.SelectboxColumn(options=wpm.SAMPLE_RESULTS)})
        if st.form_submit_button("Save draft", type="primary", disabled=not editable):
            if run(lambda: wpm.save_working_paper(w["id"], d, user, samp.fillna("").to_dict("records")), "Saved."): st.rerun()
    c1, c2, c3 = st.columns(3)
    if w["status"] in ("Draft", "Returned") and c1.button("Submit for review", disabled=not _can(user)) and run(lambda: wpm.submit_for_review(w["id"], user), "Submitted for review."): st.rerun()
    if w["status"] == "Awaiting Review" and dbm.can(user["role"], "approve"):
        cm = st.text_area("Reviewer comments", key=k + "rc")
        if c1.button("Mark reviewed") and run(lambda: wpm.review(w["id"], "Reviewed", cm, user), "Reviewed."): st.rerun()
        if c2.button("Return to auditor") and run(lambda: wpm.review(w["id"], "Returned", cm, user), "Returned."): st.rerun()
    ann = pr.get_engagement(w["annual_engagement_id"]); base = (ann["ref"], w["wp_ref"], "working-paper")
    c3.download_button("Word", ex.working_paper_docx(w, ann), ex.safe_name(*base, ext="docx"), DOCX)
    c3.download_button("Excel", ex.working_paper_xlsx(w, ann), ex.safe_name(*base, ext="xlsx"), XLSX)
    with st.expander("Version history"): st.dataframe(pd.DataFrame(wpm.list_versions(w["id"])), hide_index=True)


def _transport_tools():
    st.caption("Results flag items for INVESTIGATION. They are not audit findings until management's explanation and evidence have been evaluated.")
    a, b, c = st.tabs(["Fuel efficiency", "Fleet utilisation", "Maintenance cost"])
    with a:
        c1, c2 = st.columns(2)
        exp, tol = c1.number_input("Expected km/l (0 = none)", 0.0, 100.0, 0.0), c2.number_input("Tolerance %", 0.0, 100.0, 20.0)
        df = st.data_editor(pd.DataFrame({"vehicle": [""], "km": [None], "litres": [None]}), num_rows="dynamic", key="fuel")
        if len(df): st.dataframe(pd.DataFrame(ta.analyse_fuel(df.to_dict("records"), exp or None, tol)), hide_index=True, use_container_width=True)
    with b:
        x, y = st.number_input("Vehicles actively utilised", 0, 100000, 0), st.number_input("Total serviceable vehicles", 0, 100000, 0)
        r = ta.fleet_utilisation(x, y); st.metric("Utilisation %", "-" if r["value"] is None else r["value"]); st.caption(f"{r['status']} {r['note']}")
    with c:
        s, e = st.text_input("Review period start (YYYY-MM-DD) *"), st.text_input("Review period end (YYYY-MM-DD) *")
        df = st.data_editor(pd.DataFrame({"vehicle": [""], "date": [""], "amount": [None]}), num_rows="dynamic", key="maint")
        if s and e:
            try:
                r = ta.maintenance_cost(df.to_dict("records"), s, e)
                st.write(f"Period: {r['period']} | Total: {r['total']:,.2f}"); st.dataframe(pd.DataFrame(r["per_vehicle"].items(), columns=["vehicle", "total"]), hide_index=True)
                if r["rejected"]: st.warning(f"{len(r['rejected'])} row(s) rejected for missing/invalid data.")
            except ValueError as ex_: st.error(str(ex_))
    st.download_button("Download the editable Excel analysis tool (formulas)", ta.build_analysis_workbook(), "transport-analysis-tool.xlsx", XLSX)


# =========================== I. Post-audit follow-up ===========================
def page_followup(user):
    header("Post-Audit Follow-Up", "Management-reported status and auditor-verified status are recorded separately")
    t = st.tabs(["Records", "Add"])
    anns = {e["id"]: e for e in pr.list_engagements()}
    with t[1]:
        if not _can(user): st.info("Your role cannot add records.")
        else:
            with st.form("fuadd"):
                e = st.selectbox("Original engagement (optional)", [None] + list(anns.values()), format_func=lambda x: "(not linked)" if x is None else _label(x))
                fr = st.text_input("Original finding / recommendation reference (optional - not required)")
                rec, act = st.text_area("Original recommendation *"), st.text_area("Agreed management action")
                c1, c2 = st.columns(2)
                own, due = c1.text_input("Responsible department / action owner"), c2.text_input("Original agreed implementation date (YYYY-MM-DD)")
                if st.form_submit_button("Create follow-up record", type="primary"):
                    run(lambda: fu.create_followup({"annual_engagement_id": e["id"] if e else None, "finding_ref": fr, "original_recommendation": rec,
                                                    "agreed_action": act, "action_owner": own, "original_due_date": due}, user), "Record created.")
    with t[0]:
        lst = fu.list_followups()
        if not lst: st.info("No follow-up records yet."); return
        st.dataframe(pd.DataFrame(lst)[["id", "finding_ref", "original_recommendation", "mgmt_status", "auditor_status", "review_status"]], hide_index=True, use_container_width=True)
        f = st.selectbox("Open record", lst, format_func=lambda x: f"#{x['id']} {x['finding_ref'] or x['original_recommendation'][:50]} [{x['review_status']}]")
        locked = f["review_status"] in ("Awaiting Review", "Reviewed") or not _can(user)
        with st.form(f"fu{f['id']}"):
            d = {k: st.text_area(lab, f[k], disabled=locked) for k, lab in (("agreed_action", "Agreed management action"), ("mgmt_evidence", "Supporting evidence supplied by management"),
                                                                           ("auditor_procedures", "Auditor verification procedures and comments"), ("followup_conclusion", "Follow-up conclusion"))}
            c1, c2 = st.columns(2)
            d["mgmt_status"] = c1.selectbox("MANAGEMENT-REPORTED status", fu.MGMT_STATUSES, index=fu.MGMT_STATUSES.index(f["mgmt_status"]), disabled=locked)
            d["mgmt_update_date"] = c1.text_input("Date of management update (YYYY-MM-DD)", f["mgmt_update_date"], disabled=locked)
            d["auditor_status"] = c2.selectbox("AUDITOR-VERIFIED status", fu.AUDITOR_STATUSES, index=fu.AUDITOR_STATUSES.index(f["auditor_status"]), disabled=locked)
            d["followup_date"] = c2.text_input("Follow-up date (YYYY-MM-DD)", f["followup_date"], disabled=locked)
            if st.form_submit_button("Save", type="primary", disabled=locked) and run(lambda: fu.update_followup(f["id"], d, user), "Saved."): st.rerun()
        c1, c2, c3 = st.columns(3)
        if f["review_status"] in ("Draft", "Returned") and c1.button("Submit for review", disabled=not _can(user)) and run(lambda: fu.submit(f["id"], user), "Submitted."): st.rerun()
        if f["review_status"] == "Awaiting Review" and dbm.can(user["role"], "approve"):
            cm = st.text_area("Reviewer comments", key=f"fr{f['id']}")
            if c1.button("Mark reviewed") and run(lambda: fu.review(f["id"], "Reviewed", cm, user), "Reviewed."): st.rerun()
            if c2.button("Return") and run(lambda: fu.review(f["id"], "Returned", cm, user), "Returned."): st.rerun()
        a = anns.get(f["annual_engagement_id"]); nm = ex.safe_name(a["ref"] if a else "followup", f"followup-{f['id']}")
        c3.download_button("Word", ex.followup_docx(f, a), nm + ".docx", DOCX); c3.download_button("Excel (all records)", ex.followup_xlsx(lst, anns), "followup-register.xlsx", XLSX)


# =========================== Settings extras ===========================
def settings_templates_tab(user):
    st.markdown("Load starter template structures for Transport and IT. They are flagged **'Draft structure - source to be confirmed'** because the "
                "source documents were not available when they were built; edit them from your sources (each edit creates a new version).")
    if user["role"] != "Administrator": st.info("Administrator only."); return
    if st.button("Load starter templates") : 
        r = run(lambda: seeds.seed(user)); 
        if r is not False: st.success(f"Created {len(r)} template(s)." if r else "All starter templates already exist.")
