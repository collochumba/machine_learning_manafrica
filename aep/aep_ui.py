"""Audit Engagement Programs: Streamlit UI. Screens: register, new program, template import, editor, formal view/print,
export (Word/Excel/PDF/HTML + Drive upload status), delete (step A: database) and delete Drive files (step B, separate
confirmation). Logic lives in aep.py, aep_import.py, aep_export.py, aep_exports.py and drive.py."""
import base64
import html
import time
from datetime import date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import aep
import aep_export as ex
import aep_exports as ae
import aep_import as imp
import drive as gd
import database as dbm
import theme as th

BROWN, DARK_BROWN, CREAM = "#7A4A1E", "#5C3514", "#FAF6EA"
PAGE_SIZE = 10

_CSS = f"""<style>
.stApp{{background:{CREAM}}}
.aep-banner{{background:{th.DARK_GREEN};border-bottom:4px solid {th.GOLD};border-radius:6px;padding:.7rem 1.1rem;display:flex;align-items:center;gap:1rem;margin-bottom:1rem}}
.aep-banner img{{height:58px;background:#fff;border-radius:4px;padding:2px}}
.aep-banner .org{{color:#fff;font-weight:800;letter-spacing:.05em;font-size:1.05rem}}
.aep-banner .sys{{color:{th.GOLD};font-weight:700;letter-spacing:.04em;font-size:.95rem}}
.aep-banner .dep{{color:#e8efe9;font-size:.8rem}}
.stButton>button[kind=primary],.stFormSubmitButton>button[kind=primary]{{background:{BROWN};border-color:{BROWN};color:#fff}}
.stButton>button[kind=primary]:hover{{background:{DARK_BROWN};border-color:{DARK_BROWN};color:#fff}}
.aep-th{{background:{th.DARK_GREEN};color:#fff;padding:6px 8px;font-weight:700;font-size:.82rem;border-bottom:3px solid {th.GOLD}}}
.aep-sec{{background:#F3E7B8;border-left:5px solid {th.GOLD};padding:.45rem .7rem;font-weight:700;color:{th.DARK_GREEN};margin:1rem 0 .3rem}}
.aep-opt{{border:1px solid #cdd8cd;border-radius:8px;padding:.9rem 1rem;margin:.6rem 0 .3rem;min-height:130px}}
.aep-opt.green{{background:#EAF3EA;border-color:#9DBE9D}} .aep-opt.cream{{background:#FBF1D6;border-color:{th.GOLD}}}
.aep-opt b{{display:block;font-size:1.02rem;margin-bottom:.3rem}}
.aep-meta{{width:100%;border-collapse:collapse;background:#fff;font-size:.88rem;margin:.4rem 0}}
.aep-meta td{{border:1px solid #d9d3c0;padding:6px 9px}} .aep-meta td.k{{font-weight:700;background:#F6F1E0;width:16%}}
.aep-doc{{background:#fff;border:1px solid #bbb;padding:28px 34px;color:#111;font-family:Georgia,'Times New Roman',serif}}
.aep-doc h1{{font-size:1.1rem;margin:.1rem 0;text-align:center;color:#111 !important}}
.aep-doc table{{border-collapse:collapse;width:100%;font-size:.82rem;margin-bottom:.6rem}}
.aep-doc th,.aep-doc td{{border:1px solid #444;padding:5px 7px;vertical-align:top;text-align:left;white-space:pre-wrap;word-break:break-word}}
.aep-doc th{{background:#eee}} .aep-doc td.k{{background:#f4f4f4;font-weight:700;width:16%}}
.aep-doc .sec{{background:#F3E7B8;border:1px solid #444;padding:5px 8px;font-weight:700;margin-top:.8rem}}
.aep-doc .sig{{display:flex;justify-content:space-between;margin-top:1.4rem;font-size:.9rem}}
@media print{{
 [data-testid=stSidebar],[data-testid=stHeader],header,.stButton,iframe,[data-testid=stToolbar],.aep-banner{{display:none !important}}
 .stApp{{background:#fff}} .block-container{{padding:0 !important;max-width:100% !important}}
 .aep-doc{{border:none;padding:0}} .aep-doc thead{{display:table-header-group}} .aep-doc tr{{page-break-inside:avoid}}
 .aep-doc .sec{{page-break-after:avoid}}
}}
{ex.DOC_CSS}</style>"""


def _logo_uri():
    return ("data:image/png;base64," + base64.b64encode(th.LOGO_PATH.read_bytes()).decode()) if th.logo_available() else ""


def _e(v):
    return html.escape(str(v or ""))


def _banner():
    logo = _logo_uri()
    img = f'<img src="{logo}" alt="KSG logo">' if logo else ""
    st.markdown(f'<div class="aep-banner">{img}<div><div class="org">KENYA SCHOOL OF GOVERNMENT (KSG)</div>'
                f'<div class="sys">AUDIT ENGAGEMENT PROGRAMS</div><div class="dep">Internal Audit Department</div></div></div>',
                unsafe_allow_html=True)


document_html = ex.document_html  # shared with the exports so the screen and the files match


# ---- state ---------------------------------------------------------------------------------------------------
def _fy_options(extra=""):
    y = date.today().year
    return sorted({f"{k}/{k + 1}" for k in range(y - 3, y + 4)} | set(aep.financial_years()) | ({extra} if extra else set()), reverse=True)


def _default_fy():
    y = date.today().year
    return f"{y}/{y + 1}" if date.today().month >= 7 else f"{y - 1}/{y}"  # Kenyan FY starts 1 July


def _blank_draft():
    return {"ref": "", "meta": {"title": "", "financial_year": _default_fy(), "audit_period": "", "campus": "", "status": "Draft"},
            "sections": [], "version": None}


def _go(mode, pid=None, fresh=True):
    S = st.session_state
    S.aep_mode, S.aep_id = mode, pid
    if fresh:
        S.pop("aep_draft", None)
    S["aep_nonce"] = S.get("aep_nonce", 0) + 1


def _ensure_draft(pid):
    S = st.session_state
    if "aep_draft" in S:
        return
    if pid:
        p = aep.get_program(pid)
        S.aep_draft = {"ref": p["program_ref"], "sections": p["sections"], "version": p["version"],
                       "meta": {k: p[k] for k in ("title", "financial_year", "audit_period", "campus", "status")}}
    else:
        S.aep_draft = _blank_draft()


def _details_widgets(prefix, meta, ref):
    """Program details inputs. Widget keys are prefixed per screen and seeded from the draft, so values survive screen changes."""
    S = st.session_state
    for k, v in meta.items():
        S.setdefault(f"{prefix}{k}", v)
    c1, c2 = st.columns(2)
    c1.text_input("Program Reference", value=ref or aep.next_ref(S[f"{prefix}financial_year"]) + "  (auto generated)", disabled=True)
    c2.text_input("Program Title *", key=f"{prefix}title", placeholder="Enter program title")
    c1.selectbox("Financial Year *", _fy_options(S[f"{prefix}financial_year"]), key=f"{prefix}financial_year")
    c2.text_input("Audit Period", key=f"{prefix}audit_period", placeholder="e.g. 1 July 2025 - 30 June 2026")
    c1.text_input("Campus / Location (if applicable)", key=f"{prefix}campus", placeholder="e.g. Nairobi Campus")
    c2.selectbox("Status *", aep.STATUSES, key=f"{prefix}status")
    return {k: S[f"{prefix}{k}"] for k in meta}


def _meta_table(ref, m):
    return (f'<table class="aep-meta"><tr><td class="k">Program Reference:</td><td>{_e(ref) or "(auto generated on save)"}</td>'
            f'<td class="k">Program Title:</td><td>{_e(m["title"])}</td></tr><tr><td class="k">Financial Year:</td><td>{_e(m["financial_year"])}</td>'
            f'<td class="k">Audit Period:</td><td>{_e(m["audit_period"])}</td></tr></table>')


def _rows_df(rows):
    df = pd.DataFrame(rows, columns=["id"] + aep.ROW_FIELDS)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    return df.fillna({f: "" for f in aep.ROW_FIELDS})


def _df_rows(df):
    out = []
    for _, r in df.iterrows():
        rid = None if pd.isna(r.get("id")) else int(r["id"])
        out.append({"id": rid, **{f: ("" if pd.isna(r[f]) else str(r[f])) for f in aep.ROW_FIELDS}})
    return out


# ---- screen 1: register ------------------------------------------------------------------------------------------
def _list(user):
    st.markdown("### Audit Engagement Programs")
    st.caption("Create, manage and view audit engagement programs that guide how the audit will be conducted.")
    s = aep.stats()
    for col, (lab, val, acc) in zip(st.columns(4), [("Total Programs", s["total"], "green"), ("Completed", s["completed"], "green"),
                                                    ("In Progress", s["in_progress"], "gold"), ("Draft", s["draft"], "maroon")]):
        col.markdown(th.kpi_card(lab, val, acc), unsafe_allow_html=True)
    st.write("")
    _pending_notices(user)
    f1, f2, f3, f4 = st.columns([1.2, 1.8, 1.2, 2.4])
    fy = f1.selectbox("Financial Year", ["All Years"] + aep.financial_years(), key="aep_flt_fy")
    area = f2.selectbox("Main Audit Area", ["All Audit Areas"] + aep.heading_options(), key="aep_flt_area")
    stt = f3.selectbox("Status", ["All Statuses"] + aep.STATUSES, key="aep_flt_status")
    q = f4.text_input("Search", key="aep_flt_q", placeholder="Search program reference or title...")
    if st.button("＋ New Audit Engagement Program", type="primary", disabled=not dbm.can(user["role"], "edit")):
        _go("new"); st.rerun()
    rows = aep.list_programs(None if fy.startswith("All") else fy, None if area.startswith("All") else area,
                             None if stt.startswith("All") else stt, q)
    if not rows:
        st.info("No programs match the current filters." if s["total"] else "No programs yet. Click '＋ New Audit Engagement Program' to create one.")
        return
    pages = max(1, -(-len(rows) // PAGE_SIZE))
    page = st.number_input("Page", 1, pages, 1, key="aep_page") if pages > 1 else 1
    start = (page - 1) * PAGE_SIZE
    widths = [.4, 1.3, 2.0, 2.3, 1.0, 1.0, 3.9]
    for c, h in zip(st.columns(widths), ["No.", "Program Reference", "Program Title", "Main Audit Area", "Financial Year", "Status", "Actions"]):
        c.markdown(f'<div class="aep-th">{h}</div>', unsafe_allow_html=True)
    for n, r in enumerate(rows[start:start + PAGE_SIZE], start + 1):
        c = st.columns(widths)
        c[0].write(n); c[1].write(r["program_ref"]); c[2].write(r["title"]); c[3].write((r["main_audit_area"] or "-").replace("; ", ", "))
        c[4].write(r["financial_year"]); c[5].markdown(th.badge(r["status"]), unsafe_allow_html=True)
        a, b, x, d = c[6].columns(4)
        if a.button("View", key=f"aepv{r['id']}"): _go("view", r["id"]); st.rerun()
        if b.button("Edit", key=f"aepe{r['id']}", disabled=not dbm.can(user["role"], "edit")): _go("editor", r["id"]); st.rerun()
        if x.button("Export", key=f"aepx{r['id']}", disabled=not dbm.can(user["role"], "edit")): _go("export", r["id"]); st.rerun()
        if d.button("Delete", key=f"aepd{r['id']}", disabled=not dbm.can(user["role"], "delete"), help="Administrators only"): _go("delete", r["id"]); st.rerun()
    st.caption(f"Showing {start + 1} to {min(start + PAGE_SIZE, len(rows))} of {len(rows)} entries")


# ---- screen 2: new program ---------------------------------------------------------------------------------------------
def _valid_meta(m):
    if not m["title"].strip():
        st.error("Program Title is required."); return False
    if not m["financial_year"]:
        st.error("Financial Year is required."); return False
    return True


def _new():
    _ensure_draft(None)
    d = st.session_state.aep_draft
    st.markdown("### New Audit Engagement Program")
    st.caption("Create a new audit engagement program by entering the basic details and adding audit areas and procedures.")
    with st.container(border=True):
        st.markdown("##### Program Details")
        meta = _details_widgets("n_", d["meta"], d["ref"])
    o1, o2 = st.columns(2)
    o1.markdown('<div class="aep-opt green"><b>⬆ Paste from Template (Optional)</b>Paste audit program content from Word or Excel with the '
                'columns: Audit Area, Requirements, Procedure / Test and Control, or upload an Excel, CSV or Word file.</div>', unsafe_allow_html=True)
    o2.markdown('<div class="aep-opt cream"><b>📄 Start with Blank Program</b>Add audit area headings and enter your own audit areas, '
                'requirements, procedures/tests and controls manually.</div>', unsafe_allow_html=True)
    paste, blank = o1.button("Paste Template", type="primary"), o2.button("Continue to Add Audit Areas", type="primary")
    st.button("← Back to List", on_click=_go, args=("list",))
    if paste or blank:
        if _valid_meta(meta):
            d["meta"] = meta
            _go("import" if paste else "editor", None, fresh=False); st.rerun()


# ---- screen 2b: import -------------------------------------------------------------------------------------------------
def _import():
    _ensure_draft(None)
    d = st.session_state.aep_draft
    st.markdown("### Paste from Template")
    st.caption("Paste a table copied from Excel or Word, or upload a file. Check the column mapping and preview before importing.")
    tab1, tab2 = st.tabs(["Paste table", "Upload file (Excel, CSV, Word)"])
    rows = None
    try:
        with tab1:
            txt = st.text_area("Paste the table here (include the header row)", height=200, key="aep_imp_text")
            if txt.strip():
                rows = imp.parse_text(txt)
        with tab2:
            up = st.file_uploader("Excel (.xlsx), CSV or Word (.docx)", type=["xlsx", "csv", "docx"], key="aep_imp_file")
            if up is not None and rows is None:
                data, name = up.getvalue(), up.name.lower()
                if name.endswith(".xlsx"):
                    rows = imp.parse_xlsx(data, st.selectbox("Worksheet", imp.xlsx_sheets(data), key="aep_imp_sheet"))
                elif name.endswith(".csv"):
                    rows = imp.parse_csv(data)
                else:
                    tabs = imp.docx_tables(data)
                    k = st.selectbox("Table in the document", range(len(tabs)), key="aep_imp_tbl",
                                     format_func=lambda i: f"Table {i + 1}: {len(tabs[i])} rows - {' | '.join(tabs[i][0])[:70]}")
                    rows = tabs[k]
    except imp.ImportErr as e:
        st.error(str(e)); rows = None
    ok = False
    if rows:
        gh = imp.guess_header(rows)
        has_header = st.checkbox("The first rows include a header row", value=gh is not None, key="aep_imp_hdr")
        hidx = None
        if has_header:
            hidx = st.number_input("Header row number", 1, min(len(rows), 10), (gh or 0) + 1, key="aep_imp_hrow") - 1
        cols = [(rows[hidx][i] if hidx is not None and rows[hidx][i] else f"Column {i + 1}") for i in range(len(rows[0]))]
        auto = imp.auto_map(rows[hidx]) if hidx is not None else {}
        st.markdown("**Column mapping** (correct it if needed)")
        opts = [None] + list(range(len(cols)))
        mapping, cc = {}, st.columns(5)
        for c, f in zip(cc, imp.FIELDS + ["section"]):
            mapping[f] = c.selectbox(imp.LABELS[f], opts, index=opts.index(auto.get(f)) if auto.get(f) in opts else 0,
                                     format_func=lambda i: "(not mapped)" if i is None else cols[i], key=f"aep_map_{f}_{hidx}_{len(cols)}")
        try:
            secs, rep = imp.build(rows, hidx, mapping)
            ok = rep["row_count"] > 0
            st.success(f"{rep['row_count']} row(s) in {len(secs)} section(s) ready to import.") if ok else st.warning("No importable rows were found with this mapping.")
            if rep["ignored_columns"]: st.info("Columns not imported: " + ", ".join(rep["ignored_columns"]))
            if rep["skipped"]:
                st.warning(f"{len(rep['skipped'])} row(s) have nothing in the mapped columns and will not be imported:")
                st.dataframe(pd.DataFrame(rep["skipped"], columns=["Source row", "Content"]), hide_index=True, use_container_width=True)
            for i, s in enumerate(secs):
                st.markdown(f'<div class="aep-sec">{aep.letter(i)}. {_e(s["heading"])}</div>', unsafe_allow_html=True)
                df = pd.DataFrame(s["rows"], columns=imp.FIELDS).rename(columns={f: imp.LABELS[f] for f in imp.FIELDS})
                df.index = range(1, len(df) + 1); df.index.name = "No."
                st.dataframe(df, use_container_width=True)
        except imp.ImportErr as e:
            st.error(str(e)); secs = []
    b1, _, b2 = st.columns([1.3, 3, 1.6])
    b1.button("← Back", on_click=_go, args=("new",), kwargs={"fresh": False})
    if b2.button("Confirm Import", type="primary", disabled=not ok):
        d["sections"] = imp.merge_into(d["sections"], secs)
        _go("editor", None, fresh=False); st.rerun()


# ---- screen 3: editor --------------------------------------------------------------------------------------------------
def _editor(user, pid):
    try:
        _ensure_draft(pid)
    except ValueError as e:
        st.error(str(e)); st.button("Back to List", on_click=_go, args=("list",)); return
    S, d = st.session_state, st.session_state.aep_draft
    st.markdown("### Audit Engagement Programs")
    st.caption("Define the audit areas, requirements, procedures/tests and controls.")
    with st.expander("Program details (click to edit)", expanded=False):
        d["meta"] = _details_widgets(f"e{pid or 0}_", d["meta"], d["ref"])
    st.markdown(_meta_table(d["ref"], d["meta"]), unsafe_allow_html=True)

    nonce, current = S.get("aep_nonce", 0), []
    n_sec = len(d["sections"])
    for i, sec in enumerate(d["sections"]):
        k = f"aep_{nonce}_{i}"
        h1, h2, h3, h4 = st.columns([5, .6, .6, 1.6])
        h1.markdown(f'<div class="aep-sec">{aep.letter(i)}. {_e(sec["heading"])}</div>', unsafe_allow_html=True)
        up = h2.button("▲", key=k + "up", help="Move section up", disabled=i == 0)
        dn = h3.button("▼", key=k + "dn", help="Move section down", disabled=i == n_sec - 1)
        remove = h4.button("Remove Section", key=k + "rm")
        df = _rows_df(sec["rows"]); df.index = range(1, len(df) + 1); df.index.name = "No."
        ed = st.data_editor(df, num_rows="dynamic", key=k + "ed", use_container_width=True, column_order=aep.ROW_FIELDS,
                            column_config={"area": st.column_config.TextColumn("Audit Area", width="medium"),
                                           "requirements": st.column_config.TextColumn("Requirements", width="large"),
                                           "procedure": st.column_config.TextColumn("Procedure / Test", width="large"),
                                           "control": st.column_config.TextColumn("Control", width="large")})
        rows = _df_rows(ed)
        m0, m1, m2, m3, _ = st.columns([1.6, 1, .8, .8, 4])
        addrow = m0.button("＋ Add Procedure Row", key=k + "ar")
        rno = m1.number_input("Row No.", 1, max(1, len(rows)), 1, key=k + "rn", label_visibility="collapsed", help="Row number to move")
        rup = m2.button("Row ▲", key=k + "ru", disabled=len(rows) < 2)
        rdn = m3.button("Row ▼", key=k + "rd", disabled=len(rows) < 2)
        current.append({"id": sec.get("id"), "heading": sec["heading"], "rows": rows, "remove": remove, "up": up, "dn": dn, "addrow": addrow,
                        "rmove": (int(rno) - 1, -1) if rup else ((int(rno) - 1, 1) if rdn else None)})
    st.caption("Edit cells directly (double-click for long text). Select a row and press Delete to remove it. No. updates automatically.")
    with st.container(border=True):
        a1, a2 = st.columns([4, 1])
        used = {s["heading"].lower() for s in current}
        choice = a1.selectbox("Main audit area heading", [x for x in aep.heading_options() if x.lower() not in used] or ["(all headings added)"], key="aep_add_choice")
        custom = a1.text_input("...or type a custom heading", key=f"aep_add_custom_{nonce}")
        a2.write(""); a2.write("")
        add = a2.button("Add Section", type="primary")

    def sync():
        d["sections"] = [{"id": s["id"], "heading": s["heading"], "rows": s["rows"]} for s in current if not s["remove"]]

    changed = False
    for i, s in enumerate(current):
        if s["addrow"]:
            s["rows"].append({"id": None, **{f: "" for f in aep.ROW_FIELDS}}); changed = True
        if s["rmove"]:
            idx, dr = s["rmove"]; j = idx + dr
            if 0 <= idx < len(s["rows"]) and 0 <= j < len(s["rows"]):
                s["rows"][idx], s["rows"][j] = s["rows"][j], s["rows"][idx]
            changed = True
        if s["up"] and i > 0:
            current[i - 1], current[i] = current[i], current[i - 1]; changed = True; break
        if s["dn"] and i < len(current) - 1:
            current[i + 1], current[i] = current[i], current[i + 1]; changed = True; break
    if changed or any(s["remove"] for s in current):
        sync(); S["aep_nonce"] = nonce + 1; st.rerun()
    if add:
        head = custom.strip() or choice
        if head.startswith("(all"):
            st.error("Type a custom heading, or all headings are already added.")
        else:
            sync(); d["sections"].append({"id": None, "heading": head, "rows": []}); S["aep_nonce"] = nonce + 1; st.rerun()

    b1, _, b2, b3 = st.columns([1.2, 3, 1.2, 1.3])
    b1.button("← Back to List", on_click=_go, args=("list",))
    for col, label, status in ((b2, "Save as Draft", "Draft"), (b3, "💾 Save Program", None)):
        if col.button(label, type="primary" if status is None else "secondary", disabled=not dbm.can(user["role"], "edit")):
            sync()
            data = dict(d["meta"], program_ref=d["ref"], status=status or d["meta"]["status"])
            try:
                new_id = aep.save_program(data, d["sections"], user, pid, expected_version=d.get("version") if pid else None)
                S.aep_flash = "Audit Engagement Program saved."
                _go("view", new_id); st.rerun()
            except aep.ConflictError as e:
                S.aep_conflict = str(e)
            except (ValueError, PermissionError) as e:
                st.error(str(e))
    if S.get("aep_conflict"):
        st.error(S.aep_conflict)
        st.button("Reload the latest saved version (discards my unsaved edits)", on_click=lambda: (S.pop("aep_conflict", None), _go("editor", pid)))


# ---- screen 4: view ----------------------------------------------------------------------------------------------------
def _view(user, pid):
    try:
        p = aep.get_program(pid)
    except ValueError as e:
        st.error(str(e)); st.button("Back", on_click=_go, args=("list",)); return
    if st.session_state.pop("aep_flash", None):
        st.success("Audit Engagement Program saved.")
    st.markdown("### View / Print Audit Engagement Program")
    b = st.columns([1, 1.2, 1, 1, 3])
    with b[0]:
        components.html('<button onclick="window.parent.print()" style="padding:.45rem 1rem;border:1px solid #7A4A1E;'
                        'background:#fff;color:#7A4A1E;border-radius:6px;font-weight:600;cursor:pointer">🖨 Print</button>', height=48)
    if b[1].button("Export", disabled=not dbm.can(user["role"], "edit")): _go("export", pid); st.rerun()
    if b[2].button("✎ Edit", type="primary", disabled=not dbm.can(user["role"], "edit")): _go("editor", pid); st.rerun()
    b[3].button("← Back", on_click=_go, args=("list",))
    st.markdown(document_html(p), unsafe_allow_html=True)


# ---- Drive helpers ------------------------------------------------------------------------------------------------
@st.cache_resource(ttl=600, show_spinner=False)
def _drive_client():
    return gd.configured_client()


def _drive_state(force=False):
    """(state, message) cached for 60 s so every rerun does not call the Drive API."""
    S = st.session_state
    c = S.get("aep_drive_check")
    if force or not c or time.time() - c[0] > 60:
        if force:
            _drive_client.clear()
        S["aep_drive_check"] = c = (time.time(), *gd.status(_drive_client()))
    return c[1], c[2]


MANUAL_STEPS = (
    "**Manual upload (until automatic upload is enabled):** 1) click **Download** below; 2) open the KSG Audit Engagement Programs "
    f"folder in Google Drive: https://drive.google.com/drive/folders/{gd.folder_id()} ; 3) drag the downloaded file into the folder "
    "(or *New > File upload*). The application does not mark manually uploaded files as uploaded.")


def _pending_notices(user):
    n = ae.pending_count()
    if n:
        st.warning(f"{n} exported document(s) are waiting for a Google Drive upload (not yet confirmed). Open the program's Export screen to retry.")
    if dbm.can(user["role"], "drive_delete"):
        od = ae.open_deletions()
        if od:
            with st.expander(f"Deleted programs with Google Drive documents still in Drive ({len(od)})"):
                for x in od:
                    c1, c2 = st.columns([4, 1.4])
                    c1.write(f"{x['program_ref']} - deleted {x['deleted_at']} - {x['open_files']} Drive file(s) remaining ({x['drive_decision']})")
                    if c2.button("Review", key=f"aepod{x['id']}"): _go("delete_drive", x["id"]); st.rerun()


# ---- screen 5: export ----------------------------------------------------------------------------------------------------
def _export(user, pid):
    try:
        p = aep.get_program(pid)
    except ValueError as e:
        st.error(str(e)); st.button("Back to List", on_click=_go, args=("list",)); return
    S = st.session_state
    st.markdown("### Export Audit Engagement Program")
    st.caption(f"{p['program_ref']} - {p['title']}")
    state, msg = _drive_state()
    {"ready": st.success, "manual": st.warning, "error": st.error}[state](msg)
    if st.button("Re-check Google Drive connection", key="aep_recheck"):
        _drive_state(force=True); st.rerun()
    keys = list(ex.FORMATS)
    fmt = st.radio("Export format", keys, format_func=lambda k: ex.FORMATS[k][0], horizontal=True, key="aep_fmt")
    b1, b2, _ = st.columns([1.4, 1.2, 4])
    if b1.button("Generate export", type="primary", disabled=not dbm.can(user["role"], "edit")):
        try:
            with st.spinner("Generating..."):
                S.aep_last = dict(ae.export_program(pid, fmt, user, _drive_client()), pid=pid)
        except (ValueError, PermissionError) as e:
            st.error(str(e))
    b2.button("← Back", on_click=_go, args=("list",))
    last = S.get("aep_last")
    if last and last["pid"] == pid:
        {"uploaded": st.success, "failed": st.warning, "pending": st.warning, "manual": st.warning}.get(last["drive_status"], st.info)(last["message"])
        st.download_button(f"Download {last['filename']}", last["data"], last["filename"], last["mime"], type="primary", key="aep_dl")
        if last["drive_status"] == "manual":
            st.markdown(MANUAL_STEPS)
    hist = ae.list_exports(p["uid"])
    if hist:
        st.markdown("##### Export history")
        labels = {"uploaded": "Uploaded to Drive", "failed": "Upload failed", "pending": "Upload pending", "manual": "Download only (not in Drive)"}
        for h in hist[:15]:
            c = st.columns([3.6, 1.3, 2.2, 1.4])
            c[0].write(h["filename"]); c[1].write(ex.FORMATS[h["fmt"]][0]); c[2].write(labels.get(h["drive_status"], h["drive_status"]))
            if h["drive_status"] in ("failed", "pending"):
                if c[3].button("Retry upload", key=f"aeprt{h['id']}"):
                    st_, m_ = ae.upload_export(h["id"], _drive_client())
                    (st.success if st_ == "uploaded" else st.warning)(m_)
            elif h["drive_status"] == "uploaded":
                c[3].caption(h["uploaded_at"] or "")


# ---- screen 6: delete (step A) --------------------------------------------------------------------------------------------
def _delete(user, pid):
    try:
        p = aep.get_program(pid)
    except ValueError as e:
        st.error(str(e)); st.button("Back to List", on_click=_go, args=("list",)); return
    st.markdown("### Delete Audit Engagement Program")
    if not dbm.can(user["role"], "delete"):
        st.error("Only an Administrator can delete an Audit Engagement Program.")
        st.button("Back to List", on_click=_go, args=("list",)); return
    n_ex = sum(1 for h in ae.list_exports(p["uid"]) if h["drive_status"] == "uploaded")
    st.error(f"You are about to permanently delete **{p['program_ref']} - {p['title']}** "
             f"({len(p['sections'])} audit area section(s), {aep.count_rows(pid)} procedure row(s)).")
    st.markdown("This removes the program's database record and all of its procedure rows. **It cannot be undone.** "
               "Other programs are not affected. " + (f"{n_ex} exported document(s) for this program are recorded in Google Drive; "
               "they are **not** deleted by this step - you will be asked about them separately next." if n_ex else
               "No exported Google Drive documents are recorded for this program."))
    typed = st.text_input(f"Type the program reference ({p['program_ref']}) to confirm", key=f"aep_del_t{pid}")
    ok = st.checkbox("I understand this deletion is permanent.", key=f"aep_del_c{pid}")
    b1, b2, _ = st.columns([1.6, 1.2, 4])
    if b1.button("Permanently delete program", type="primary", disabled=not (ok and typed.strip().upper() == p["program_ref"].upper())):
        try:
            res = aep.delete_program(pid, user, confirmed=True)
            _go("delete_drive", res["deletion_id"]); st.rerun()
        except (ValueError, PermissionError) as e:
            st.error(str(e))
    b2.button("Cancel", on_click=_go, args=("list",))  # nothing was changed


# ---- screen 7: delete Drive files (step B, its own confirmation) ---------------------------------------------------------------
def _delete_drive(user, did):
    d = ae.get_deletion(did)
    if not d:
        st.error("Deletion record not found."); st.button("Back to List", on_click=_go, args=("list",)); return
    S = st.session_state
    st.markdown("### Google Drive documents")
    st.success(f"{d['program_ref']} was permanently deleted from the application database.")
    open_files = [f for f in d["files"] if f["status"] in ("retained", "failed")]
    if not d["files"]:
        st.info("No exported Google Drive documents are recorded for this program. Files that were uploaded to Drive manually are not tracked by the "
                "application and are not touched.")
    else:
        names = {"retained": "Still in Drive", "deleted": "Deleted from Drive", "failed": "Delete failed - can retry", "missing": "Not found in Drive",
                 "skipped_mismatch": "Skipped (not this program's file)"}
        st.dataframe(pd.DataFrame([{"File": f["filename"], "Status": names.get(f["status"], f["status"]), "Detail": f["detail"]} for f in d["files"]]),
                     hide_index=True, use_container_width=True)
    if open_files:
        state, msg = _drive_state()
        st.warning("These documents are **still in Google Drive**. Deleting the program did not delete them. You can keep them, or permanently delete "
                   "exactly these files below. Drive deletion is permanent and separate from the step you already completed.")
        if state != "ready":
            st.error(msg + " Drive deletion is unavailable; delete the files manually in Drive (names listed above).")
        can = dbm.can(user["role"], "drive_delete") and state == "ready"
        if not dbm.can(user["role"], "drive_delete"):
            st.info("Only an Administrator can delete Drive documents.")
        c = st.checkbox(f"I understand that {len(open_files)} Google Drive file(s) will be permanently deleted and cannot be recovered.", key=f"aep_dd_c{did}")
        t = st.text_input("Type DELETE FROM DRIVE to confirm", key=f"aep_dd_t{did}")
        b1, b2, _ = st.columns([2, 1.8, 3])
        if b1.button("Permanently delete these Drive files", type="primary", disabled=not (can and c and t.strip() == "DELETE FROM DRIVE")):
            try:
                res = ae.delete_drive_files(did, user, confirmed=True, client=_drive_client())
                bad = [r for r in res if r["status"] == "failed"]
                (st.warning if bad else st.success)(f"{sum(r['status'] == 'deleted' for r in res)} file(s) deleted" + (f"; {len(bad)} could not be deleted - listed above, you can retry." if bad else "."))
                S[f"aep_dd_done{did}"] = True
            except (ValueError, PermissionError) as e:
                st.error(str(e))
        if b2.button("Keep documents in Drive"):
            ae.decline_drive_deletion(did, user)
            S.aep_flash_msg = "The exported documents remain in Google Drive."
            _go("list"); st.rerun()
    elif d["files"]:
        st.info("No Drive files remain to be deleted for this program.")
    st.button("← Back to List", on_click=_go, args=("list",), key="aep_dd_back")


def page_aep(user):
    st.markdown(_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("aep_mode", "list")
    _banner()
    msg = st.session_state.pop("aep_flash_msg", None)
    if msg:
        st.info(msg)
    mode, pid = st.session_state.aep_mode, st.session_state.get("aep_id")
    try:
        if mode == "view": _view(user, pid)
        elif mode == "editor": _editor(user, pid)
        elif mode == "new": _new()
        elif mode == "import": _import()
        elif mode == "export": _export(user, pid)
        elif mode == "delete": _delete(user, pid)
        elif mode == "delete_drive": _delete_drive(user, pid)
        else: _list(user)
    except dbm.DatabaseBusy as e:
        st.error(str(e))
