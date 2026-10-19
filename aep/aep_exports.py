"""Export, Drive upload (retry-safe queue) and Drive deletion for Audit Engagement Programs. No Streamlit.

Export row lifecycle (aep_exports.drive_status):
  manual    - Drive not configured; file is downloaded only. NEVER shown as uploaded.
  pending   - generated, upload not yet confirmed (content kept in the DB so it can be retried)
  failed    - upload attempt failed (content kept; Retry is safe)
  uploaded  - Google Drive returned a file id (content blob cleared to save space)
The local download is always returned to the caller, whatever happens with Drive.

Duplicate prevention: one row per (program uid, program VERSION, format). The version increases on every save, so re-exporting an
unchanged program reuses that row and never uploads twice (generated files are not byte-identical between runs because Word/Excel/PDF
embed timestamps, so a content hash cannot be the key; the SHA-256 is kept for integrity only). Before each upload Drive is also asked
whether a file with the same uid+version+format exists, which covers an earlier attempt that succeeded but whose reply was lost."""
import hashlib

import aep
import aep_export as ex
import database as db
import drive as gd


def _client(client):
    return client if client is not None else gd.configured_client()


def _props(p_uid, ref, version, sha, fmt):
    return {"aep_uid": p_uid, "aep_ref": ref, "aep_version": str(version), "sha256": sha, "fmt": fmt}


def list_exports(uid):
    with db.connect() as c:
        return [dict(r) for r in c.execute("SELECT id,fmt,filename,drive_status,drive_file_id,drive_error,attempts,created_by,created_at,uploaded_at,size_bytes "
                                           "FROM aep_exports WHERE program_uid=? ORDER BY id DESC", (uid,))]


def pending_count():
    with db.connect() as c:
        return c.execute("SELECT COUNT(*) FROM aep_exports WHERE drive_status IN ('pending','failed')").fetchone()[0]


def export_program(pid, fmt, user, client=None, upload=True):
    """Generate the document, record it, and try to upload it. Always returns the bytes for local download:
    {"export_id","filename","mime","data","drive_status","message"}. Raises ValueError for a bad format/missing program.
    A failing export (builder error) is reported as ValueError and writes nothing."""
    db._need(user, "edit")
    p = aep.get_program(pid)
    try:
        data, name, mime = ex.build(p, fmt)
    except ValueError:
        raise
    except Exception as e:  # builder/library failure: no partial record is created
        raise ValueError(f"The {ex.FORMATS.get(fmt, (fmt,))[0]} export could not be generated ({type(e).__name__}). Your program is unchanged.") from None
    sha = hashlib.sha256(data).hexdigest()
    client = _client(client) if upload else None
    with db.connect(immediate=True) as c:
        row = c.execute("SELECT * FROM aep_exports WHERE program_uid=? AND fmt=? AND program_version=? AND drive_status!='manual' "
                        "ORDER BY id DESC LIMIT 1", (p["uid"], fmt, p["version"])).fetchone()
        if row:  # duplicate content: reuse the record (and its upload status) instead of creating/uploading another
            eid = row["id"]
        else:
            status = "pending" if client is not None else "manual"
            eid = c.execute("INSERT INTO aep_exports(program_uid,program_ref,program_version,fmt,filename,mime,sha256,size_bytes,content,"
                            "drive_status,created_by,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (p["uid"], p["program_ref"], p["version"], fmt, name, mime, sha, len(data),
                             data if client is not None else None, status, user["username"], db.now())).lastrowid
        db._log(c, user, "export_aep", f"{p['program_ref']} {fmt}")
    out = {"export_id": eid, "filename": name, "mime": mime, "data": data}
    if client is None:
        out.update(drive_status="manual", message=gd.MESSAGES["not_configured"] + " Download the file and upload it to the Drive folder manually.")
        return out
    st, msg = upload_export(eid, client)
    out.update(drive_status=st, message=msg)
    return out


def upload_export(export_id, client=None):
    """Upload (or retry) one export. Returns (status, user-safe message). Safe to call repeatedly."""
    client = _client(client)
    with db.connect() as c:
        r = c.execute("SELECT * FROM aep_exports WHERE id=?", (export_id,)).fetchone()
    if not r:
        return "failed", "That export record no longer exists."
    if r["drive_status"] == "uploaded":
        return "uploaded", "Already uploaded to Google Drive; no duplicate was created."
    if client is None:
        return "manual", gd.MESSAGES["not_configured"]
    if r["content"] is None:
        return "failed", "The stored copy of this export is unavailable. Export the program again."
    props = _props(r["program_uid"], r["program_ref"], r["program_version"], r["sha256"], r["fmt"])
    try:
        existing = client.find_existing(r["program_uid"], r["program_version"], r["fmt"])
        file_id = existing["id"] if existing else client.upload(r["filename"], r["mime"], bytes(r["content"]), props)
    except gd.DriveError as e:
        with db.connect() as c:
            c.execute("UPDATE aep_exports SET drive_status='failed', drive_error=?, attempts=attempts+1 WHERE id=?", (e.kind, export_id))
        return "failed", f"Not uploaded to Google Drive: {e} Your download and saved program are unaffected. You can retry."
    with db.connect() as c:
        c.execute("UPDATE aep_exports SET drive_status='uploaded', drive_file_id=?, drive_error='', attempts=attempts+1, uploaded_at=?, content=NULL "
                  "WHERE id=?", (file_id, db.now(), export_id))
    return "uploaded", "Uploaded to Google Drive (confirmed by Drive)."


def retry_failed(user, client=None):
    """Retry every pending/failed upload. Returns {"uploaded":n,"failed":n}."""
    db._need(user, "edit")
    client = _client(client)
    with db.connect() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM aep_exports WHERE drive_status IN ('pending','failed') ORDER BY id")]
    res = {"uploaded": 0, "failed": 0}
    for i in ids:
        st, _ = upload_export(i, client)
        res["uploaded" if st == "uploaded" else "failed"] += 1
    return res


# ------------------------------------------------------------------ STEP B: Drive deletion (separate from DB deletion)
def get_deletion(deletion_id):
    with db.connect() as c:
        d = c.execute("SELECT * FROM aep_deletions WHERE id=?", (deletion_id,)).fetchone()
        if not d:
            return None
        d = dict(d)
        d["files"] = [dict(r) for r in c.execute("SELECT * FROM aep_deletion_files WHERE deletion_id=? ORDER BY id", (deletion_id,))]
        return d


def open_deletions():
    """Deletion manifests that still have Drive files that were not deleted (retained/failed)."""
    with db.connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT d.id, d.program_ref, d.deleted_at, d.drive_decision, "
            "(SELECT COUNT(*) FROM aep_deletion_files f WHERE f.deletion_id=d.id AND f.status IN ('retained','failed')) AS open_files "
            "FROM aep_deletions d WHERE EXISTS(SELECT 1 FROM aep_deletion_files f WHERE f.deletion_id=d.id AND f.status IN ('retained','failed')) "
            "ORDER BY d.id DESC")]


def decline_drive_deletion(deletion_id, user):
    """User chose to keep the Drive documents: record it; nothing is touched in Drive."""
    with db.connect() as c:
        c.execute("UPDATE aep_deletions SET drive_decision='declined' WHERE id=?", (deletion_id,))
        db._log(c, user, "keep_drive_files", str(deletion_id))


def delete_drive_files(deletion_id, user, confirmed=False, client=None):
    """STEP B. Permanently delete ONLY the Drive files listed in this deletion's manifest, after verifying each file is
    in the configured folder and tagged with the deleted program's uid. Per-file outcome is recorded; failures stay
    'failed' so the call can be repeated. Returns the per-file result list."""
    db._need(user, "drive_delete")
    if confirmed is not True:
        raise ValueError("Drive deletion was not confirmed. No Drive files were deleted.")
    d = get_deletion(deletion_id)
    if not d:
        raise ValueError("Deletion record not found.")
    client = _client(client)
    if client is None:
        raise ValueError(gd.MESSAGES["not_configured"] + " Delete the files manually in Google Drive; the manifest still lists them.")
    try:  # also pick up files tagged with this program's uid whose upload reply was lost (still only this program's files)
        known = {f["drive_file_id"] for f in d["files"]}
        with db.connect() as c:
            for extra in client.list_by_uid(d["program_uid"]):
                if extra["id"] not in known:
                    c.execute("INSERT INTO aep_deletion_files(deletion_id,drive_file_id,filename,status,updated_at) VALUES(?,?,?,?,?)",
                              (deletion_id, extra["id"], extra.get("name", ""), "retained", db.now()))
        d = get_deletion(deletion_id)
    except gd.DriveError:
        pass  # discovery is best-effort; the manifest ids are still processed
    results = []
    for f in d["files"]:
        if f["status"] not in ("retained", "failed"):
            results.append({"filename": f["filename"], "status": f["status"], "detail": f["detail"]}); continue
        status, detail = "failed", ""
        try:
            meta = client.get_meta(f["drive_file_id"])
            props = meta.get("appProperties") or {}
            in_folder = client.folder_id in (meta.get("parents") or [])
            if props.get("aep_uid") != d["program_uid"] or not in_folder:
                status, detail = "skipped_mismatch", "File does not belong to this program/folder; not deleted."
            else:
                client.delete(f["drive_file_id"])
                status, detail = "deleted", ""
        except gd.DriveError as e:
            if e.kind == "not_found":
                status, detail = "missing", "File was not found in Drive (already removed or inaccessible)."
            else:
                status, detail = "failed", str(e)
        with db.connect() as c:
            c.execute("UPDATE aep_deletion_files SET status=?, detail=?, updated_at=? WHERE id=?", (status, detail, db.now(), f["id"]))
        results.append({"filename": f["filename"], "status": status, "detail": detail})
    with db.connect() as c:
        left = c.execute("SELECT COUNT(*) FROM aep_deletion_files WHERE deletion_id=? AND status IN ('retained','failed')", (deletion_id,)).fetchone()[0]
        c.execute("UPDATE aep_deletions SET drive_decision=? WHERE id=?", ("confirmed" if not left else "partial", deletion_id))
        db._log(c, user, "delete_drive_files", f"deletion {deletion_id}: " + ", ".join(f"{r['status']}" for r in results))
    return results
