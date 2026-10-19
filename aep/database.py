"""SQLite data layer: schema, validation, permissions, versions, backups."""
import hashlib, json, os, secrets, shutil, sqlite3, uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.getenv("AWP_DATA_DIR", "data"))
# AEP_DB_PATH points the live SQLite file at the host's PERSISTENT disk/volume (e.g. /var/data/aep.db).
DB_PATH = Path(os.getenv("AEP_DB_PATH") or (DATA_DIR / "awp.db"))
EVIDENCE_DIR = DATA_DIR / "evidence"
BACKUP_DIR = Path(os.getenv("AEP_BACKUP_DIR") or (DB_PATH.parent / "backups"))
BACKUP_KEEP = int(os.getenv("AEP_BACKUP_KEEP", "30"))
BUSY_TIMEOUT_MS = int(os.getenv("AEP_DB_BUSY_TIMEOUT_MS", "10000"))
# WAL gives readers/writers better concurrency on a local persistent disk; use DELETE on network filesystems.
JOURNAL_MODE = os.getenv("AEP_SQLITE_JOURNAL", "WAL").upper()
MIN_PASSWORD = 12
LEGACY_DEFAULT_PASSWORD = "ChangeMe!2024"  # only used to DETECT and force-change old seeded accounts

ROLES = ("Administrator", "Auditor", "Reviewer")
PERMS = {"edit": {"Administrator", "Auditor"}, "approve": {"Administrator", "Reviewer"},
         "admin": {"Administrator"},
         "delete": {"Administrator"},        # permanently delete an AEP record
         "drive_delete": {"Administrator"}}  # permanently delete exported documents in Google Drive
PLAN_STATUSES = ["Draft", "In Progress", "Awaiting Review", "Returned", "Approved"]
ROW_STATUSES = ["Not Started", "In Progress", "Completed"]
LOCKED = ("Awaiting Review", "Approved")
ENG_FIELDS = ["ref", "title", "org", "category", "area", "sub_area", "period", "start_date", "end_date",
              "objective", "scope", "policies", "team", "reviewer", "risk_level", "instructions"]
ROW_FIELDS = ["wp_ref", "area", "objective", "risk", "risk_level", "control_objective", "criteria",
              "procedure", "evidence_required", "sampling_method", "sample_size", "auditor", "expected",
              "actual", "evidence_ref", "conclusion", "review_comments", "status"]
LIB_FIELDS = ["kind", "category", "title", "content", "source", "doc_title", "clause", "exact_wording"]
LIB_KINDS = ["Risk", "Control objective", "Audit criterion", "Audit procedure", "Evidence requirement",
             "Sampling approach", "Expected result", "Custom area"]

_T = lambda cols: ", ".join(f"{c} TEXT" for c in cols)
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt TEXT, pw_hash TEXT, role TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS engagements(id INTEGER PRIMARY KEY, {_T(ENG_FIELDS)}, custom_cols TEXT DEFAULT '[]',
  status TEXT DEFAULT 'Draft', version INTEGER DEFAULT 1, is_demo INTEGER DEFAULT 0, created_by TEXT, created_at TEXT,
  updated_at TEXT, approved_by TEXT, approved_at TEXT, UNIQUE(ref));
CREATE TABLE IF NOT EXISTS plan_rows(id INTEGER PRIMARY KEY, engagement_id INTEGER NOT NULL REFERENCES engagements(id) ON DELETE CASCADE, {_T(ROW_FIELDS)}, extra TEXT DEFAULT '{{}}');
CREATE TABLE IF NOT EXISTS library(id INTEGER PRIMARY KEY, {_T(LIB_FIELDS)}, verified INTEGER DEFAULT 0, created_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS saved_templates(id INTEGER PRIMARY KEY, name TEXT UNIQUE, category TEXT, rows TEXT, created_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS comments(id INTEGER PRIMARY KEY, engagement_id INTEGER REFERENCES engagements(id) ON DELETE CASCADE, author TEXT, text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS versions(id INTEGER PRIMARY KEY, engagement_id INTEGER REFERENCES engagements(id) ON DELETE CASCADE, version INTEGER, snapshot TEXT, note TEXT, created_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS evidence(id INTEGER PRIMARY KEY, engagement_id INTEGER REFERENCES engagements(id) ON DELETE CASCADE, filename TEXT, path TEXT, uploaded_by TEXT, uploaded_at TEXT);
CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY, ts TEXT, user TEXT, action TEXT, detail TEXT);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


class DatabaseBusy(ValueError):
    """Raised (with a user-safe message) when SQLite stays locked past the busy timeout."""


@contextmanager
def connect(immediate=False):
    """Open the live database. immediate=True takes the write lock up front (BEGIN IMMEDIATE) so a
    read-check-write sequence (e.g. optimistic-lock version check) is atomic against other writers."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=BUSY_TIMEOUT_MS / 1000)
    c.row_factory = sqlite3.Row
    try:
        c.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")
        c.execute("PRAGMA foreign_keys=ON")
        if JOURNAL_MODE in ("WAL", "DELETE", "TRUNCATE"):
            c.execute(f"PRAGMA journal_mode={JOURNAL_MODE}")
        if immediate:
            c.execute("BEGIN IMMEDIATE")
        yield c
        c.commit()
    except sqlite3.OperationalError as e:
        c.rollback()
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            raise DatabaseBusy("The database is busy because another user is saving. Nothing was lost - "
                               "please wait a few seconds and try again.") from None
        raise
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def can(role, action):
    return role in PERMS.get(action, set())


def _need(user, action):
    if not can(user["role"], action):
        raise PermissionError(f"Role '{user['role']}' is not permitted to {action}.")


def _log(c, user, action, detail=""):
    c.execute("INSERT INTO audit_log(ts,user,action,detail) VALUES(?,?,?,?)",
              (now(), user["username"] if isinstance(user, dict) else user, action, detail))


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 120_000).hex()


def init_db():
    with connect() as c:
        c.executescript(SCHEMA)
    import migrations  # versioned additive migrations; backs up first, rolls back on failure
    applied = migrations.apply(DB_PATH, lambda: backup_db("pre-migration"))
    if applied:
        with connect() as c:
            _log(c, "system", "migrate", ",".join(map(str, applied)))
    import plan_register
    plan_register.ensure_seed()
    # No shared default password: accounts are only auto-seeded if the operator supplies AWP_DEFAULT_PASSWORD
    # (12+ chars). Otherwise the first visitor creates the administrator through the first-run setup screen.
    pw = os.getenv("AWP_DEFAULT_PASSWORD") or ""
    with connect() as c:
        if not c.execute("SELECT 1 FROM users").fetchone() and len(pw) >= MIN_PASSWORD and pw != LEGACY_DEFAULT_PASSWORD:
            for u, r in (("admin", "Administrator"), ("auditor", "Auditor"), ("reviewer", "Reviewer")):
                s_ = secrets.token_hex(16)
                c.execute("INSERT INTO users(username,salt,pw_hash,role) VALUES(?,?,?,?)", (u, s_, _hash(pw, s_), r))


def needs_setup():
    """True when no user exists yet (first run): the UI then shows the create-administrator form."""
    with connect() as c:
        return c.execute("SELECT 1 FROM users").fetchone() is None


def create_first_admin(username, pw):
    """Create the first Administrator. Refused once any user exists, so it can't be used as a back door."""
    username = (username or "").strip()
    if not username or len(username) > 64:
        raise ValueError("Enter a username (up to 64 characters).")
    if len(pw or "") < MIN_PASSWORD or pw == LEGACY_DEFAULT_PASSWORD:
        raise ValueError(f"Choose a password of at least {MIN_PASSWORD} characters.")
    s_ = secrets.token_hex(16)
    with connect(immediate=True) as c:
        if c.execute("SELECT 1 FROM users").fetchone():
            raise PermissionError("Setup is already complete. Sign in instead.")
        c.execute("INSERT INTO users(username,salt,pw_hash,role) VALUES(?,?,?,?)", (username, s_, _hash(pw, s_), "Administrator"))
        _log(c, "system", "first_admin_created", username)


def list_users():
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT username, role FROM users ORDER BY username")]


_FAILS = {}  # username -> [timestamps]; per-process login throttle (5 failures / 15 min)
_MAX_FAILS, _WINDOW = 5, 15 * 60


def login_blocked(username):
    import time
    k = (username or "").strip().lower()
    _FAILS[k] = [t for t in _FAILS.get(k, []) if time.time() - t < _WINDOW]
    return len(_FAILS[k]) >= _MAX_FAILS


def authenticate(username, pw):
    """Returns {"username","role","must_change"} or None. Blocks an account after repeated failures."""
    import time
    k = (username or "").strip().lower()
    if login_blocked(username):
        raise PermissionError("Too many failed sign-in attempts. Wait 15 minutes and try again.")
    with connect() as c:
        r = c.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
        if r and secrets.compare_digest(_hash(pw, r["salt"]), r["pw_hash"]):
            _FAILS.pop(k, None)
            _log(c, r["username"], "login")
            legacy = secrets.compare_digest(_hash(LEGACY_DEFAULT_PASSWORD, r["salt"]), r["pw_hash"])
            return {"username": r["username"], "role": r["role"], "must_change": legacy}
    _FAILS.setdefault(k, []).append(time.time())
    return None


def save_user(username, pw, role, actor):
    _need(actor, "admin")
    if role not in ROLES or not username.strip() or len(pw) < MIN_PASSWORD or pw == LEGACY_DEFAULT_PASSWORD:
        raise ValueError(f"A valid username, role and a password of {MIN_PASSWORD}+ characters are required.")
    s = secrets.token_hex(16)
    with connect() as c:
        c.execute("INSERT INTO users(username,salt,pw_hash,role) VALUES(?,?,?,?) ON CONFLICT(username) "
                  "DO UPDATE SET salt=excluded.salt, pw_hash=excluded.pw_hash, role=excluded.role",
                  (username.strip(), s, _hash(pw, s), role))
        _log(c, actor, "save_user", username)


def change_own_password(user, pw):
    if len(pw) < MIN_PASSWORD or pw == LEGACY_DEFAULT_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
    s = secrets.token_hex(16)
    with connect() as c:
        c.execute("UPDATE users SET salt=?, pw_hash=? WHERE username=?", (s, _hash(pw, s), user["username"]))
        _log(c, user, "change_password")


# ---------- engagements ----------
def _eng(c, eid):
    r = c.execute("SELECT * FROM engagements WHERE id=?", (eid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["custom_cols"] = json.loads(d["custom_cols"] or "[]")
    return d


def get_engagement(eid):
    with connect() as c:
        return _eng(c, eid)


def list_engagements():
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM engagements ORDER BY id DESC")]


def _next_ref(c, category):
    w = category.split()
    code = (w[0][:4] if len(w) == 1 else "".join(x[0] for x in w) or "GEN").upper()
    n = c.execute("SELECT COUNT(*) FROM engagements").fetchone()[0] + 1
    while True:
        ref = f"AWP/{code}/{datetime.now().year}/{n:03d}"
        if not c.execute("SELECT 1 FROM engagements WHERE ref=?", (ref,)).fetchone():
            return ref
        n += 1


def _clean_eng(data):
    d = {f: str(data.get(f) or "").strip() for f in ENG_FIELDS}
    for f in ("title", "org", "category"):
        if not d[f]:
            raise ValueError(f"'{f}' is required.")
    if d["start_date"] and d["end_date"] and d["end_date"] < d["start_date"]:
        raise ValueError("Audit end date cannot be before the start date.")
    return d


def create_engagement(data, user, is_demo=False):
    _need(user, "edit")
    d = _clean_eng(data)
    with connect() as c:
        d["ref"] = d["ref"] or _next_ref(c, d["category"])
        if c.execute("SELECT 1 FROM engagements WHERE ref=?", (d["ref"],)).fetchone():
            raise ValueError(f"Reference '{d['ref']}' already exists.")
        cols = ENG_FIELDS + ["created_by", "created_at", "updated_at", "is_demo"]
        vals = [d[f] for f in ENG_FIELDS] + [user["username"], now(), now(), int(is_demo)]
        cur = c.execute(f"INSERT INTO engagements({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)
        _log(c, user, "create_engagement", d["ref"])
        return cur.lastrowid


def _check_editable(c, eid, user):
    _need(user, "edit")
    e = _eng(c, eid)
    if e["status"] in LOCKED:
        raise PermissionError(f"Plan is '{e['status']}' and locked. Return it or create a new version to edit.")
    return e


def update_engagement(eid, data, user):
    with connect() as c:
        e = _check_editable(c, eid, user)
        d = _clean_eng({**e, **data})
        if c.execute("SELECT 1 FROM engagements WHERE ref=? AND id<>?", (d["ref"], eid)).fetchone():
            raise ValueError(f"Reference '{d['ref']}' already exists.")
        c.execute(f"UPDATE engagements SET {','.join(f + '=?' for f in ENG_FIELDS)}, updated_at=? WHERE id=?",
                  [d[f] for f in ENG_FIELDS] + [now(), eid])
        _log(c, user, "update_engagement", d["ref"])


def delete_engagement(eid, user):
    _need(user, "admin")
    with connect() as c:
        e = _eng(c, eid)
        try:
            c.execute("DELETE FROM engagements WHERE id=?", (eid,))
        except sqlite3.IntegrityError:
            raise ValueError("Cannot delete: this AWP is linked to an annual engagement, working papers or follow-up "
                             "records. Archive or return it instead.") from None
        _log(c, user, "delete_engagement", e["ref"])


def add_custom_col(eid, name, user):
    name = name.strip()
    with connect() as c:
        e = _check_editable(c, eid, user)
        if not name or name in ROW_FIELDS or name in e["custom_cols"]:
            raise ValueError("Column name is empty or already exists.")
        c.execute("UPDATE engagements SET custom_cols=? WHERE id=?", (json.dumps(e["custom_cols"] + [name]), eid))
        _log(c, user, "add_custom_col", name)


# ---------- rows ----------
def get_rows(eid):
    with connect() as c:
        return _rows(c, eid)


def _rows(c, eid):
    out = []
    for r in c.execute("SELECT * FROM plan_rows WHERE engagement_id=? ORDER BY id", (eid,)):
        d = {f: r[f] or "" for f in ROW_FIELDS}
        d["extra"] = json.loads(r["extra"] or "{}")
        out.append(d)
    return out


def replace_rows(eid, rows, user):
    """Atomically replace all rows (order = list order)."""
    with connect() as c:
        _check_editable(c, eid, user)
        c.execute("DELETE FROM plan_rows WHERE engagement_id=?", (eid,))
        for r in rows:
            vals = [str(r.get(f, "") or "") for f in ROW_FIELDS]
            c.execute(f"INSERT INTO plan_rows(engagement_id,{','.join(ROW_FIELDS)},extra) VALUES(?,{','.join('?' * len(ROW_FIELDS))},?)",
                      [eid] + vals + [json.dumps(r.get("extra", {}))])
        c.execute("UPDATE engagements SET updated_at=? WHERE id=?", (now(), eid))
        _log(c, user, "save_rows", f"engagement {eid}: {len(rows)} rows")


# ---------- workflow, comments, versions ----------
def set_status(eid, new, user):
    if new not in PLAN_STATUSES:
        raise ValueError("Unknown status.")
    _need(user, "approve" if new in ("Approved", "Returned") else "edit")
    with connect() as c:
        e = _eng(c, eid)
        if new == "Approved" and e["status"] != "Awaiting Review":
            raise ValueError("Only plans awaiting review can be approved.")
        ab, at = (user["username"], now()) if new == "Approved" else (None, None)
        c.execute("UPDATE engagements SET status=?, approved_by=?, approved_at=?, updated_at=? WHERE id=?",
                  (new, ab, at, now(), eid))
        _log(c, user, "status", f"{e['ref']}: {e['status']} -> {new}")


def add_comment(eid, text, user):
    if not text.strip():
        raise ValueError("Comment is empty.")
    with connect() as c:
        c.execute("INSERT INTO comments(engagement_id,author,text,created_at) VALUES(?,?,?,?)",
                  (eid, user["username"], text.strip(), now()))
        _log(c, user, "comment", f"engagement {eid}")


def get_comments(eid):
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM comments WHERE engagement_id=? ORDER BY id", (eid,))]


def new_version(eid, note, user):
    """Snapshot the current plan, bump the version and reopen it as Draft."""
    _need(user, "edit")
    with connect() as c:
        e = _eng(c, eid)
        snap = json.dumps({"engagement": e, "rows": _rows(c, eid)})
        c.execute("INSERT INTO versions(engagement_id,version,snapshot,note,created_by,created_at) VALUES(?,?,?,?,?,?)",
                  (eid, e["version"], snap, note or "", user["username"], now()))
        c.execute("UPDATE engagements SET version=version+1, status='Draft', approved_by=NULL, approved_at=NULL, updated_at=? WHERE id=?", (now(), eid))
        _log(c, user, "new_version", f"{e['ref']} v{e['version'] + 1}")


def list_versions(eid):
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT id,version,note,created_by,created_at FROM versions WHERE engagement_id=? ORDER BY id DESC", (eid,))]


# ---------- library & templates ----------
def add_lib(data, user, verified=False):
    _need(user, "edit")
    d = {f: str(data.get(f) or "").strip() for f in LIB_FIELDS}
    if not d["title"] or d["kind"] not in LIB_KINDS:
        raise ValueError("Kind and title are required.")
    with connect() as c:
        if c.execute("SELECT 1 FROM library WHERE kind=? AND title=? AND category=?", (d["kind"], d["title"], d["category"])).fetchone():
            raise ValueError("A library item with this kind, title and category already exists.")
        c.execute(f"INSERT INTO library({','.join(LIB_FIELDS)},verified,created_by,created_at) VALUES({','.join('?' * len(LIB_FIELDS))},?,?,?)",
                  [d[f] for f in LIB_FIELDS] + [int(verified), user["username"], now()])
        _log(c, user, "add_library", d["title"])


def update_lib(lid, data, user, verified):
    _need(user, "edit")
    with connect() as c:
        c.execute(f"UPDATE library SET {','.join(f + '=?' for f in LIB_FIELDS)}, verified=? WHERE id=?",
                  [str(data.get(f) or "").strip() for f in LIB_FIELDS] + [int(verified), lid])
        _log(c, user, "update_library", str(lid))


def delete_lib(lid, user):
    _need(user, "edit")
    with connect() as c:
        c.execute("DELETE FROM library WHERE id=?", (lid,))
        _log(c, user, "delete_library", str(lid))


def list_lib(search="", kind=None, category=None):
    q, p = "SELECT * FROM library WHERE 1=1", []
    if kind:
        q += " AND kind=?"; p.append(kind)
    if category:
        q += " AND category=?"; p.append(category)
    if search:
        q += " AND (title LIKE ? OR content LIKE ? OR source LIKE ?)"; p += [f"%{search}%"] * 3
    with connect() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY kind, title", p)]


def save_template(name, eid, user):
    _need(user, "edit")
    with connect() as c:
        e = _eng(c, eid)
        if e["status"] != "Approved":
            raise ValueError("Only approved plans can be saved as reusable templates.")
        if not name.strip():
            raise ValueError("Template name required.")
        try:
            c.execute("INSERT INTO saved_templates(name,category,rows,created_by,created_at) VALUES(?,?,?,?,?)",
                      (name.strip(), e["category"], json.dumps(_rows(c, eid)), user["username"], now()))
        except sqlite3.IntegrityError:
            raise ValueError("A template with this name already exists.")
        _log(c, user, "save_template", name)


def list_templates():
    with connect() as c:
        return [{**dict(r), "rows": json.loads(r["rows"])} for r in c.execute("SELECT * FROM saved_templates ORDER BY name")]


# ---------- evidence ----------
def add_evidence(eid, filename, data, user):
    _need(user, "edit")
    folder = EVIDENCE_DIR / str(eid)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex[:8]}_{Path(filename).name}"
    path.write_bytes(data)
    with connect() as c:
        c.execute("INSERT INTO evidence(engagement_id,filename,path,uploaded_by,uploaded_at) VALUES(?,?,?,?,?)",
                  (eid, Path(filename).name, str(path), user["username"], now()))
        _log(c, user, "upload_evidence", filename)


def list_evidence(eid):
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM evidence WHERE engagement_id=? ORDER BY id", (eid,))]


# ---------- stats, logs, backup ----------
def stats():
    with connect() as c:
        q = lambda s: {(r[0] or "Unspecified"): r[1] for r in c.execute(s)}
        return {"total": c.execute("SELECT COUNT(*) FROM engagements").fetchone()[0],
                "status": q("SELECT status,COUNT(*) FROM engagements GROUP BY status"),
                "category": q("SELECT category,COUNT(*) FROM engagements GROUP BY category"),
                "risk": q("SELECT risk_level,COUNT(*) FROM plan_rows GROUP BY risk_level"),
                "rows": q("SELECT status,COUNT(*) FROM plan_rows GROUP BY status")}


def get_log(limit=200):
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT ts,user,action,detail FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]


def remove_demo(user):
    _need(user, "admin")
    with connect() as c:
        c.execute("DELETE FROM engagements WHERE is_demo=1")
        _log(c, user, "remove_demo")


def backup_db(tag="manual"):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    p = BACKUP_DIR / f"awp_{datetime.now():%Y%m%d_%H%M%S}_{tag}.db"
    src, dst = sqlite3.connect(DB_PATH), sqlite3.connect(p)
    src.backup(dst); dst.close(); src.close()
    prune_backups()
    return p


def prune_backups(keep=None):
    """Keep only the newest `keep` backups (default AEP_BACKUP_KEEP) so the volume does not fill up."""
    keep = BACKUP_KEEP if keep is None else keep
    files = sorted(BACKUP_DIR.glob("awp_*.db"), key=lambda f: f.name)
    for f in files[:max(0, len(files) - keep)]:
        f.unlink(missing_ok=True)


def auto_backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not list(BACKUP_DIR.glob(f"awp_{datetime.now():%Y%m%d}_*_auto.db")):
        backup_db("auto")


def restore_db(data, user):
    _need(user, "admin")
    tmp = DATA_DIR / "restore_tmp.db"
    tmp.write_bytes(data)
    try:
        t = sqlite3.connect(tmp)
        names = {r[0] for r in t.execute("SELECT name FROM sqlite_master")}
        t.close()
    except sqlite3.DatabaseError:
        names = set()
    if not {"engagements", "plan_rows", "users"} <= names:
        tmp.unlink(missing_ok=True)
        raise ValueError("Not a valid AWP database file.")
    backup_db("pre-restore")
    shutil.copyfile(tmp, DB_PATH)
    tmp.unlink()
    init_db()
