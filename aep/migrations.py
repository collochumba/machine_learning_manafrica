"""Versioned, additive SQLite migrations (tracked with PRAGMA user_version).

Rules: new tables only (existing tables are never altered or dropped); a backup is taken before any
pending migration runs; each migration runs in one transaction and is rolled back if it fails.
Parent links use ON DELETE RESTRICT so deleting an engagement or AWP can never silently remove
working papers or follow-up records."""
import sqlite3


class MigrationError(RuntimeError):
    pass


M1_REGISTER = """
CREATE TABLE audit_areas(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, description TEXT DEFAULT '',
  category TEXT DEFAULT '', status TEXT DEFAULT 'Active', is_candidate INTEGER DEFAULT 1,
  created_by TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE annual_engagements(id INTEGER PRIMARY KEY, financial_year TEXT NOT NULL, ref TEXT NOT NULL,
  title TEXT NOT NULL, objective TEXT DEFAULT '', scope TEXT DEFAULT '', risk_rating TEXT DEFAULT '',
  audit_type TEXT DEFAULT '', planned_start TEXT DEFAULT '', planned_end TEXT DEFAULT '',
  department TEXT DEFAULT '', team TEXT DEFAULT '', source_workbook TEXT DEFAULT '', source_sheet TEXT DEFAULT '',
  source_row INTEGER, is_test_data INTEGER DEFAULT 0, status TEXT DEFAULT 'Planned',
  created_by TEXT, created_at TEXT, updated_at TEXT, UNIQUE(financial_year, ref));
CREATE TABLE engagement_areas(annual_engagement_id INTEGER NOT NULL REFERENCES annual_engagements(id) ON DELETE RESTRICT,
  area_id INTEGER NOT NULL REFERENCES audit_areas(id) ON DELETE RESTRICT, PRIMARY KEY(annual_engagement_id, area_id));
CREATE TABLE library_area_links(library_id INTEGER NOT NULL REFERENCES library(id) ON DELETE CASCADE,
  area_id INTEGER NOT NULL REFERENCES audit_areas(id) ON DELETE RESTRICT, PRIMARY KEY(library_id, area_id));
CREATE TABLE library_meta(library_id INTEGER PRIMARY KEY REFERENCES library(id) ON DELETE CASCADE,
  verification_status TEXT DEFAULT 'Unverified', notes TEXT DEFAULT '', expected_result TEXT DEFAULT '');
"""

M2_TEMPLATES = """
CREATE TABLE awp_templates(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, family TEXT DEFAULT '',
  area_id INTEGER REFERENCES audit_areas(id) ON DELETE RESTRICT, status TEXT DEFAULT 'Active',
  source_status TEXT DEFAULT 'Auditor-defined', current_version INTEGER DEFAULT 1,
  created_by TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE awp_template_versions(id INTEGER PRIMARY KEY, template_id INTEGER NOT NULL REFERENCES awp_templates(id) ON DELETE RESTRICT,
  version INTEGER NOT NULL, content TEXT NOT NULL, change_note TEXT DEFAULT '', created_by TEXT, created_at TEXT,
  UNIQUE(template_id, version));
"""

M3_AWP_LINK = """
CREATE TABLE awp_source(awp_id INTEGER PRIMARY KEY REFERENCES engagements(id) ON DELETE RESTRICT,
  annual_engagement_id INTEGER REFERENCES annual_engagements(id) ON DELETE RESTRICT,
  template_version_id INTEGER REFERENCES awp_template_versions(id) ON DELETE RESTRICT,
  template_snapshot TEXT DEFAULT '', created_by TEXT, created_at TEXT);
"""

M4_WORKING_PAPERS = """
CREATE TABLE working_papers(id INTEGER PRIMARY KEY,
  annual_engagement_id INTEGER NOT NULL REFERENCES annual_engagements(id) ON DELETE RESTRICT,
  awp_id INTEGER REFERENCES engagements(id) ON DELETE RESTRICT, awp_wp_ref TEXT DEFAULT '',
  wp_ref TEXT NOT NULL, area TEXT DEFAULT '', audit_objective TEXT DEFAULT '', control_objective TEXT DEFAULT '',
  risk TEXT DEFAULT '', criteria TEXT DEFAULT '', procedure TEXT DEFAULT '', population TEXT DEFAULT '',
  sample_selection TEXT DEFAULT '', documents_required TEXT DEFAULT '', work_performed TEXT DEFAULT '',
  evidence_refs TEXT DEFAULT '', test_results TEXT DEFAULT '', exceptions TEXT DEFAULT '', analysis TEXT DEFAULT '',
  conclusion TEXT DEFAULT '', prepared_by TEXT DEFAULT '', prepared_date TEXT DEFAULT '', reviewer TEXT DEFAULT '',
  review_date TEXT DEFAULT '', reviewer_comments TEXT DEFAULT '', status TEXT DEFAULT 'Draft', version INTEGER DEFAULT 1,
  created_by TEXT, created_at TEXT, updated_at TEXT, UNIQUE(annual_engagement_id, wp_ref));
CREATE TABLE working_paper_samples(id INTEGER PRIMARY KEY, wp_id INTEGER NOT NULL REFERENCES working_papers(id) ON DELETE RESTRICT,
  item_ref TEXT DEFAULT '', description TEXT DEFAULT '', attributes TEXT DEFAULT '{}', evidence_ref TEXT DEFAULT '',
  result TEXT DEFAULT 'Not tested', exception_note TEXT DEFAULT '', position INTEGER DEFAULT 0);
CREATE TABLE working_paper_versions(id INTEGER PRIMARY KEY, wp_id INTEGER NOT NULL REFERENCES working_papers(id) ON DELETE RESTRICT,
  version INTEGER, snapshot TEXT, note TEXT, created_by TEXT, created_at TEXT);
"""

M5_FOLLOWUP = """
CREATE TABLE followups(id INTEGER PRIMARY KEY,
  annual_engagement_id INTEGER REFERENCES annual_engagements(id) ON DELETE RESTRICT,
  awp_id INTEGER REFERENCES engagements(id) ON DELETE RESTRICT, finding_ref TEXT DEFAULT '',
  original_recommendation TEXT NOT NULL, agreed_action TEXT DEFAULT '', action_owner TEXT DEFAULT '',
  original_due_date TEXT DEFAULT '', mgmt_status TEXT DEFAULT 'Not Started', mgmt_update_date TEXT DEFAULT '',
  mgmt_evidence TEXT DEFAULT '', auditor_procedures TEXT DEFAULT '', auditor_status TEXT DEFAULT 'Unable to Verify',
  followup_conclusion TEXT DEFAULT '', followup_date TEXT DEFAULT '', reviewer TEXT DEFAULT '',
  reviewer_comments TEXT DEFAULT '', review_status TEXT DEFAULT 'Draft', created_by TEXT, created_at TEXT, updated_at TEXT);
"""

M6_AEP = """
CREATE TABLE aep_programs(id INTEGER PRIMARY KEY, program_ref TEXT NOT NULL UNIQUE COLLATE NOCASE,
  title TEXT NOT NULL, annual_engagement_id INTEGER NOT NULL REFERENCES annual_engagements(id) ON DELETE RESTRICT,
  financial_year TEXT NOT NULL, campus TEXT DEFAULT '', audit_period TEXT DEFAULT '', main_audit_area TEXT DEFAULT '',
  status TEXT DEFAULT 'Draft', work_performed TEXT DEFAULT '', test_results TEXT DEFAULT '', findings TEXT DEFAULT '',
  prepared_by TEXT DEFAULT '', prepared_date TEXT DEFAULT '', reviewed_by TEXT DEFAULT '', reviewed_date TEXT DEFAULT '',
  created_by TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE aep_sections(id INTEGER PRIMARY KEY, program_id INTEGER NOT NULL REFERENCES aep_programs(id) ON DELETE RESTRICT,
  heading TEXT NOT NULL, position INTEGER DEFAULT 0);
CREATE TABLE aep_procedures(id INTEGER PRIMARY KEY, section_id INTEGER NOT NULL REFERENCES aep_sections(id) ON DELETE RESTRICT,
  position INTEGER DEFAULT 0, area TEXT DEFAULT '', requirements TEXT DEFAULT '', procedure TEXT DEFAULT '', control TEXT DEFAULT '');
CREATE INDEX idx_aep_prog_eng ON aep_programs(annual_engagement_id);
CREATE INDEX idx_aep_sec_prog ON aep_sections(program_id);
CREATE INDEX idx_aep_proc_sec ON aep_procedures(section_id);
"""

M7_AEP_STANDALONE = """
CREATE TABLE aep_p2(id INTEGER PRIMARY KEY, program_ref TEXT NOT NULL UNIQUE COLLATE NOCASE, title TEXT NOT NULL,
  annual_engagement_id INTEGER REFERENCES annual_engagements(id) ON DELETE SET NULL,
  financial_year TEXT NOT NULL, campus TEXT DEFAULT '', audit_period TEXT DEFAULT '', main_audit_area TEXT DEFAULT '',
  status TEXT DEFAULT 'Draft', work_performed TEXT DEFAULT '', test_results TEXT DEFAULT '', findings TEXT DEFAULT '',
  prepared_by TEXT DEFAULT '', prepared_date TEXT DEFAULT '', reviewed_by TEXT DEFAULT '', reviewed_date TEXT DEFAULT '',
  created_by TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE aep_s2(id INTEGER PRIMARY KEY, program_id INTEGER NOT NULL REFERENCES aep_p2(id) ON DELETE RESTRICT,
  heading TEXT NOT NULL, position INTEGER DEFAULT 0);
CREATE TABLE aep_r2(id INTEGER PRIMARY KEY, section_id INTEGER NOT NULL REFERENCES aep_s2(id) ON DELETE RESTRICT,
  position INTEGER DEFAULT 0, area TEXT DEFAULT '', requirements TEXT DEFAULT '', procedure TEXT DEFAULT '', control TEXT DEFAULT '');
INSERT INTO aep_p2 SELECT * FROM aep_programs;
INSERT INTO aep_s2 SELECT * FROM aep_sections;
INSERT INTO aep_r2 SELECT * FROM aep_procedures;
DROP TABLE aep_procedures; DROP TABLE aep_sections; DROP TABLE aep_programs;
ALTER TABLE aep_p2 RENAME TO aep_programs; ALTER TABLE aep_s2 RENAME TO aep_sections; ALTER TABLE aep_r2 RENAME TO aep_procedures;
UPDATE aep_programs SET status='In Progress' WHERE status='Pending Review';
CREATE INDEX idx_aep_sec_prog ON aep_sections(program_id);
CREATE INDEX idx_aep_proc_sec ON aep_procedures(section_id);
"""

M8_AEP_HARDEN = """
CREATE TABLE aep_p3(id INTEGER PRIMARY KEY, uid TEXT NOT NULL UNIQUE, program_ref TEXT NOT NULL UNIQUE COLLATE NOCASE,
  title TEXT NOT NULL, annual_engagement_id INTEGER REFERENCES annual_engagements(id) ON DELETE SET NULL,
  financial_year TEXT NOT NULL, campus TEXT DEFAULT '', audit_period TEXT DEFAULT '', main_audit_area TEXT DEFAULT '',
  status TEXT DEFAULT 'Draft', version INTEGER NOT NULL DEFAULT 1, created_by TEXT, created_at TEXT, updated_at TEXT,
  updated_by TEXT DEFAULT '');
CREATE TABLE aep_s3(id INTEGER PRIMARY KEY, program_id INTEGER NOT NULL REFERENCES aep_p3(id) ON DELETE CASCADE,
  heading TEXT NOT NULL, position INTEGER DEFAULT 0);
CREATE TABLE aep_r3(id INTEGER PRIMARY KEY, section_id INTEGER NOT NULL REFERENCES aep_s3(id) ON DELETE CASCADE,
  position INTEGER DEFAULT 0, area TEXT DEFAULT '', requirements TEXT DEFAULT '', procedure TEXT DEFAULT '', control TEXT DEFAULT '');
INSERT INTO aep_p3(id,uid,program_ref,title,annual_engagement_id,financial_year,campus,audit_period,main_audit_area,status,
  version,created_by,created_at,updated_at,updated_by)
  SELECT id, lower(hex(randomblob(16))), program_ref, title, annual_engagement_id, financial_year, campus, audit_period,
         main_audit_area, status, 1, created_by, created_at, updated_at, '' FROM aep_programs;
INSERT INTO aep_s3 SELECT id, program_id, heading, position FROM aep_sections;
INSERT INTO aep_r3 SELECT id, section_id, position, area, requirements, procedure, control FROM aep_procedures;
DROP TABLE aep_procedures; DROP TABLE aep_sections; DROP TABLE aep_programs;
ALTER TABLE aep_p3 RENAME TO aep_programs; ALTER TABLE aep_s3 RENAME TO aep_sections; ALTER TABLE aep_r3 RENAME TO aep_procedures;
CREATE INDEX idx_aep_sec_prog ON aep_sections(program_id);
CREATE INDEX idx_aep_proc_sec ON aep_procedures(section_id);
"""

M9_AEP_EXPORTS = """
CREATE TABLE aep_exports(id INTEGER PRIMARY KEY, program_uid TEXT NOT NULL, program_ref TEXT NOT NULL,
  program_version INTEGER, fmt TEXT NOT NULL, filename TEXT NOT NULL, mime TEXT NOT NULL, sha256 TEXT NOT NULL,
  size_bytes INTEGER, content BLOB, drive_status TEXT NOT NULL DEFAULT 'pending', drive_file_id TEXT,
  drive_error TEXT DEFAULT '', attempts INTEGER DEFAULT 0, created_by TEXT, created_at TEXT, uploaded_at TEXT);
CREATE INDEX idx_aep_exp_uid ON aep_exports(program_uid);
CREATE TABLE aep_deletions(id INTEGER PRIMARY KEY, program_uid TEXT NOT NULL, program_ref TEXT NOT NULL,
  program_title TEXT DEFAULT '', deleted_by TEXT, deleted_at TEXT, drive_decision TEXT NOT NULL DEFAULT 'none');
CREATE TABLE aep_deletion_files(id INTEGER PRIMARY KEY, deletion_id INTEGER NOT NULL REFERENCES aep_deletions(id) ON DELETE CASCADE,
  drive_file_id TEXT NOT NULL, filename TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'retained', detail TEXT DEFAULT '',
  updated_at TEXT);
"""

MIGRATIONS = [(1, "annual register, audit areas, library links", M1_REGISTER),
              (2, "AWP template library and versions", M2_TEMPLATES),
              (3, "AWP source links", M3_AWP_LINK),
              (4, "working papers", M4_WORKING_PAPERS),
              (5, "post-audit follow-up", M5_FOLLOWUP),
              (6, "audit engagement programs", M6_AEP),
              (7, "standalone AEP (engagement link optional)", M7_AEP_STANDALONE),
              (8, "AEP: version column, uid, cascade deletes, out-of-scope fields removed", M8_AEP_HARDEN),
              (9, "AEP exports, Drive upload tracking and deletion manifests", M9_AEP_EXPORTS)]
LATEST = MIGRATIONS[-1][0]


def current_version(path):
    c = sqlite3.connect(path)
    try:
        return c.execute("PRAGMA user_version").fetchone()[0]
    finally:
        c.close()


def pending(path):
    v = current_version(path)
    return [m for m in MIGRATIONS if m[0] > v]


def apply(path, backup, log=None):
    """Apply pending migrations. `backup()` must create a backup file and return its path."""
    todo = pending(path)
    if not todo:
        return []
    bpath = backup()  # a failed backup stops the migration before anything is touched
    done = []
    c = sqlite3.connect(path, isolation_level=None)
    try:
        c.execute("PRAGMA foreign_keys=ON")
        for ver, label, sql in todo:
            try:
                c.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version={ver};\nCOMMIT;")
            except sqlite3.Error as e:
                if c.in_transaction:
                    c.execute("ROLLBACK")
                raise MigrationError(f"Database upgrade {ver} ({label}) failed and was rolled back. "
                                     f"Your data is unchanged; a backup is at {bpath}. Detail: {type(e).__name__}") from None
            done.append(ver)
            if log:
                log(f"migration {ver}: {label}")
    finally:
        c.close()
    return done
