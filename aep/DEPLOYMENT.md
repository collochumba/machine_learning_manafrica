# KSG Audit Engagement Programs - deployment, Drive setup, secrets, export and deletion guide

Nothing in this guide has been executed against a live host, a real Google Cloud project or a real Drive folder. The code is tested with a
mocked Drive (see README, "Test results"). Steps marked **ACTION** need a person with the stated access.

## 1. Persistent SQLite storage
The live database is a single SQLite file whose path comes from `AEP_DB_PATH`. It must be on a disk that **survives restarts and redeploys**.
- Mount a persistent volume (e.g. `/var/data`) and set `AEP_DB_PATH=/var/data/aep.db`. Backups go to `/var/data/backups` automatically
  (override with `AEP_BACKUP_DIR`; `AEP_BACKUP_KEEP` newest are retained, default 30). Back up that volume with the host's own snapshots too.
- **Do not** put the database on Google Drive, in a temp folder, or on a host without persistent disk. **Streamlit Community Cloud has no
  persistent disk**: its files are lost on restart, so it is NOT suitable for production use of this application. Use a host that offers a
  persistent volume (a KSG-managed server/VM, or a PaaS with a mounted disk - check the provider's current plans, as disks are often a paid feature).
- Run **one** application instance. SQLite is a single-file database: it is appropriate for about five users making occasional edits. It is
  not designed for several independent app instances sharing one file, for network filesystems (use `AEP_SQLITE_JOURNAL=DELETE` if you must), or
  for sustained heavy concurrent writing. The application uses a 10 s busy timeout and version checks, so two people saving at once get a clear message
  rather than corrupted data - but this does not make SQLite a multi-writer, horizontally scalable database. If KSG later needs that, move to PostgreSQL.
- **ACTION (KSG ICT / host):** confirm KSG policy allows this hosting arrangement and the storage of audit documents there before go-live.

## 2. Upgrading an existing installation (data is preserved)
1. Stop the app. Copy the whole old `data/` folder somewhere safe.
2. Replace the code, keep your `data/` folder (or set `AEP_DB_PATH` to the existing `awp.db`).
3. Start the app. Migrations 8 and 9 run automatically **after a pre-migration backup** in `backups/`, inside a transaction (rolled back on failure).
   - Migration 8 rebuilds the three AEP tables: adds `uid`, `version`, `updated_by`; changes child links to cascade deletes; **drops the unused
     out-of-scope columns** (`work_performed`, `test_results`, `findings`, `prepared_by/date`, `reviewed_by/date`). Titles, references, financial
     years, campuses, audit periods, statuses, headings and every procedure row keep their ids and content. The dropped columns remain only in the backup.
   - Migration 9 adds export-tracking and deletion-manifest tables.
4. Existing accounts keep working. Any account still using the old shared password `ChangeMe!2024` is forced to choose a new password at next sign-in.
   Minimum password length is now 12.

## 3. First run and accounts
No default password exists. On first start with an empty user table the app shows **First-time setup** to create the Administrator.
(Optional: set `AWP_DEFAULT_PASSWORD` to 12+ characters to pre-seed `admin`, `auditor`, `reviewer`; avoid this - individual accounts are better.)
Administrators add individual accounts under *User accounts* in the sidebar. Roles: **Administrator** (everything, deletion of programs and Drive files),
**Auditor** (create/edit/export), **Reviewer** (view only for AEPs). Sign-in is throttled (5 failures / 15 min per account, per process).
**Remaining gap before institutional production:** there is no password-reset flow, no MFA/SSO, and the throttle is in memory. Put the app behind
HTTPS (reverse proxy or platform TLS) and, ideally, behind KSG's network/SSO.

## 4. Free Google Drive setup (OAuth, least privilege)
Exports (Word/Excel/PDF/HTML) are uploaded to folder `1XgJneSIIQSKUgoWd_v2izaoWt4mFnOP-` (`AEP_DRIVE_FOLDER_ID`). **The database never goes to Drive.**
Why OAuth and not a service account: a service account has no storage of its own and generally cannot create files in an ordinary "My Drive"
folder (Drive reports a storage-quota error); it works with a **Shared Drive** in a Google Workspace. If the folder is in a personal or standard
My Drive, use OAuth. If KSG has Workspace Shared Drives, a service account added as a member of that Shared Drive is the cleaner production choice
(`AEP_DRIVE_AUTH=service_account`).

**ACTION (Google account owner that owns/can edit the folder):**
1. Open https://console.cloud.google.com/ and create/select a project (no billing account should be needed for the Drive API; if the console demands
   one, stop - do not enable billing without approval).
2. Enable the API: https://console.cloud.google.com/apis/library/drive.googleapis.com
3. *APIs & Services > OAuth consent screen*: choose **Internal** if the account is in a Google Workspace organisation (no verification, tokens do not expire),
   otherwise **External**. For External apps left in "Testing", Google expires refresh tokens after 7 days; set the publishing status to **In production**
   (the default scope below is non-sensitive; confirm in the console whether Google requires verification for your account).
   A Workspace administrator may block third-party/OAuth apps - if so, ask KSG ICT to allow the client or use a Shared Drive + service account.
4. *Credentials > Create credentials > OAuth client ID > Desktop app*; download `client_secret.json`.
5. On your own computer: `pip install google-auth-oauthlib` then `python drive_authorize.py client_secret.json`; sign in and approve. Copy the printed JSON.
6. Store it as the host secret `AEP_DRIVE_OAUTH_TOKEN_JSON` (see section 5). Delete local copies of `client_secret.json` and the token.
7. Restart the app, open any program > Export. The banner says **Google Drive connected ... folder '<name>'** only when the folder check succeeds.

Scope: default `https://www.googleapis.com/auth/drive.file` (only files this app creates). **Known uncertainty:** with this narrow scope Drive may refuse to
see an *existing* folder the app did not create ("folder not found"). If the Export screen shows that error although the account can open the folder, the owner has
two choices: (a) set `AEP_DRIVE_SCOPE=https://www.googleapis.com/auth/drive` (broad: the app could then see all of that account's Drive - re-run
`drive_authorize.py`; only do this with the owner's explicit approval), or (b) create a new folder through the app's account and set `AEP_DRIVE_FOLDER_ID`.
The application never claims upload success unless Drive returns a file id.

Service account variant: create a service account (Credentials > Service account), create a JSON key, add the service-account e-mail as an
**Editor member of the Shared Drive/folder**, store the key as secret `AEP_DRIVE_SERVICE_ACCOUNT_JSON`, set `AEP_DRIVE_AUTH=service_account`.
If KSG policy forbids service-account keys, use OAuth.

**ACTION (KSG):** confirm audit documents may be stored in this Drive account. Do not place confidential KSG audit information in a personal Google
account without institutional authorisation.

## 5. Secrets configuration
Put these in the hosting platform's secrets/environment manager - never in the repository, chat, e-mail or logs:
`AEP_DRIVE_OAUTH_TOKEN_JSON` (or `AEP_DRIVE_SERVICE_ACCOUNT_JSON`), plus the plain settings in `.env.example`. A secret *file* mounted by the platform can be
referenced with `AEP_DRIVE_OAUTH_TOKEN_FILE` / `AEP_DRIVE_SERVICE_ACCOUNT_FILE`. `.gitignore` already excludes `.env`, key/token files, databases and backups.
The app logs only error type/status codes for Drive failures and shows fixed messages to users. Rotate by revoking the token at
https://myaccount.google.com/permissions and re-running `drive_authorize.py`.

## 6. Manual-upload fallback
If credentials are missing or invalid the app still creates, saves, exports (all four formats) and lets users download. The Export screen shows a warning that
automatic upload is **not** available, lists the steps (download, open the Drive folder, drag the file in), and records the export as "Download only (not in Drive)".
It is never marked uploaded. Enabling Drive later needs only the secrets above - no code change. Earlier download-only exports are not uploaded retroactively; export again.

## 7. Exporting safely
Open a program > **Export** > choose format > **Generate export**. Filenames: `AEP-YYYY-NNN_Title_vN_YYYYMMDD.ext`. Results: *Uploaded* (Drive confirmed),
*Upload failed* (download still available; **Retry upload** is safe and will not create duplicates - the same program version + format is uploaded once,
and Drive is checked for an earlier file whose reply was lost), or *Download only*. Failed uploads are kept in the database until they succeed.

## 8. Deleting safely (two separate steps, Administrator only)
**Step A - database:** *Delete* shows the reference/title, requires typing the reference and ticking a confirmation. It permanently removes the program,
its sections and procedure rows in one transaction (other programs untouched). Cancel changes nothing. This cannot be undone from the app.
**Step B - Google Drive (separate confirmation):** shown right after Step A (and later under "Deleted programs with Google Drive documents still in Drive").
Choose **Keep documents in Drive** (nothing is touched) or tick the box and type `DELETE FROM DRIVE` to permanently delete **only** the files recorded for that
program (and files in the folder tagged with that program's unique id). Each file is verified (in the configured folder, tagged with the deleted program's id)
before deletion; per-file results are stored; failures stay listed for retry. Drive deletion bypasses the trash. Files uploaded to Drive manually are not tracked
and are never touched. The manifest (`aep_deletions`, `aep_deletion_files`) holds only reference, title, timestamps, user, Drive file ids and outcomes.

## 9. Remaining actions (not done by this delivery)
| Who | Action |
|---|---|
| KSG ICT / management | Approve hosting and Drive storage of audit documents; decide on HTTPS, SSO/MFA, backup of the volume; allow the OAuth client if Workspace blocks it |
| Hosting provider / ICT | Provide a persistent volume, set `AEP_DB_PATH` and secrets, run exactly one instance |
| Google account owner | Steps in section 4 (project, API, consent screen, OAuth client, run `drive_authorize.py`, share/confirm access to the folder) |
| Application owner | Create the Administrator on first run; create individual accounts; run the manual browser check in README; confirm the official KSG logo may be used |
