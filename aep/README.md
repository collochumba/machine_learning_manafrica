# KSG Audit Engagement Programs
Streamlit + SQLite. A focused module for creating, viewing, editing, exporting (Word / Excel / PDF / HTML) and safely deleting Audit Engagement Programs
(columns: No. | Audit Area | Requirements | Procedure / Test | Control). Optional automatic upload of exports to a Google Drive folder, with a manual-upload fallback.
**Read DEPLOYMENT.md** for persistent storage, Google Drive setup, secrets, upgrading, export/deletion procedures and remaining actions.

## Run locally (Windows PowerShell)
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```
On first run create the Administrator in the setup screen (there is no default password).

## Module map
`app.py` login/setup/sidebar | `aep_ui.py` screens | `aep.py` data layer (optimistic locking, delete + manifest) | `aep_import.py` paste/upload parser |
`aep_export.py` Word/Excel/PDF/HTML builders | `aep_exports.py` export records, retry-safe upload queue, Drive deletion | `drive.py` Google Drive client |
`database.py` SQLite connection, users, backups | `migrations.py` versions 1-9 | `drive_authorize.py` one-time OAuth helper | `tests/`.
Older AWP / working-paper / follow-up / transport modules (`ui_pages.py`, `working_papers.py`, `followup.py`, ...) are **not reachable from the UI**. They are kept only because
existing databases and migrations 1-5 reference their tables; they can be removed in a later clean-up once KSG confirms no data in them is needed.

## Tests
`python -m pytest tests -q`. Google Drive is always mocked in tests (no credentials, no network). `tests/test_ui_smoke.py` runs the real `app.py` headlessly with
Streamlit's AppTest; it is not a real browser.

## Manual browser check (not automated)
1. `streamlit run app.py`; create the Administrator; sign in.  2. New program > paste a table with the five headings > preview > save.
3. Open it in two browser windows, save in one, then save in the other: the second must show the "changed by another user" message and not overwrite.
4. Export each of the four formats, open each file.  5. With Drive unconfigured, confirm the "NOT available" banner and manual steps.  6. Delete a test program: both confirmations required, Cancel leaves it intact.
7. With Drive configured (test folder first): export, check the file appears; delete the program, choose Keep, then repeat and choose Delete from Drive.
