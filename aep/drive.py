"""Google Drive integration (official Drive API v3). Isolated behind a tiny interface so the AEP module never depends on it:
if Drive is not configured, configured_client() returns None and the app falls back to manual upload.

Drive stores EXPORTED DOCUMENTS ONLY. The live SQLite database never goes to Drive.

Configuration (environment variables / secrets manager - never in source control):
  AEP_DRIVE_FOLDER_ID               destination folder id (default: the KSG AEP folder below)
  AEP_DRIVE_AUTH                    "oauth" (default) or "service_account"; "none" disables Drive
  AEP_DRIVE_OAUTH_TOKEN_JSON | _FILE      authorised-user token created once with drive_authorize.py
  AEP_DRIVE_SERVICE_ACCOUNT_JSON | _FILE  service-account key (needs a Shared Drive or a Workspace setup, see DEPLOYMENT.md)
  AEP_DRIVE_SCOPE                   default https://www.googleapis.com/auth/drive.file (least privilege)

Error messages returned to users are fixed strings: they never contain tokens, keys or raw API responses."""
import io
import json
import logging
import os

log = logging.getLogger("aep.drive")

DEFAULT_FOLDER_ID = "1XgJneSIIQSKUgoWd_v2izaoWt4mFnOP-"
DEFAULT_SCOPE = "https://www.googleapis.com/auth/drive.file"

MESSAGES = {
    "not_configured": "Google Drive upload is not configured.",
    "auth": "Google Drive sign-in failed or expired. The Drive credentials must be renewed by the account owner.",
    "permission": "The Drive account does not have permission for this folder or file.",
    "quota": "Google Drive refused the request (storage quota or rate limit). Try again later.",
    "not_found": "The Drive folder or file was not found (it may have been moved or deleted, or is not accessible).",
    "network": "Could not reach Google Drive (network problem). Try again.",
    "other": "Google Drive returned an unexpected error.",
}


class DriveError(Exception):
    """kind is one of MESSAGES; str(e) is always safe to show to users."""

    def __init__(self, kind, detail=""):
        self.kind = kind if kind in MESSAGES else "other"
        self.detail = detail  # short, non-sensitive diagnostic (status/reason), for logs only
        super().__init__(MESSAGES[self.kind])


def classify(exc):
    """Map any exception from the Google client libraries to a DriveError, logging only status/reason/type."""
    if isinstance(exc, DriveError):
        return exc
    status = getattr(getattr(exc, "resp", None), "status", None)
    reason = ""
    try:
        reason = (json.loads(exc.content.decode("utf-8", "ignore"))["error"]["errors"][0]["reason"]) if hasattr(exc, "content") else ""
    except Exception:
        reason = ""
    name = type(exc).__name__
    if name == "RefreshError" or status == 401:
        kind = "auth"
    elif status == 403 and reason in ("storageQuotaExceeded", "userRateLimitExceeded", "rateLimitExceeded", "quotaExceeded", "dailyLimitExceeded"):
        kind = "quota"
    elif status == 429:
        kind = "quota"
    elif status == 403:
        kind = "permission"
    elif status == 404:
        kind = "not_found"
    elif isinstance(exc, (ConnectionError, TimeoutError, OSError)) or name in ("ServerNotFoundError", "TransportError", "ConnectionResetError"):
        kind = "network"
    elif status and status >= 500:
        kind = "network"
    else:
        kind = "other"
    log.warning("drive error kind=%s type=%s status=%s reason=%s", kind, name, status, reason)
    return DriveError(kind, f"{name} {status or ''} {reason}".strip())


class DriveClient:
    """Thin wrapper over a googleapiclient Resource. Tests substitute a fake with the same methods."""

    def __init__(self, service, folder_id):
        self.service, self.folder_id = service, folder_id

    def check(self):
        """Verify the destination folder is reachable. Returns the folder name; raises DriveError."""
        try:
            f = self.service.files().get(fileId=self.folder_id, fields="id,name,mimeType,trashed", supportsAllDrives=True).execute()
        except Exception as e:
            raise classify(e) from None
        if f.get("mimeType") != "application/vnd.google-apps.folder" or f.get("trashed"):
            raise DriveError("not_found")
        return f.get("name", "")

    def find_existing(self, uid, version, fmt):
        """Idempotency: a file already uploaded for this program/version/format (e.g. earlier attempt whose response was lost).
        uid is a hex string we generate; version an int; fmt one of a fixed set - nothing user-typed is placed in the query."""
        q = (f"'{self.folder_id}' in parents and trashed=false and appProperties has {{ key='aep_uid' and value='{uid}' }} "
             f"and appProperties has {{ key='aep_version' and value='{int(version)}' }} and appProperties has {{ key='fmt' and value='{fmt}' }}")
        try:
            r = self.service.files().list(q=q, fields="files(id,name)", pageSize=1, supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        except Exception as e:
            raise classify(e) from None
        return (r.get("files") or [None])[0]

    def list_by_uid(self, uid):
        """Files in the destination folder tagged with this program's uid (finds uploads whose reply was lost)."""
        q = f"'{self.folder_id}' in parents and trashed=false and appProperties has {{ key='aep_uid' and value='{uid}' }}"
        try:
            r = self.service.files().list(q=q, fields="files(id,name)", pageSize=100, supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        except Exception as e:
            raise classify(e) from None
        return r.get("files") or []

    def upload(self, name, mime, data, props):
        from googleapiclient.http import MediaIoBaseUpload
        try:
            media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
            f = self.service.files().create(body={"name": name, "parents": [self.folder_id], "appProperties": props},
                                            media_body=media, fields="id,name", supportsAllDrives=True).execute()
        except Exception as e:
            raise classify(e) from None
        if not f.get("id"):
            raise DriveError("other", "no id in response")  # never claim success without Drive confirming an id
        return f["id"]

    def get_meta(self, file_id):
        try:
            return self.service.files().get(fileId=file_id, fields="id,name,parents,trashed,appProperties", supportsAllDrives=True).execute()
        except Exception as e:
            raise classify(e) from None

    def delete(self, file_id):
        try:
            self.service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
        except Exception as e:
            raise classify(e) from None


def _secret(name):
    raw = os.getenv(name + "_JSON")
    if raw:
        return raw
    path = os.getenv(name + "_FILE")
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return ""


def folder_id():
    return os.getenv("AEP_DRIVE_FOLDER_ID") or DEFAULT_FOLDER_ID


def configured_client():
    """A ready DriveClient, or None when credentials are absent/invalid (-> manual-upload fallback). Never raises."""
    mode = (os.getenv("AEP_DRIVE_AUTH") or "oauth").lower()
    if mode == "none":
        return None
    scope = os.getenv("AEP_DRIVE_SCOPE") or DEFAULT_SCOPE
    try:
        if mode == "service_account":
            raw = _secret("AEP_DRIVE_SERVICE_ACCOUNT")
            if not raw:
                return None
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_info(json.loads(raw), scopes=[scope])
        else:
            raw = _secret("AEP_DRIVE_OAUTH_TOKEN")
            if not raw:
                return None
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_info(json.loads(raw), scopes=[scope])
            if creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
        from googleapiclient.discovery import build
        return DriveClient(build("drive", "v3", credentials=creds, cache_discovery=False), folder_id())
    except Exception as e:  # bad JSON, revoked token, missing libs: fall back, log type only
        log.warning("drive not available: %s", type(e).__name__)
        return None


def status(client=None):
    """(state, message) for the UI. state: 'ready' | 'manual' | 'error'. Verifies the folder when a client exists."""
    client = client if client is not None else configured_client()
    if client is None:
        return "manual", ("Automatic Google Drive upload is NOT available (credentials not configured). "
                          "Exports can still be downloaded; upload them to the Drive folder manually.")
    try:
        name = client.check()
        return "ready", f"Google Drive connected. Exports are uploaded to the folder '{name}'."
    except DriveError as e:
        return "error", f"Google Drive is configured but the folder check failed: {e}"
