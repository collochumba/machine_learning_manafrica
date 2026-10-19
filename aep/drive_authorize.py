"""ONE-TIME helper to authorise Google Drive on the account owner's computer.

Usage from the project root:
    python drive_authorize.py client_secret.json

The OAuth client must be of type "Desktop app". The requested scope is read
from AEP_DRIVE_SCOPE in .env (when python-dotenv is installed), or from the
environment. Store the printed token securely as AEP_DRIVE_OAUTH_TOKEN_JSON.
Never commit the client JSON or token to source control.
"""
import os
import sys

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

# Load local settings when running this helper directly from the project root.
# Existing shell environment variables take precedence over values in .env.
load_dotenv(override=False)

DEFAULT_SCOPE = "https://www.googleapis.com/auth/drive.file"
SCOPE = (os.getenv("AEP_DRIVE_SCOPE") or DEFAULT_SCOPE).strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python drive_authorize.py client_secret.json")

    client_file = sys.argv[1]
    if not os.path.isfile(client_file):
        sys.exit(f"OAuth client file not found: {client_file}")

    if not SCOPE.startswith("https://www.googleapis.com/auth/drive"):
        sys.exit("AEP_DRIVE_SCOPE is not a recognised Google Drive scope.")

    print(f"Starting Google Drive OAuth authorisation with scope: {SCOPE}")
    print("Keep this terminal open until the browser reports that authentication is complete.")

    creds = InstalledAppFlow.from_client_secrets_file(
        client_file,
        scopes=[SCOPE],
    ).run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
    )

    if not creds.refresh_token:
        sys.exit(
            "Google did not return a refresh token. Remove the app's access at "
            "https://myaccount.google.com/permissions and run this command again."
        )

    print("\nAuthentication succeeded.")
    print("Store the following complete JSON securely as AEP_DRIVE_OAUTH_TOKEN_JSON.")
    print("Treat it like a password. Do not commit it or share it.\n")
    print(creds.to_json())
