"""KSG Audit Engagement Programs - run: streamlit run app.py
Only the Audit Engagement Programs module is exposed. Older AWP/working-paper code stays in the package (dormant, unreachable from
the UI) because the database migrations and any existing data depend on it; see README."""
import html

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import aep_ui
import database as dbm
import theme as th

st.set_page_config(page_title=f"{th.ORG_NAME} - Audit Engagement Programs", page_icon=th.page_icon(), layout="wide")
th.inject_css(st)

PAGES = ["Audit Engagement Programs"]


@st.cache_resource
def _boot():
    dbm.init_db(); dbm.auto_backup(); return True


def _login_head():
    if th.logo_available():
        l1, l2, l3 = st.columns([1, 1, 1]); th.show_logo(l2, width=130)
    st.markdown(f'<div class="ksg-login-head"><div class="ksg-login-org">{th.ORG_HTML}</div>'
                f'<div class="ksg-login-sys">{th.SYSTEM_NAME}</div><div class="ksg-gold-bar"></div></div>', unsafe_allow_html=True)


def first_run_setup():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        _login_head()
        st.info("First-time setup: create the Administrator account. There is no default password. "
                f"Choose a unique password of at least {dbm.MIN_PASSWORD} characters.")
        with st.form("setup"):
            u = st.text_input("Administrator username")
            p1, p2 = st.text_input("Password", type="password"), st.text_input("Confirm password", type="password")
            if st.form_submit_button("Create administrator", type="primary", use_container_width=True):
                if p1 != p2:
                    st.error("The passwords do not match.")
                else:
                    try:
                        dbm.create_first_admin(u, p1); st.success("Administrator created. Sign in with it."); st.rerun()
                    except (ValueError, PermissionError) as e:
                        st.error(str(e))


def login():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        _login_head()
        with st.form("login"):
            u, p = st.text_input("Username"), st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                try:
                    user = dbm.authenticate(u, p)
                except PermissionError as e:
                    st.error(str(e)); user = "blocked"
                if user and user != "blocked":
                    st.session_state.user = user; st.rerun()
                elif user is None:
                    st.error("Invalid username or password.")
        st.markdown(f'<div class="ksg-footer">Internal use only - {html.escape(th.ORG_NAME)}</div>', unsafe_allow_html=True)


def force_password_change(user):
    st.warning("This account still uses the old shared default password. Choose a new password before continuing.")
    with st.form("chg"):
        p1, p2 = st.text_input("New password", type="password"), st.text_input("Confirm new password", type="password")
        if st.form_submit_button("Change password", type="primary"):
            if p1 != p2:
                st.error("The passwords do not match.")
            else:
                try:
                    dbm.change_own_password(user, p1); user["must_change"] = False; st.rerun()
                except ValueError as e:
                    st.error(str(e))


def admin_panel(user):
    with st.sidebar.expander("User accounts (Administrator)"):
        st.caption("Existing: " + ", ".join(f"{x['username']} ({x['role']})" for x in dbm.list_users()))
        with st.form("usr", clear_on_submit=True):
            u, p, r = st.text_input("Username"), st.text_input(f"Password ({dbm.MIN_PASSWORD}+ chars)", type="password"), st.selectbox("Role", dbm.ROLES)
            if st.form_submit_button("Create / update user"):
                try:
                    dbm.save_user(u, p, r, user); st.success("User saved.")
                except (ValueError, PermissionError) as e:
                    st.error(str(e))

    # Temporary, administrator-only safeguard before the database migration.
    with st.sidebar.expander("Database backup (Administrator)"):
        st.warning("Download and store this backup securely. It contains confidential audit and user data.")
        if st.button("Prepare online database backup", key="prepare_online_db_backup"):
            try:
                backup_path = dbm.backup_db("before-migration")
                st.session_state["online_db_backup_bytes"] = backup_path.read_bytes()
                st.session_state["online_db_backup_name"] = backup_path.name
                st.success("Backup prepared. Use the download button below.")
            except Exception as e:
                st.error(f"Backup could not be created ({type(e).__name__}). No migration was performed.")
        if st.session_state.get("online_db_backup_bytes") and st.session_state.get("online_db_backup_name"):
            st.download_button(
                "Download online database backup",
                data=st.session_state["online_db_backup_bytes"],
                file_name=st.session_state["online_db_backup_name"],
                mime="application/vnd.sqlite3",
                key="download_online_db_backup",
                use_container_width=True,
            )


try:
    _boot()
except Exception as e:  # e.g. migration failure: show the clear message, never a traceback
    st.error(f"The application could not start: {e}"); st.stop()
if dbm.needs_setup():
    first_run_setup(); st.stop()
if "user" not in st.session_state:
    login(); st.stop()
user = st.session_state.user
with st.sidebar:
    if th.logo_available(): l1, l2, l3 = st.columns([1, 2, 1]); th.show_logo(l2, width=90)
    st.markdown(f'<div class="ksg-side-org">{th.ORG_HTML}</div><div class="ksg-side-sys">{th.SYSTEM_NAME}</div>', unsafe_allow_html=True)
    st.divider()
    st.radio("Navigation", PAGES, key="nav_aep", label_visibility="collapsed")
    st.divider()
    st.markdown(f'<div class="ksg-side-user">{html.escape(user["username"])}</div><div class="ksg-side-role">{html.escape(user["role"])}</div>', unsafe_allow_html=True)
    if dbm.can(user["role"], "admin") and not user.get("must_change"): admin_panel(user)
    if st.button("Sign Out"): st.session_state.clear(); st.rerun()
if user.get("must_change"):
    force_password_change(user); st.stop()
aep_ui.page_aep(user)
