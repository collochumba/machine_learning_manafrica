"""Central KSG theme: palette, risk/status styling, CSS, logo handling and small HTML helpers.

All colours used by the UI and the Excel/Word exports are defined here. The same values are mirrored in
.streamlit/config.toml, which styles Streamlit's native widgets (inputs, alerts, grid headers).
"""
import html
import os
from pathlib import Path
from string import Template

# ---- Palette -------------------------------------------------------------------------------------------------
GREEN = "#006B3F"          # primary KSG green
DARK_GREEN = "#004D2C"
GOLD = "#D4AF37"
MAROON = "#6B1F2B"
WHITE = "#FFFFFF"
BG = "#F7F8F5"             # light application background
TEXT = "#26332D"
LIGHT_GREEN = "#E8F3ED"
LIGHT_GOLD = "#F7F0D0"
LIGHT_MAROON = "#F4E7EA"
BORDER = "#D8E1DB"
# Derived shades (used only where the base palette would not give readable contrast)
DARK_MAROON = "#4A1019"    # CRITICAL risk
GOLD_TEXT = "#6B5500"      # dark gold for text on light-gold backgrounds
MUTED = "#5B6760"          # secondary text / neutral labels
NEUTRAL_BG = "#EEF1EE"

ORG_NAME = os.getenv("ORG_NAME", "Kenya School of Government")
ORG_HTML = html.escape(ORG_NAME.upper())  # safe for HTML injection
SYSTEM_NAME = "AUDIT ENGAGEMENT PROGRAMS"
SYSTEM_FULL_NAME = "Audit Working Plan Management System"

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "ksg_logo.png"

# ---- Risk and status styling: key -> (background, text colour, border) ----------------------------------------
RISK_STYLES = {
    "low": (LIGHT_GREEN, GREEN, GREEN),
    "medium": (LIGHT_GOLD, GOLD_TEXT, GOLD),
    "high": (LIGHT_MAROON, MAROON, MAROON),
    "critical": (DARK_MAROON, WHITE, DARK_MAROON),
    "not assessed": (NEUTRAL_BG, MUTED, BORDER),
}
STATUS_STYLES = {
    "approved": (GREEN, WHITE, GREEN),
    "completed": (GREEN, WHITE, GREEN),
    "in progress": (LIGHT_GOLD, GOLD_TEXT, GOLD),
    "pending review": (LIGHT_GOLD, GOLD_TEXT, GOLD),
    "awaiting review": (LIGHT_GOLD, GOLD_TEXT, GOLD),
    "returned": (LIGHT_MAROON, MAROON, MAROON),
    "rejected": (LIGHT_MAROON, MAROON, MAROON),
    "draft": (NEUTRAL_BG, MUTED, BORDER),
    "not started": (NEUTRAL_BG, MUTED, BORDER),
}
_NEUTRAL = (NEUTRAL_BG, MUTED, BORDER)

# Chart colour maps (same palette, same meaning as the badges)
RISK_CHART = {"Low": GREEN, "Medium": GOLD, "High": MAROON, "Critical": DARK_MAROON, "Not assessed": "#9AA59E"}
STATUS_CHART = {"Approved": GREEN, "Completed": GREEN, "In Progress": GOLD, "Awaiting Review": "#E6CF7A",
                "Returned": MAROON, "Draft": "#9AA59E", "Not Started": "#C5CEC8"}
CHART_COLORWAY = [GREEN, GOLD, MAROON, DARK_GREEN, "#9AA59E"]


def risk_style(value):
    return RISK_STYLES.get(str(value or "").strip().lower(), _NEUTRAL)


def status_style(value):
    return STATUS_STYLES.get(str(value or "").strip().lower(), _NEUTRAL)


def badge(value, kind="status"):
    """HTML pill for a risk level or status. The text is escaped: never pass pre-built HTML in."""
    if not str(value or "").strip():
        return ""
    bg, fg, bd = (risk_style if kind == "risk" else status_style)(value)
    return (f'<span class="ksg-badge" style="background:{bg};color:{fg};border:1px solid {bd}">'
            f'{html.escape(str(value))}</span>')


def style_status_risk(df, status_cols=(), risk_cols=()):
    """Return a pandas Styler colouring status/risk columns of a read-only table (for st.dataframe)."""
    def _mk(fn):
        def f(v):
            bg, fg, _ = fn(v) if str(v or "").strip() else (None, None, None)
            return f"background-color:{bg};color:{fg};font-weight:600" if bg else ""
        return f
    sty = df.style
    mapper = getattr(sty, "map", None) or sty.applymap  # pandas >= 2.1 has Styler.map
    for cols, fn in ((status_cols, status_style), (risk_cols, risk_style)):
        cols = [c for c in cols if c in df.columns]
        if cols:
            sty = mapper(_mk(fn), subset=cols); mapper = getattr(sty, "map", None) or sty.applymap
    return sty


def style_fig(fig, height=330):
    fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=TEXT), colorway=CHART_COLORWAY,
                      title_font=dict(color=DARK_GREEN, size=15), height=height, margin=dict(l=20, r=20, t=50, b=20),
                      showlegend=fig.layout.showlegend)
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER)
    return fig


# ---- Logo ----------------------------------------------------------------------------------------------------
def logo_available():
    return LOGO_PATH.is_file()


def show_logo(container, width=110):
    """Render assets/ksg_logo.png into a Streamlit container. Returns False (and shows nothing) if missing/unreadable."""
    if not logo_available():
        return False
    try:
        container.image(str(LOGO_PATH), width=width)
        return True
    except Exception:  # unreadable image must never stop the app
        return False


def page_icon():
    return str(LOGO_PATH) if logo_available() else "📋"


# ---- HTML components -----------------------------------------------------------------------------------------
def kpi_card(label, value, accent="green"):
    return (f'<div class="ksg-kpi ksg-kpi-{accent}"><div class="ksg-kpi-label">{html.escape(str(label))}</div>'
            f'<div class="ksg-kpi-value">{html.escape(str(value))}</div></div>')


def page_header(title, subtitle=None):
    sub = f'<div class="ksg-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="ksg-page-title"><h2>{html.escape(title)}</h2>{sub}<div class="ksg-gold-bar"></div></div>'


def html_table(rows, columns, risk_cols=("risk_level",), status_cols=("status",)):
    """Read-only, wrapped table with a green header. `columns` = [(key, label)]. All cell text is escaped."""
    head = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl in columns)
    body = []
    for r in rows:
        cells = []
        for k, _ in columns:
            v = r.get(k, "")
            if v is None or v == "": v = (r.get("extra") or {}).get(k, "") if isinstance(r.get("extra"), dict) else ""
            if k in risk_cols: cells.append(f"<td>{badge(v, 'risk')}</td>")
            elif k in status_cols: cells.append(f"<td>{badge(v, 'status')}</td>")
            else: cells.append(f"<td>{html.escape(str(v or ''))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="ksg-table-wrap"><table class="ksg-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# ---- CSS -----------------------------------------------------------------------------------------------------
_CSS = Template("""<style>
.stApp{background:$BG;color:$TEXT}
h1,h2,h3,h4{color:$DARK_GREEN !important}
.block-container{padding-top:2rem;max-width:1400px}
header[data-testid=stHeader]{background:transparent}

/* Buttons: primary = KSG green, secondary = outlined green */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{border-radius:6px;font-weight:600;border:1px solid $GREEN;color:$GREEN;background:$WHITE}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{border-color:$DARK_GREEN;color:$DARK_GREEN;background:$LIGHT_GREEN}
.stButton>button[kind=primary],.stDownloadButton>button[kind=primary],.stFormSubmitButton>button[kind=primary]{background:$GREEN;color:$WHITE;border:1px solid $GREEN}
.stButton>button[kind=primary]:hover,.stDownloadButton>button[kind=primary]:hover,.stFormSubmitButton>button[kind=primary]:hover{background:$DARK_GREEN;color:$WHITE;border-color:$DARK_GREEN}
.stButton>button:disabled,.stDownloadButton>button:disabled,.stFormSubmitButton>button:disabled{opacity:.5}
.stButton>button:focus-visible,.stFormSubmitButton>button:focus-visible{outline:2px solid $GOLD;outline-offset:2px}

/* Forms and inputs */
[data-testid=stForm]{background:$WHITE;border:1px solid $BORDER;border-top:3px solid $GOLD;border-radius:8px;padding:1.2rem}
[data-testid=stTextInput] input,[data-testid=stTextArea] textarea,[data-testid=stNumberInput] input,[data-baseweb=select]>div{background:$WHITE !important;color:$TEXT !important;border-color:$BORDER !important}
[data-testid=stWidgetLabel] p,label p{color:$TEXT !important;font-weight:600}

/* Tabs */
[data-baseweb=tab-list]{border-bottom:1px solid $BORDER}
button[data-baseweb=tab]{color:$MUTED;font-weight:600}
button[data-baseweb=tab][aria-selected=true]{color:$GREEN}
[data-baseweb=tab-highlight]{background:$GOLD !important;height:3px}

/* Expanders */
[data-testid=stExpander]{background:$WHITE;border:1px solid $BORDER;border-radius:8px}
[data-testid=stExpander] summary{color:$DARK_GREEN;font-weight:600}

/* Native metrics (used for progress) */
[data-testid=stMetric]{background:$WHITE;padding:12px 14px;border-radius:8px;border:1px solid $BORDER;border-left:5px solid $GREEN}
[data-testid=stMetricValue]{color:$DARK_GREEN}
[data-testid=stProgress] [role=progressbar]>div{background:$GREEN}

/* Sidebar */
[data-testid=stSidebar]{background:$DARK_GREEN;border-right:3px solid $GOLD}
[data-testid=stSidebar] h1,[data-testid=stSidebar] h2,[data-testid=stSidebar] h3,[data-testid=stSidebar] p,[data-testid=stSidebar] span,[data-testid=stSidebar] label{color:$WHITE !important}
[data-testid=stSidebar] hr{border-color:rgba(255,255,255,.25);margin:.6rem 0}
.ksg-side-org{color:$WHITE;font-weight:700;letter-spacing:.06em;font-size:.82rem;line-height:1.25;margin:.2rem 0 .1rem}
.ksg-side-sys{color:$GOLD;font-size:.72rem;margin-bottom:.2rem}
.ksg-side-user{color:$WHITE;font-weight:600;font-size:.95rem}
.ksg-side-role{color:$GOLD;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}
[data-testid=stSidebar] [role=radiogroup]{gap:2px}
[data-testid=stSidebar] [role=radiogroup] label{padding:.5rem .7rem;border-radius:6px;border-left:4px solid transparent;width:100%;cursor:pointer}
[data-testid=stSidebar] [role=radiogroup] label:hover{background:rgba(255,255,255,.10)}
[data-testid=stSidebar] [role=radiogroup] label>div:first-child{display:none}
[data-testid=stSidebar] [role=radiogroup] label:has(input:checked){background:rgba(212,175,55,.18);border-left-color:$GOLD}
[data-testid=stSidebar] [role=radiogroup] label:has(input:checked) p{color:$GOLD !important;font-weight:700}
[data-testid=stSidebar] .stButton>button{background:transparent;color:$WHITE;border:1px solid $GOLD;width:100%}
[data-testid=stSidebar] .stButton>button:hover{background:$GOLD;color:$DARK_GREEN;border-color:$GOLD}

/* Page header, banner */
.ksg-page-title h2{margin:0 0 .15rem 0;padding:0;font-size:1.7rem}
.ksg-sub{color:$MUTED;font-size:.92rem}
.ksg-gold-bar{height:3px;width:64px;background:$GOLD;margin:.45rem 0 1rem}
.ksg-banner-org{color:$DARK_GREEN;font-weight:800;letter-spacing:.07em;font-size:1.05rem;margin-top:.2rem}
.ksg-banner-sys{color:$MUTED;font-size:.9rem}

/* KPI cards */
.ksg-kpi{background:$WHITE;border:1px solid $BORDER;border-left:6px solid $GREEN;border-radius:8px;padding:.9rem 1.1rem;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.ksg-kpi-label{color:$MUTED;font-size:.82rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.ksg-kpi-value{color:$DARK_GREEN;font-size:2rem;font-weight:800;line-height:1.2}
.ksg-kpi-green{border-left-color:$GREEN}
.ksg-kpi-gold{border-left-color:$GOLD}
.ksg-kpi-maroon{border-left-color:$MAROON}
.ksg-kpi-maroon .ksg-kpi-value{color:$MAROON}

/* Badges */
.ksg-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.78rem;font-weight:700;white-space:nowrap}

/* Formatted read-only working-plan table */
.ksg-table-wrap{overflow:auto;max-height:560px;border:1px solid $BORDER;border-radius:8px;background:$WHITE}
.ksg-table{border-collapse:separate;border-spacing:0;width:100%;font-size:.84rem;color:$TEXT}
.ksg-table th{position:sticky;top:0;background:$GREEN;color:$WHITE;text-align:left;padding:8px 10px;border-bottom:3px solid $GOLD;border-right:1px solid $DARK_GREEN;min-width:120px;z-index:1}
.ksg-table td{padding:8px 10px;border-bottom:1px solid $BORDER;border-right:1px solid $BORDER;vertical-align:top;white-space:pre-wrap;word-break:break-word;min-width:120px;max-width:340px;background:$WHITE}
.ksg-table tbody tr:nth-child(even) td{background:$BG}
.ksg-table td:first-child,.ksg-table th:first-child{min-width:90px}

/* Login */
.ksg-login-head{text-align:center;margin:1.5rem 0 1rem}
.ksg-login-org{color:$DARK_GREEN;font-weight:800;letter-spacing:.08em;font-size:1.35rem;margin-top:.4rem}
.ksg-login-sys{color:$TEXT;font-size:1.05rem;margin-top:.15rem}
.ksg-login-head .ksg-gold-bar{margin:.7rem auto 0;width:90px}
.ksg-footer{text-align:center;color:$MUTED;font-size:.78rem;margin-top:1rem}
</style>""")


def inject_css(st):
    st.markdown(_CSS.substitute(
        GREEN=GREEN, DARK_GREEN=DARK_GREEN, GOLD=GOLD, MAROON=MAROON, WHITE=WHITE, BG=BG, TEXT=TEXT,
        LIGHT_GREEN=LIGHT_GREEN, BORDER=BORDER, MUTED=MUTED), unsafe_allow_html=True)
