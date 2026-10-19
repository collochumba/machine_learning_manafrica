"""Module G analytics: configurable transport tests. Results are EXCEPTIONS FOR INVESTIGATION, never findings.
Missing or invalid inputs are reported (status 'Insufficient data' / 'Invalid input'), never silently treated as 0."""
from datetime import date, datetime

INSUFFICIENT, INVALID, OK, FLAG = "Insufficient data", "Invalid input", "Within range", "Flag for investigation"


def _num(v):
    """Return float or None (blank/non-numeric). Booleans and NaN are rejected."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(str(v).replace(",", "").strip())
    except ValueError:
        return None
    return None if f != f else f


def fuel_efficiency(km, litres):
    """km travelled / litres consumed. Returns dict(value, status, note)."""
    k, l = _num(km), _num(litres)
    if k is None or l is None:
        return {"value": None, "status": INSUFFICIENT, "note": "Kilometres and litres are both required."}
    if k < 0 or l < 0:
        return {"value": None, "status": INVALID, "note": "Values cannot be negative."}
    if l == 0:
        return {"value": None, "status": INVALID, "note": "Litres is zero; efficiency cannot be computed."}
    return {"value": round(k / l, 2), "status": OK, "note": ""}


def fleet_utilisation(active, serviceable):
    """vehicles actively utilised / total serviceable vehicles x 100."""
    a, s = _num(active), _num(serviceable)
    if a is None or s is None:
        return {"value": None, "status": INSUFFICIENT, "note": "Active and serviceable vehicle counts are required."}
    if a < 0 or s < 0 or a != int(a) or s != int(s):
        return {"value": None, "status": INVALID, "note": "Counts must be whole, non-negative numbers."}
    if s == 0:
        return {"value": None, "status": INVALID, "note": "No serviceable vehicles; utilisation cannot be computed."}
    if a > s:
        return {"value": None, "status": INVALID, "note": "Active vehicles exceed serviceable vehicles - check the counts."}
    return {"value": round(a / s * 100, 1), "status": OK, "note": ""}


def _d(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v).strip()[:10])
    except ValueError:
        return None


def maintenance_cost(records, period_start, period_end):
    """Total maintenance expenditure per vehicle for a clearly identified review period.
    records: iterable of dicts {vehicle, date, amount}. Returns dict(period, per_vehicle, total, rejected)."""
    s, e = _d(period_start), _d(period_end)
    if not s or not e:
        raise ValueError("A review period (start and end date) is required.")
    if e < s:
        raise ValueError("Review period end is before its start.")
    per, rejected = {}, []
    for i, r in enumerate(records, 1):
        veh, dt, amt = str(r.get("vehicle") or "").strip(), _d(r.get("date")), _num(r.get("amount"))
        if not veh or dt is None or amt is None or amt < 0:
            rejected.append({"row": i, "reason": "Missing/invalid vehicle, date or amount", "record": r}); continue
        if s <= dt <= e:
            per[veh] = round(per.get(veh, 0) + amt, 2)
    return {"period": f"{s.isoformat()} to {e.isoformat()}", "per_vehicle": per,
            "total": round(sum(per.values()), 2), "rejected": rejected}


def analyse_fuel(rows, expected_kmpl=None, tolerance_pct=20.0):
    """Per-row efficiency with optional expected km/l and tolerance. Flags rows for INVESTIGATION only."""
    exp, out = _num(expected_kmpl), []
    for r in rows:
        res = fuel_efficiency(r.get("km"), r.get("litres"))
        flag = res["status"]
        if res["value"] is not None and exp:
            dev = (res["value"] - exp) / exp * 100
            if abs(dev) > tolerance_pct:
                flag = FLAG
                res["note"] = f"{dev:+.1f}% against expected {exp:g} km/l (tolerance +/-{tolerance_pct:g}%) - obtain management explanation."
        out.append({**r, "km_per_litre": res["value"], "status": flag, "note": res["note"]})
    return out


def flag_multiple_fuel(rows, max_per_day=1):
    """Flag vehicles with more than `max_per_day` fuel transactions on one day (suspicious pattern to investigate)."""
    seen, out = {}, []
    for r in rows:
        key = (str(r.get("vehicle") or "").strip(), _d(r.get("date")))
        if key[0] and key[1]:
            seen.setdefault(key, []).append(r)
    for (veh, dt), grp in seen.items():
        if len(grp) > max_per_day:
            out.append({"vehicle": veh, "date": dt.isoformat(), "transactions": len(grp),
                        "status": FLAG, "note": f"{len(grp)} transactions on one day (threshold {max_per_day}) - investigate."})
    return out


def build_analysis_workbook():
    """Editable Excel tool with live formulas; inputs are blank (no invented data)."""
    from io import BytesIO
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    import theme as th
    wb = Workbook()
    hdr = lambda ws, vals: [ws.append(vals), [setattr(c, "font", Font(bold=True, color="FFFFFF")) or setattr(c, "fill", PatternFill("solid", fgColor=th.GREEN.lstrip("#"))) for c in ws[1]]]
    f = wb.active; f.title = "Fuel efficiency"
    hdr(f, ["Vehicle reg.", "Period", "Km travelled", "Litres consumed", "Km per litre", "Expected km/l (enter)", "Tolerance % (enter)", "Flag for investigation"])
    for r in range(2, 202):
        f[f"E{r}"] = f'=IF(OR(C{r}="",D{r}="",NOT(ISNUMBER(C{r})),NOT(ISNUMBER(D{r}))),"Insufficient data",IF(D{r}<=0,"Invalid input",ROUND(C{r}/D{r},2)))'
        f[f"H{r}"] = f'=IF(OR(NOT(ISNUMBER(E{r})),NOT(ISNUMBER(F{r})),NOT(ISNUMBER(G{r}))),"",IF(ABS(E{r}-F{r})/F{r}*100>G{r},"Flag for investigation","Within range"))'
    u = wb.create_sheet("Fleet utilisation")
    hdr(u, ["Review period (enter)", "Vehicles actively utilised", "Total serviceable vehicles", "Utilisation %"])
    u["D2"] = '=IF(OR(NOT(ISNUMBER(B2)),NOT(ISNUMBER(C2))),"Insufficient data",IF(OR(C2<=0,B2>C2,B2<0),"Invalid input",ROUND(B2/C2*100,1)))'
    m = wb.create_sheet("Maintenance cost")
    hdr(m, ["Vehicle reg.", "Date", "Amount", "In review period?", "", "Review start (enter)", "Review end (enter)"])
    for r in range(2, 502):
        m[f"D{r}"] = f'=IF(OR(A{r}="",C{r}="",B{r}=""),"",IF(AND(B{r}>=$F$2,B{r}<=$G$2),"Yes","No"))'
    m["I1"], m["J1"] = "Vehicle reg.", "Total in period"
    m["I2"] = "(list vehicle registrations in column I)"
    for r in range(3, 103):
        m[f"J{r}"] = f'=IF(I{r}="","",SUMIFS($C$2:$C$501,$A$2:$A$501,I{r},$D$2:$D$501,"Yes"))'
    n = wb.create_sheet("Notes")
    n.append(["Results flag items for investigation only. They are not audit findings until the auditor has obtained and evaluated management's explanation and evidence."])
    for ws in wb: ws.sheet_properties.tabColor = th.GREEN.lstrip("#")
    b = BytesIO(); wb.save(b); return b.getvalue()
