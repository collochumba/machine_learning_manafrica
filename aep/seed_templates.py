"""Starter template families for Transport (Module G) and IT (Module H).

IMPORTANT - source status: the Transport Audit Engagement Program, the IT audit reference book and the Annual Audit
Plan were NOT available when this library was built. The structure below follows the topic lists in the project
brief only. Procedures are generic audit prompts, flagged 'Draft structure - source to be confirmed', and carry NO
source attribution. Replace/extend them from the source documents (edit the template -> saves a new version)."""
import template_lib as tl

PENDING = "Draft structure - source to be confirmed"
NOTE_T = "Transport Audit Engagement Program.docx - section to be mapped by the auditor (document not supplied to this build)"
NOTE_IT = "IT audit reference book - chapter/checklist to be mapped by the auditor (book not supplied to this build)"
CRIT_NOTE = ("Applicable KSG policy, circular, law or government standard NOT yet identified - auditor to supply and verify "
             "before treating any procedure as a compliance test.")

TRANSPORT_TOPICS = [
    ("Transport policies, procedures and administrative circulars", "Obtain the transport policies, procedures and circulars in force and assess whether they cover vehicle use, fuel, maintenance and disposal.", "Policies, procedures, circulars"),
    ("Fleet register completeness and physical verification", "Compare the fleet register with a physical listing of vehicles and trace a sample both ways.", "Fleet register, physical verification schedule"),
    ("Vehicle log books, ownership and registration details", "Examine log books and registration/ownership records for completeness and agreement to the fleet register.", "Log books, registration documents"),
    ("Insurance, inspection and statutory documentation", "Confirm that insurance, inspection and other statutory documents for sampled vehicles are current for the audit period.", "Insurance certificates, inspection certificates"),
    ("Work tickets and completeness of journey information", "Test sampled work tickets for completeness of journey details and sequential control.", "Work tickets"),
    ("Transport requisitions and journey authorisation", "Test sampled journeys for an approved transport requisition and authorisation before travel.", "Requisitions, authorisations"),
    ("Journey purpose, destination and mileage verification", "Verify purpose, destination and mileage on sampled journeys against supporting records.", "Work tickets, travel documents"),
    ("Fuel cards, fuel usage and transaction reconciliation", "Reconcile fuel card/fuel purchase transactions to work tickets and vehicle records.", "Fuel card statements, fuel records"),
    ("Fuel efficiency and abnormal consumption analysis", "Compute kilometres per litre per vehicle and investigate abnormal consumption (see analysis tool).", "Fuel records, odometer readings"),
    ("Multiple or suspicious fuel transactions", "Identify multiple or unusual fuel transactions (same vehicle/day, round amounts, unusual times) and seek explanations.", "Fuel transaction listing"),
    ("Vehicle maintenance and servicing", "Review servicing schedules against service records for sampled vehicles.", "Maintenance schedules, service records"),
    ("Repairs, expenditure, requisitions, LPOs, invoices and payment records", "Trace sampled repairs from requisition through LPO, invoice and payment.", "Requisitions, LPOs, invoices, payment vouchers"),
    ("Accident and breakdown records", "Review accident and breakdown records for reporting, follow-up and cost recovery.", "Accident/breakdown reports"),
    ("Driver authorisation and driving licences", "Verify that drivers of sampled journeys were authorised and held valid licences.", "Driver authorisation list, licences"),
    ("Vehicle custody and after-hours parking", "Review custody allocation and after-hours parking arrangements for sampled vehicles.", "Custody records, parking arrangements"),
    ("Physical fleet inspection and operational status", "Physically inspect sampled vehicles and record condition and operational status.", "Inspection checklist"),
    ("Fleet utilisation and vehicle downtime", "Compute fleet utilisation and analyse downtime (see analysis tool).", "Utilisation records, downtime records"),
    ("Disposal of obsolete or unserviceable vehicles", "Review identification, approval and disposal of obsolete/unserviceable vehicles.", "Disposal records, approvals"),
    ("Management explanations and supporting evidence", "Record management explanations for exceptions and evaluate the supporting evidence provided.", "Management responses, evidence"),
]
IT_CATEGORIES = [
    "IT governance and strategic planning", "IT policies, standards and procedures", "Data centre administration",
    "Physical and environmental security", "CCTV and access controls", "Data backup and restoration",
    "Business continuity and disaster recovery", "Systems development and change management",
    "Windows infrastructure and directory services", "User access and logical security",
    "Network and perimeter security", "Vulnerability management", "Endpoint security and malware protection",
    "Database security", "Application controls and audit trails", "IT asset management and maintenance",
    "System availability and incident management"]


def _crit(extra=""):
    return [{"text": "Applicable criterion to be identified by the auditor", "source": "", "clause": "", "verification": "Unverified"}]


def transport_content():
    procs = [{"ref": f"T{i:02d}", "title": t, "procedure": p, "evidence_required": e, "source_ref": NOTE_T,
              "expected": "", "sampling": "", "risk": "", "risk_level": "", "control_objective": ""}
             for i, (t, p, e) in enumerate(TRANSPORT_TOPICS, 1)]
    c = tl.blank_content()
    c.update(description="Transport, Fuel and Fleet Management - full review structure. " + CRIT_NOTE,
             objective="", scope="", sampling_methodology="To be determined by the auditor (population, period and sample size).",
             criteria=_crit(), procedures=procs)
    return c


def it_content(cat):
    c = tl.blank_content()
    c.update(description=f"{cat} - draft structure. " + CRIT_NOTE, criteria=_crit(),
             procedures=[{"ref": "I01", "title": cat,
                          "procedure": f"Obtain and review documentation, configuration and records relating to {cat.lower()}; evaluate design and operation of controls against the criteria the auditor identifies.",
                          "evidence_required": "Documents, configuration extracts and records identified by the auditor", "source_ref": NOTE_IT,
                          "expected": "", "sampling": "", "risk": "", "risk_level": "", "control_objective": ""}])
    return c


def seed(user):
    """Idempotent: creates missing starter templates only; never touches existing ones. Returns names created."""
    import plan_register as pr
    areas = {a["name"]: a["id"] for a in pr.list_areas()}
    made = []
    plan = [("Transport, Fuel and Fleet Management - full review", "Transport", areas.get("Transport, Fuel and Fleet Management"), transport_content())]
    ita = areas.get("Information Systems and ICT Infrastructure")
    plan += [(f"IT - {c}", "ICT", ita, it_content(c)) for c in IT_CATEGORIES]
    existing = {t["name"].lower() for t in tl.list_templates(status=None)}
    for name, fam, aid, content in plan:
        if name.lower() not in existing:
            tl.create_template(name, content, user, family=fam, area_id=aid, source_status=PENDING, note="Starter structure (source pending)")
            made.append(name)
    return made
