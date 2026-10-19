"""Configurable starting templates. Items are suggestions only: the auditor selects what applies
and must verify any applicable requirement. No policy clauses or laws are asserted here."""
_RAW = {
"ICT": """Asset management|Assets not recorded or safeguarded|Obtain asset register; sample items to verify existence, tagging, custody and condition|Asset register, tags, custody records
Equipment maintenance|Equipment failure through poor maintenance|Review maintenance schedule and service records; confirm work performed|Maintenance schedule, service reports
Antivirus protection|Malware infection|Verify antivirus coverage, licence validity and update status on sampled devices|Antivirus console report, licences
Logical access controls|Unauthorised system access|Review user lists, access approvals, leavers and privileged accounts|User lists, access request forms
CCTV surveillance|Security incidents undetected|Verify CCTV coverage, functionality, retention and footage access|CCTV inventory, access logs
Network security|Network compromise|Review firewall configuration, segmentation and patching|Configuration records, patch logs
Backup management|Data loss|Review backup schedule, offsite storage and restoration tests|Backup logs, restore test records
System availability|Service disruption|Review downtime logs, incidents and recovery arrangements|Downtime logs, incident register
Access logs|Inappropriate activity undetected|Verify logging is enabled, retained and reviewed|Log extracts, review evidence
IT service management|Untracked or unresolved user issues|Test helpdesk tickets for logging, prioritisation and closure|Helpdesk ticket register""",
"Transport": """Vehicle allocation|Vehicles allocated without authority|Compare allocation records with approvals and needs|Allocation register, approvals
Vehicle logbooks|Incomplete or unreliable usage records|Test logbooks for completeness, authorisation and consistency|Logbooks
Fuel consumption|Fuel misuse|Reconcile fuel issued to logbook mileage and expected consumption|Fuel records, logbooks, receipts
Mileage records|Inaccurate mileage reporting|Recompute mileage and compare with odometer and trip records|Mileage records, trip sheets
Maintenance schedules|Vehicles unmaintained|Review schedules against service history|Maintenance schedule, job cards
Vehicle servicing|Unnecessary or unperformed servicing|Verify servicing invoices, job cards and approvals|Service invoices, job cards
Insurance|Uninsured vehicles|Confirm valid cover for all vehicles|Insurance certificates
Inspection records|Unroadworthy vehicles|Review inspection records and follow-up of defects|Inspection reports
Driver authorization|Unauthorised drivers|Verify licences and driver authorisation|Driver licences, authorisation list
Trip authorization|Unauthorised trips|Test trips for approved travel requests|Trip authorisations
Vehicle utilization|Under-utilised fleet|Analyse utilisation by vehicle|Logbooks, utilisation analysis""",
"Procurement": """Procurement planning|Purchases outside the plan|Compare purchases with approved procurement plan|Procurement plan, LPOs
Requisitions|Unjustified requisitions|Test requisitions for need and approval|Requisition forms
Approvals|Approval limits bypassed|Verify approvals against delegated authority|Approval records
Tender evaluation|Unfair evaluation|Review evaluation process and records|Tender documents, evaluation reports
Contract management|Poor contract performance|Review contracts, variations and monitoring|Contracts, variation orders
Supplier records|Ineligible or fictitious suppliers|Verify supplier registration and due diligence|Supplier files
Delivery verification|Goods not received|Match deliveries to orders and inspection records|GRNs, inspection reports
Payment documentation|Payments without support|Test payments for complete supporting documents|Vouchers, invoices""",
"Finance": """Expenditure verification|Irregular expenditure|Vouch sampled expenditure to supporting documents|Payment vouchers
Reconciliations|Unresolved differences|Review bank and ledger reconciliations|Reconciliation statements
Payment authorization|Unauthorised payments|Verify authorisation levels and segregation of duties|Approval records
Supporting documents|Missing documentation|Check completeness of supporting documents|Source documents
Budget controls|Overspending|Compare actuals with approved budget and commitments|Budget reports
Financial reporting|Misstatement|Review preparation and review of financial reports|Financial reports""",
"Human Resources": """Recruitment|Irregular appointments|Test appointments against approvals|Personnel files
Payroll|Ghost or incorrect pay|Reconcile payroll to establishment|Payroll, establishment list
Leave management|Unrecorded leave|Test leave records|Leave records
Staff records|Incomplete files|Review personnel file completeness|Personnel files""",
"General Operations": """Records management|Loss of records|Review record keeping arrangements|Registers
Risk management|Unmanaged risks|Review risk register and mitigation|Risk register
Asset custody|Asset loss|Verify sampled assets|Asset register""",
}
CATEGORIES = list(_RAW)


def _build():
    out = {}
    for cat, raw in _RAW.items():
        out[cat] = [dict(zip(("area", "risk", "procedure", "evidence_required"), ln.split("|")))
                    for ln in raw.splitlines()]
    return out


TEMPLATES = _build()
