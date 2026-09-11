"""
Phase 1 — the mock CRM.

Everything here is invented. No real client or production data is used anywhere
in this project.

Held in memory rather than a database on purpose: this is a one-week prototype,
and an in-memory list makes a write genuinely observable when you run it. TASKS
gets appended to in Phase 3.
"""

from datetime import date, timedelta

# A fixed "today" rather than date.today(). This keeps anything we compute later
# (e.g. "new leads this week") deterministic, so the same question always gives
# the same answer and a reviewer can check our arithmetic by hand.
TODAY = date(2026, 9, 11)

# Who is using the assistant. Without this the system has no way to resolve "me",
# "my leads" or "assigned to me" — during testing the assistant was asked "who's
# assigned to me?" and answered that the leads belonged to someone called Zisan
# Ahmed, as though that were a third party. It had no identity to compare against.
#
# In production this comes from the session, and it is also what permission
# scoping would be built on.
CURRENT_USER = "Zisan Ahmed"


def _days_ago(n: int) -> str:
    return (TODAY - timedelta(days=n)).isoformat()


def _days_ahead(n: int) -> str:
    return (TODAY + timedelta(days=n)).isoformat()


# --- Leads -------------------------------------------------------------------
# status: new | contacted | viewing_booked | closed
LEADS = [
    {
        "id": "L-1001",
        "name": "Aisha Khan",
        "email": "aisha.khan@example.com",
        "status": "new",
        "assigned_to": None,
        "property_interest": "2-bed flat, Hackney",
        "created": _days_ago(1),
        "last_contacted": None,
    },
    {
        "id": "L-1002",
        "name": "Sajmon Gjyzeli",
        "email": "sajmon@example.com",
        "status": "new",
        "assigned_to": None,
        "property_interest": "3-bed terrace, Walthamstow",
        "created": _days_ago(3),
        "last_contacted": None,
    },
    {
        "id": "L-1003",
        "name": "Dhruval",
        "email": "dhruval@example.com",
        "status": "new",
        "assigned_to": "Zisan Ahmed",
        "property_interest": "Studio, Bermondsey",
        "created": _days_ago(5),
        "last_contacted": None,
    },
    {
        "id": "L-1004",
        "name": "Margarita Damai",
        "email": "margarita@example.com",
        "status": "contacted",
        "assigned_to": "Zisan Ahmed",
        "property_interest": "2-bed flat, Peckham",
        "created": _days_ago(11),
        "last_contacted": _days_ago(9),
    },
    {
        "id": "L-1005",
        "name": "Tanvir Shakib",
        "email": "shakib@example.com",
        "status": "viewing_booked",
        "assigned_to": "Zisan Ahmed",
        "property_interest": "1-bed flat, Deptford",
        "created": _days_ago(14),
        "last_contacted": _days_ago(2),
    },
]

# --- Tasks -------------------------------------------------------------------
TASKS = [
    {
        "id": "T-2001",
        "title": "Chase valuation paperwork for 14 Ardley Road",
        "due": _days_ahead(1),
        "related_to": "L-1004",
        "done": False,
    },
    {
        "id": "T-2002",
        "title": "Send tenancy pack to Tanvir Shakib",
        "due": _days_ahead(3),
        "related_to": "L-1005",
        "done": False,
    },
]


def next_task_id() -> str:
    """Sequential id, so a task created in Phase 3 is obviously a new record."""
    return f"T-{2000 + len(TASKS) + 1}"


def agents() -> set[str]:
    """Everyone leads are currently assigned to."""
    return {l["assigned_to"] for l in LEADS if l["assigned_to"] is not None}


def lead_ids() -> set[str]:
    """Every lead id we actually hold.

    Used from Phase 3 onward to reject a write that references a lead we do not
    have — rule 3, enforced in code rather than left to the prompt.
    """
    return {lead["id"] for lead in LEADS}
