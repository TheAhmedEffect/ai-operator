"""
The tools the model may propose, and the code that runs them.

Rule 1 — "model talks, code operates" — is a module boundary: this file can
change data but never speaks to the model, and llm.py speaks to the model but
cannot change data.

Rule 2 — "confirm before you act" — is enforced by structure, not by discipline.
Tools live in two separate registries:

    READ_TOOLS   Ask mode. Read-only. Safe to run the moment the model asks.
    WRITE_TOOLS  Run a task mode. Mutates state. There is NO code path that
                 reaches these except execute_confirmed(), which requires a
                 PendingWrite, which only propose_write() can create, and which
                 the caller must have had a human approve.

The practical effect: forgetting to ask for confirmation is not a mistake this
program is capable of making. That survives someone adding a second write tool
next month without reading this docstring — which a well-placed `if` would not.
"""

import inspect
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Any, Callable, Mapping

import data


class ToolError(Exception):
    """A tool could not do what was asked.

    Rule 4 — fail loud. Every failure raises this with a readable reason that is
    handed back to the model to relay in plain language. Nothing returns None
    and hopes the caller notices.
    """


# =============================================================================
# ASK TOOLS  (read-only)
# =============================================================================

def get_insights() -> dict[str, Any]:
    """Computed statistics over the mock CRM. Reads only; changes nothing."""
    week_ago = data.TODAY - timedelta(days=7)
    new_this_week = [
        lead for lead in data.LEADS
        if date.fromisoformat(lead["created"]) >= week_ago
    ]
    return {
        "new_leads_this_week": len(new_this_week),
        "unassigned_leads": len([l for l in data.LEADS if l["assigned_to"] is None]),
        "never_contacted_leads": len([l for l in data.LEADS if l["last_contacted"] is None]),
        "total_leads": len(data.LEADS),
        "open_tasks": len([t for t in data.TASKS if not t["done"]]),
    }


def search_leads(query: str) -> dict[str, Any]:
    """Search leads by name, status or property interest.

    Rule 3 lives here: no match returns an explicit `match_count: 0`, which is a
    positive statement that we looked and found nothing — not silence for the
    model to fill.
    """
    if not query or not query.strip():
        raise ToolError("search_leads needs a non-empty query.")
    q = query.strip().lower()
    hits = [
        lead for lead in data.LEADS
        if q in lead["name"].lower()
        or q in lead["status"].lower()
        or q in lead["property_interest"].lower()
    ]
    return {"query": query, "match_count": len(hits), "matches": hits}


SORTABLE = {"created", "last_contacted", "name", "status"}
STATUSES = {"new", "contacted", "viewing_booked", "closed"}


def list_leads(
    sort_by: str = "created",
    order: str = "desc",
    limit: int | None = None,
    status: str | None = None,
    assigned_to: str | None = None,
    contacted: bool | None = None,
) -> dict[str, Any]:
    """List leads, sorted and optionally filtered. Read-only.

    Added after testing revealed a gap. Asked for "the most recent lead", the model
    had no tool that could order anything, so it searched by status and reasoned
    over what came back. It got the right answer — but only because the newest lead
    happened to have status 'new'. A recently-contacted lead would have been missed.

    The model produced a plausible, confident, *accidentally correct* answer by
    improvising around a missing capability. That is a tool design failure, not a
    model failure, and the fix is to give it the tool rather than a better prompt.

    Leads with no value for the sort field (e.g. never contacted) always sort last,
    in both directions — "most recently contacted" should not surface someone who
    has never been contacted at all.
    """
    if sort_by not in SORTABLE:
        raise ToolError(
            f"Cannot sort by '{sort_by}'. Sortable fields: {', '.join(sorted(SORTABLE))}."
        )
    if order not in {"asc", "desc"}:
        raise ToolError(f"order must be 'asc' or 'desc', not '{order}'.")
    if status is not None and status not in STATUSES:
        raise ToolError(
            f"'{status}' is not a lead status. Valid: {', '.join(sorted(STATUSES))}."
        )
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ToolError(f"limit must be a positive whole number, not '{limit}'.")
    if assigned_to is not None:
        # CURRENT_USER is always valid even if they currently hold no leads —
        # otherwise "who's assigned to me?" would error confusingly on a dataset
        # where this agent happens to own nothing.
        known = data.agents() | {"unassigned", data.CURRENT_USER}
        if assigned_to not in known:
            raise ToolError(
                f"No agent called '{assigned_to}'. Known: {', '.join(sorted(known))}."
            )

    pool = [l for l in data.LEADS if status is None or l["status"] == status]

    # Filtering belongs here, not in the model. Returning every record and letting
    # the model pick works on five leads and fails on five hundred — slowly, and
    # without saying so.
    if assigned_to == "unassigned":
        pool = [l for l in pool if l["assigned_to"] is None]
    elif assigned_to is not None:
        pool = [l for l in pool if l["assigned_to"] == assigned_to]

    if contacted is True:
        pool = [l for l in pool if l["last_contacted"] is not None]
    elif contacted is False:
        pool = [l for l in pool if l["last_contacted"] is None]

    have_value = [l for l in pool if l.get(sort_by) is not None]
    no_value = [l for l in pool if l.get(sort_by) is None]
    have_value.sort(key=lambda l: l[sort_by], reverse=(order == "desc"))
    ordered = have_value + no_value

    total = len(ordered)
    if limit is not None:
        ordered = ordered[:limit]

    return {
        "sorted_by": sort_by,
        "order": order,
        "filters": {"status": status, "assigned_to": assigned_to, "contacted": contacted},
        "total_matched": total,
        "returned": len(ordered),
        "leads": ordered,
    }


# =============================================================================
# RUN A TASK TOOLS  (writes)
# =============================================================================

def create_task(title: str, due: str, related_to: str | None = None) -> dict[str, Any]:
    """Create a task record.

    Never call this directly. It is only reachable via execute_confirmed(), and
    only with arguments that validate_create_task() has already cleaned and a
    human has already approved.
    """
    task = {
        "id": data.next_task_id(),
        "title": title,
        "due": due,
        "related_to": related_to,
        "done": False,
    }
    data.TASKS.append(task)
    return {
        "created": task,
        "open_tasks_now": len([t for t in data.TASKS if not t["done"]]),
    }


def validate_create_task(args: dict[str, Any]) -> dict[str, Any]:
    """Check and normalise the model's proposed arguments.

    This runs BEFORE the user is asked to confirm, so we never present someone
    with a confirmation prompt for an action that was going to fail anyway.

    This is the "code operates" half of rule 1 doing real work: the model is a
    good proposer and a bad validator. It will happily emit a confident,
    well-formed call referencing a lead that does not exist.
    """
    title = (args.get("title") or "").strip()
    if not title:
        raise ToolError("A task needs a title.")

    due_raw = (args.get("due") or "").strip()
    if not due_raw:
        raise ToolError("A task needs a due date in YYYY-MM-DD format.")
    try:
        due = datetime.strptime(due_raw, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise ToolError(f"'{due_raw}' is not a date I can use. I need YYYY-MM-DD.")

    related_to = args.get("related_to") or None
    if related_to is not None and related_to not in data.lead_ids():
        # Rule 3, enforced in code: refuse to attach a task to a lead we do not hold.
        raise ToolError(
            f"No lead with id '{related_to}' exists in the CRM. "
            f"Known leads: {', '.join(sorted(data.lead_ids()))}."
        )

    return {"title": title, "due": due, "related_to": related_to}


# =============================================================================
# REGISTRIES AND THE CONFIRMATION GATE
# =============================================================================

READ_TOOLS: dict[str, Callable[..., Any]] = {
    "get_insights": get_insights,
    "search_leads": search_leads,
    "list_leads": list_leads,
}

WRITE_TOOLS: dict[str, Callable[..., Any]] = {
    "create_task": create_task,
}

VALIDATORS: dict[str, Callable[[dict], dict]] = {
    "create_task": validate_create_task,
}


@dataclass(frozen=True)
class PendingWrite:
    """A write the model has proposed and a human has not yet approved.

    Frozen deliberately, and `args` is a read-only mapping rather than a dict.
    Both halves matter: `frozen=True` stops the field being rebound, and
    MappingProxyType stops the contents being edited in place. Without the
    second, `pending.args["title"] = ...` would still work and the guarantee
    below would be a convention rather than a mechanism.

    The guarantee: the arguments shown to the user in the confirmation prompt are
    the exact arguments that execute. Nothing — not the model, not a later code
    path — can alter the payload between the question being asked and the action
    being taken.
    """
    tool: str
    args: Mapping[str, Any]
    summary: str


def _readable_date(iso: str) -> str:
    """'2026-09-15' -> 'Tuesday 15 September 2026'.

    The confirmation prompt is the only place a human inspects what is about to
    happen, so it has to be readable at a glance. A bare ISO date is not: during
    testing the model resolved "Friday" to 2026-09-15, a Tuesday, and the prompt
    said "due 2026-09-15" — technically accurate and completely unreviewable.
    Spelling out the weekday turns a silent error into an obvious one.
    """
    d = date.fromisoformat(iso)
    return d.strftime("%A %d %B %Y").replace(" 0", " ")


def summarise_write(tool_name: str, args: dict[str, Any]) -> str:
    """Plain-language description of a proposed write, for the confirm prompt.

    The user is approving what this sentence says, so it must describe exactly
    what will happen — no rounding, no omissions, nothing that needs decoding.
    """
    if tool_name == "create_task":
        related = f", linked to {args['related_to']}" if args.get("related_to") else ""
        return (f'create a task: "{args["title"]}", '
                f'due {_readable_date(args["due"])}{related}')
    return f"run {tool_name} with {args}"


def propose_write(tool_name: str, raw_args: dict[str, Any]) -> PendingWrite:
    """Validate a proposed write and hold it. Executes NOTHING."""
    if tool_name not in WRITE_TOOLS:
        raise ToolError(f"'{tool_name}' is not a write tool I know about.")
    clean = VALIDATORS[tool_name](raw_args)
    # Read-only view: what the user is about to be shown is what will run.
    return PendingWrite(tool_name, MappingProxyType(clean), summarise_write(tool_name, clean))


def execute_confirmed(pending: PendingWrite) -> Any:
    """Run a write a human has approved.

    THE ONLY function in this project that calls anything in WRITE_TOOLS.
    If you are reading this looking for another way to trigger a mutation,
    there isn't one, and that is the point.
    """
    return WRITE_TOOLS[pending.tool](**pending.args)


def execute_read(tool_name: str, args: dict[str, Any]) -> Any:
    """Run a read-only tool. Safe as soon as the model asks for it.

    Argument shape is checked with inspect.signature().bind() *before* calling,
    rather than by catching TypeError afterwards. Catching it around the call
    would also swallow any TypeError raised inside the tool body and misreport a
    genuine bug as "you called this wrong" — an error message that sends the
    reader to the wrong place.
    """
    if tool_name not in READ_TOOLS:
        raise ToolError(f"'{tool_name}' is not a tool I have.")

    fn = READ_TOOLS[tool_name]
    try:
        inspect.signature(fn).bind(**args)
    except TypeError as exc:
        raise ToolError(f"'{tool_name}' was called with wrong arguments: {exc}")

    return fn(**args)


def is_write(tool_name: str) -> bool:
    return tool_name in WRITE_TOOLS


# =============================================================================
# SCHEMAS HANDED TO THE MODEL
# =============================================================================

def _date_table(days: int = 14) -> str:
    """A literal date lookup for the next fortnight, embedded in the tool schema.

    Testing showed the model resolving "Monday" to a Sunday, and giving different
    answers to the same question on different runs. Date arithmetic is something
    language models are unreliable at and computers are perfect at, so we stop
    asking it to calculate and give it a table to read instead.

    Cheaper and more robust than a stricter prompt, and it moves the failure from
    "silently wrong date" to "date not in the table, so ask the user".
    """
    lines = []
    for offset in range(days + 1):
        d = data.TODAY + timedelta(days=offset)
        label = {0: "  (today)", 1: "  (tomorrow)"}.get(offset, "")
        lines.append(f"  {d.strftime('%A')} {d.isoformat()}{label}")
    return "\n".join(lines)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_insights",
            "description": (
                "Ask mode, read-only. Returns computed statistics about the CRM: "
                "new leads this week, unassigned leads, never-contacted leads, "
                "total leads, and open tasks. Use for any 'how many' question."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_leads",
            "description": (
                "Ask mode, read-only. Keyword search across three fields ONLY: "
                "lead name, status, and property interest. Use when the user names "
                "something specific ('Aisha', 'Hackney').\n"
                "It does NOT search by assigned agent — a zero result here says "
                "nothing about who a lead is assigned to. For ownership use "
                "list_leads(assigned_to=...), and for ordering or recency use "
                "list_leads with sort_by.\n"
                "If match_count is 0 there is no lead matching that text: say so "
                "plainly, describe what you actually searched, and do not invent one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword, e.g. 'new', 'Hackney', 'Aisha'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_leads",
            "description": (
                "Ask mode, read-only. Lists leads in a defined order, optionally "
                "filtered by status. Use this for any question about ordering or "
                "recency — 'the most recent lead', 'the oldest one', 'who haven't "
                "we contacted', 'show me the newest three'. Do not try to work "
                "ordering out yourself from a search result; call this instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["created", "last_contacted", "name", "status"],
                        "description": (
                            "Field to order by. 'created' for newest/oldest lead, "
                            "'last_contacted' for most/least recently contacted."
                        ),
                    },
                    "order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "'desc' for most recent first, 'asc' for oldest first.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional. Max leads to return, e.g. 1 for 'the most recent'.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["new", "contacted", "viewing_booked", "closed"],
                        "description": "Optional status filter.",
                    },
                    "assigned_to": {
                        "type": "string",
                        "description": (
                            f"Optional. Filter to one agent's leads, or 'unassigned' "
                            f"for leads with no owner. The current user is "
                            f"{data.CURRENT_USER} — use that when they say 'me' or 'my'."
                        ),
                    },
                    "contacted": {
                        "type": "boolean",
                        "description": (
                            "Optional. false returns only leads never contacted, "
                            "true only those contacted at least once. Use the filter "
                            "rather than listing everything and picking them out."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": (
                "Run a task mode. Creates a task record. This is a WRITE: it does "
                "not execute when you call it. The user is shown a summary and "
                "must confirm first. Call it once with your best arguments; do "
                "not ask the user for permission yourself, the system does that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "What the task is."},
                    "due": {
                        "type": "string",
                        "description": (
                            "Due date as YYYY-MM-DD. Do NOT calculate dates "
                            "yourself — use this lookup:\n" + _date_table() +
                            "\nIf the date you need is not in this table, ask the "
                            "user for an explicit date instead of guessing at one."
                        ),
                    },
                    "related_to": {
                        "type": "string",
                        "description": (
                            "Optional lead id, e.g. 'L-1001'. Only use an id you "
                            "have actually seen in tool output in this conversation."
                        ),
                    },
                },
                "required": ["title", "due"],
            },
        },
    },
]
