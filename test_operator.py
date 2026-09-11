"""
Deterministic regression tests for the AI Operator.

    pip install pytest
    pytest -v

None of these make a live model call. They test the layer the model talks *to*,
which is where every safety guarantee actually lives — and it means the suite is
fast, free and gives the same answer every time.

Why this file exists: six bugs were found by hand during the build, and two of
them were regressions introduced while fixing something else. A prompt change
broke working scoping behaviour and it was only caught by chance, re-running an
old question. Prompts are code, and code that matters needs assertions.

What is NOT covered, honestly: whether the model *chooses* the right tool. That
needs a live call. The tests below assert that once a tool is chosen, the code
around it behaves — no mutation without confirmation, no invented ids, no
silently altered payloads.
"""

import copy
from types import SimpleNamespace

import pytest

import data
import llm
import tools
from tools import ToolError


# --- Isolation ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_dataset():
    """Snapshot and restore the in-memory CRM around every test.

    TASKS is mutated by create_task, so without this a test that writes would
    leak into the next one and the suite would depend on execution order.
    """
    leads = copy.deepcopy(data.LEADS)
    tasks = copy.deepcopy(data.TASKS)
    yield
    data.LEADS[:] = leads
    data.TASKS[:] = tasks


def fake_call(call_id: str, name: str, arguments: str = "{}"):
    """Minimal stand-in for an OpenAI tool_call object."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


# =============================================================================
# Structural invariants
# =============================================================================

def test_read_and_write_registries_are_disjoint():
    """A tool is read-only or it mutates. Never both, never neither."""
    assert set(tools.READ_TOOLS) & set(tools.WRITE_TOOLS) == set()


def test_every_write_tool_has_a_validator():
    """A write with no validator would reach a confirmation prompt unchecked."""
    assert set(tools.WRITE_TOOLS) <= set(tools.VALIDATORS)


def test_every_schema_maps_to_a_real_tool():
    """The model can only be offered tools that actually exist."""
    advertised = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert advertised == set(tools.READ_TOOLS) | set(tools.WRITE_TOOLS)


# =============================================================================
# Ask tools never mutate
# =============================================================================

@pytest.mark.parametrize("name,args", [
    ("get_insights", {}),
    ("search_leads", {"query": "new"}),
    ("search_leads", {"query": "nothing matches this"}),
    ("list_leads", {}),
    ("list_leads", {"sort_by": "created", "order": "desc", "limit": 1}),
    ("list_leads", {"assigned_to": data.CURRENT_USER}),
])
def test_ask_tools_do_not_mutate(name, args):
    before_tasks = copy.deepcopy(data.TASKS)
    before_leads = copy.deepcopy(data.LEADS)
    tools.execute_read(name, args)
    assert data.TASKS == before_tasks
    assert data.LEADS == before_leads


def test_unknown_tool_is_rejected():
    with pytest.raises(ToolError):
        tools.execute_read("drop_all_leads", {})


def test_write_tool_cannot_be_run_through_the_read_path():
    """create_task is a write; execute_read must refuse it outright."""
    with pytest.raises(ToolError):
        tools.execute_read("create_task", {"title": "x", "due": "2026-09-14"})


def test_wrong_arguments_fail_loudly():
    with pytest.raises(ToolError):
        tools.execute_read("search_leads", {"not_a_parameter": "x"})


# =============================================================================
# Rule 3 — never invent what we do not have
# =============================================================================

def test_search_miss_is_explicit_not_empty():
    """Zero matches is a positive statement, not silence."""
    result = tools.execute_read("search_leads", {"query": "Michael Fenwick"})
    assert result["match_count"] == 0
    assert result["matches"] == []


def test_unknown_lead_rejected_before_confirmation():
    """Validation refuses, so no PendingWrite is ever created to confirm."""
    with pytest.raises(ToolError) as exc:
        tools.propose_write("create_task", {
            "title": "Chase paperwork", "due": "2026-09-14", "related_to": "L-9999",
        })
    assert "L-9999" in str(exc.value)


def test_unparseable_date_rejected():
    with pytest.raises(ToolError):
        tools.propose_write("create_task", {"title": "Call them", "due": "Friday"})


def test_empty_title_rejected():
    with pytest.raises(ToolError):
        tools.propose_write("create_task", {"title": "   ", "due": "2026-09-14"})


def test_unknown_agent_rejected():
    with pytest.raises(ToolError):
        tools.execute_read("list_leads", {"assigned_to": "Dave"})


def test_current_user_is_always_a_valid_agent():
    """M-3: must not error even if this agent happens to hold no leads."""
    result = tools.execute_read("list_leads", {"assigned_to": data.CURRENT_USER})
    assert all(l["assigned_to"] == data.CURRENT_USER for l in result["leads"])


# =============================================================================
# Rule 2 — confirm before you act
# =============================================================================

VALID_TASK = {"title": "Call Aisha Khan", "due": "2026-09-14", "related_to": "L-1001"}


def test_proposing_a_write_executes_nothing():
    before = len(data.TASKS)
    pending = tools.propose_write("create_task", VALID_TASK)
    assert isinstance(pending, tools.PendingWrite)
    assert len(data.TASKS) == before


def test_cancelled_write_changes_nothing():
    before = copy.deepcopy(data.TASKS)
    pending = tools.propose_write("create_task", VALID_TASK)
    messages = llm.new_conversation()
    llm.resolve_pending(messages, pending, approved=False)
    assert data.TASKS == before


def test_confirmed_write_creates_exactly_one_record():
    before = len(data.TASKS)
    pending = tools.propose_write("create_task", VALID_TASK)
    messages = llm.new_conversation()
    llm.resolve_pending(messages, pending, approved=True)
    assert len(data.TASKS) == before + 1
    assert data.TASKS[-1]["title"] == "Call Aisha Khan"


def test_confirmation_prompt_names_the_weekday():
    """Bug 3: a bare ISO date is unreviewable, so a wrong weekday stays invisible."""
    pending = tools.propose_write("create_task", VALID_TASK)
    assert "Monday" in pending.summary
    assert "Call Aisha Khan" in pending.summary


# =============================================================================
# M-2 — the confirmed payload is the executed payload
# =============================================================================

def test_pending_args_cannot_be_mutated_in_place():
    pending = tools.propose_write("create_task", VALID_TASK)
    with pytest.raises(TypeError):
        pending.args["title"] = "Something the user never approved"


def test_pending_fields_cannot_be_rebound():
    pending = tools.propose_write("create_task", VALID_TASK)
    with pytest.raises(Exception):
        pending.args = {"title": "swapped", "due": "2026-09-14", "related_to": None}


def test_executed_record_matches_what_was_confirmed():
    pending = tools.propose_write("create_task", VALID_TASK)
    approved = dict(pending.args)
    result = tools.execute_confirmed(pending)
    created = result["created"]
    assert created["title"] == approved["title"]
    assert created["due"] == approved["due"]
    assert created["related_to"] == approved["related_to"]


# =============================================================================
# Scoping and filtering  (the code half of bugs 5 and 6)
# =============================================================================

def test_assigned_to_filter_scopes_correctly():
    result = tools.execute_read("list_leads", {"assigned_to": data.CURRENT_USER})
    assert result["total_matched"] == 3
    assert {l["id"] for l in result["leads"]} == {"L-1003", "L-1004", "L-1005"}


def test_no_filter_returns_everyone():
    """Bug 6: "we" questions must not be narrowed to the current user."""
    result = tools.execute_read("list_leads", {})
    assert result["total_matched"] == len(data.LEADS)


def test_contacted_filter():
    never = tools.execute_read("list_leads", {"contacted": False})
    assert never["total_matched"] == 3
    assert all(l["last_contacted"] is None for l in never["leads"])

    contacted = tools.execute_read("list_leads", {"contacted": True})
    assert contacted["total_matched"] == 2


def test_filters_compose():
    """My leads AND never contacted — one record, not three."""
    result = tools.execute_read("list_leads", {
        "assigned_to": data.CURRENT_USER, "contacted": False,
    })
    assert result["total_matched"] == 1
    assert result["leads"][0]["id"] == "L-1003"


def test_unassigned_filter():
    result = tools.execute_read("list_leads", {"assigned_to": "unassigned"})
    assert {l["id"] for l in result["leads"]} == {"L-1001", "L-1002"}


def test_null_sort_values_always_sort_last():
    """Never-contacted leads must not surface as "most recently contacted"."""
    for order in ("asc", "desc"):
        result = tools.execute_read(
            "list_leads", {"sort_by": "last_contacted", "order": order})
        contacted = [l["last_contacted"] for l in result["leads"]]
        seen_none = False
        for value in contacted:
            if value is None:
                seen_none = True
            else:
                assert not seen_none, f"a null sorted before a real value ({order})"


def test_most_recent_lead_is_correct():
    """Bug 4: this used to be answered by luck, via a status search."""
    result = tools.execute_read(
        "list_leads", {"sort_by": "created", "order": "desc", "limit": 1})
    assert result["leads"][0]["id"] == "L-1001"
    assert result["total_matched"] == len(data.LEADS)   # limit truncates, not filters


def test_invalid_sort_field_rejected():
    with pytest.raises(ToolError) as exc:
        tools.execute_read("list_leads", {"sort_by": "vibes"})
    assert "created" in str(exc.value)   # the error lists what IS allowed


# =============================================================================
# H-1 — every tool call in a batch gets a response
# =============================================================================

def test_unprocessed_calls_still_get_a_tool_message():
    """The API requires one tool message per tool_call. Stopping a batch early
    must not leave any unanswered, or the NEXT model call fails on malformed
    history — one turn later, far from the cause."""
    messages: list[dict] = []
    remaining = [fake_call("call_2", "search_leads"), fake_call("call_3", "get_insights")]

    llm._answer_remaining(messages, remaining, "a write is awaiting confirmation")

    assert len(messages) == 2
    assert [m["tool_call_id"] for m in messages] == ["call_2", "call_3"]
    assert all(m["role"] == "tool" for m in messages)
    assert all("not_processed" in m["content"] for m in messages)


def test_answer_remaining_on_empty_batch_is_a_no_op():
    messages: list[dict] = []
    llm._answer_remaining(messages, [], "nothing to skip")
    assert messages == []


# =============================================================================
# Determinism
# =============================================================================

def test_insights_are_stable():
    """TODAY is pinned, so the same question gives the same answer every run."""
    assert tools.get_insights() == tools.get_insights()


def test_insights_match_the_dataset():
    insights = tools.get_insights()
    assert insights["total_leads"] == len(data.LEADS)
    assert insights["new_leads_this_week"] == 3
    assert insights["unassigned_leads"] == 2
    assert insights["never_contacted_leads"] == 3


def test_date_table_covers_the_next_fortnight():
    """Bug 3's fix: the model reads dates rather than calculating them."""
    table = tools._date_table()
    assert data.TODAY.isoformat() in table
    assert "(today)" in table and "(tomorrow)" in table
    assert table.count("\n") == 14   # 15 lines, today + 14 ahead
