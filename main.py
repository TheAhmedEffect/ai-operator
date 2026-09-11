"""
AI Operator — a CRM chat assistant prototype.

    python main.py            the assistant (Phase 4)
    python main.py --data     inspect the mock CRM      (Phase 1)
    python main.py --ask      read-only tool demo       (Phase 2)
    python main.py --write    confirmation gate demo    (Phase 3)
"""

import sys

import data
import llm
import tools


# =============================================================================
# PHASE 4 — the chat loop
# =============================================================================

YES = {"y", "yes", "yeah", "yep", "go ahead", "do it", "confirm", "ok", "okay"}
NO = {"n", "no", "nope", "cancel", "stop", "don't", "dont"}


def run_chat() -> None:
    """Interactive session.

    One `messages` list lives for the whole session, so the assistant remembers
    what was said earlier — you can ask about a lead, then say "create a task for
    her" and it knows who "her" is.

    Three things can happen on a turn:
        1. plain answer            - model replied in words
        2. read tool(s) ran        - our code executed them, model phrased the result
        3. a write is proposed     - HELD. The next thing typed is a yes/no answer
                                     to that specific proposal, and the model is
                                     not consulted until it is resolved.

    (3) is a state rather than a branch, which is why `pending` is checked before
    anything else in the loop. While a write is held, there is no code path that
    reaches the model at all.
    """
    client = llm.get_client()
    messages = llm.new_conversation()
    pending: tools.PendingWrite | None = None

    print("AI OPERATOR")
    print(f"{llm.MODEL}  ·  {data.TODAY.strftime('%A %d %B %Y')}")
    print("Ask a question or run a task across your CRM.")
    print("Type 'tasks' or 'leads' to see the raw data, 'quit' to exit.\n")
    print("How can I help you today?\n")

    while True:
        try:
            prompt = "  (yes / no) > " if pending else "you > "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not user_input:
            continue

        # ---- A write is held: this turn is the answer to it, nothing else ----
        if pending is not None:
            lowered = user_input.lower()
            if lowered in YES:
                outcome = llm.resolve_pending(messages, pending, approved=True)
            elif lowered in NO:
                outcome = llm.resolve_pending(messages, pending, approved=False)
            else:
                print("  Please answer yes or no — the task is still waiting.\n")
                continue
            print(f"operator > {outcome}\n")
            pending = None
            continue

        # ---- Session commands, no model call needed --------------------------
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("bye")
            return
        if user_input.lower() == "tasks":
            print()
            print_tasks()
            continue
        if user_input.lower() == "leads":
            print()
            print_leads()
            continue

        # ---- Normal turn ------------------------------------------------------
        reply, pending = llm.run_turn(client, messages, user_input, verbose=True)

        if pending is not None:
            print(f"\noperator > I'll {pending.summary}. Shall I go ahead?\n")
        else:
            print(f"operator > {reply}\n")


# =============================================================================
# Shared printing
# =============================================================================

def print_leads() -> None:
    print(f"LEADS ({len(data.LEADS)})")
    print("-" * 72)
    for lead in data.LEADS:
        contacted = lead["last_contacted"] or "never"
        assigned = lead["assigned_to"] or "unassigned"
        print(f"  {lead['id']}  {lead['name']:<16} {lead['status']:<15} "
              f"created {lead['created']}")
        print(f"          {lead['property_interest']}")
        print(f"          assigned: {assigned}   last contacted: {contacted}")
    print()


def print_tasks(header: str = "TASKS") -> None:
    print(f"{header} ({len(data.TASKS)})")
    print("-" * 72)
    for task in data.TASKS:
        state = "done" if task["done"] else "open"
        print(f"  {task['id']}  {task['title']}")
        print(f"          due {task['due']}   {state}   related to {task['related_to']}")
    print()


# =============================================================================
# PHASE 1 — inspect the dataset
# =============================================================================

def show_dataset() -> None:
    print("AI OPERATOR — mock CRM")
    print(f"Today is pinned to {data.TODAY.isoformat()}\n")
    print_leads()
    print_tasks()
    print(f"Known lead ids: {', '.join(sorted(data.lead_ids()))}")
    print(f"Next task id would be: {data.next_task_id()}")


# =============================================================================
# PHASE 2 — read-only demo
# =============================================================================

ASK_QUESTIONS = [
    "How many new leads do I have this week?",
    "Do we have any leads interested in Hackney?",
    "What's the status of our lead Michael Fenwick?",
    "Book me a flight to Lisbon on Friday.",
]


def run_ask_demo() -> None:
    client = llm.get_client()
    print("AI OPERATOR — Phase 2: Ask tools (read-only)")
    print(f"Model: {llm.MODEL}   Today: {data.TODAY.isoformat()}")
    print("=" * 72)
    for i, question in enumerate(ASK_QUESTIONS, start=1):
        print(f"\n[{i}] you: {question}")
        messages = llm.new_conversation()
        reply, _ = llm.run_turn(client, messages, question)
        print(f"    operator: {reply}")


# =============================================================================
# PHASE 3 — confirmation gate demo
# =============================================================================

def _ask_confirmation(pending: tools.PendingWrite) -> bool:
    print(f"\n    operator: I'll {pending.summary}. Shall I go ahead?")
    while True:
        answer = input("    (yes / no): ").strip().lower()
        if answer in YES:
            return True
        if answer in NO:
            return False
        print("    Please answer yes or no.")


def _write_scenario(client, prompt: str, label: str) -> None:
    print(f"\n{label}")
    print("-" * 72)
    print(f"you: {prompt}")

    messages = llm.new_conversation()
    reply, pending = llm.run_turn(client, messages, prompt)

    if pending is None:
        print(f"    operator: {reply}")
        return

    before = len(data.TASKS)
    approved = _ask_confirmation(pending)
    outcome = llm.resolve_pending(messages, pending, approved)
    print(f"    operator: {outcome}")
    changed = len(data.TASKS) - before
    print(f"    [tasks before: {before}  ->  after: {len(data.TASKS)}"
          f"   ({'+1 written' if changed else 'nothing written'})]")


def run_write_demo() -> None:
    client = llm.get_client()
    print("AI OPERATOR — Phase 3: write tool + confirmation gate")
    print(f"Model: {llm.MODEL}   Today: {data.TODAY.isoformat()}")
    print("=" * 72)
    print()
    print_tasks("TASKS BEFORE")

    _write_scenario(client, "Create a task to call Aisha Khan on Friday",
                    "SCENARIO 1 — valid write. Answer 'yes'.")
    # The date is deliberate. Without one the model correctly asks for a date and
    # never proposes the write, so validation is never reached and the scenario
    # demonstrates nothing. Supplying it lets the call through to the check we
    # actually want to show: an unknown lead id refused before any confirmation.
    _write_scenario(client, "Create a task tomorrow to chase the paperwork for lead L-9999",
                    "SCENARIO 2 — unknown lead. Validation should refuse it "
                    "BEFORE you are asked to confirm.")
    _write_scenario(client, "Create a task to email Sajmon Gjyzeli tomorrow",
                    "SCENARIO 3 — valid write. Answer 'no'.")

    print()
    print_tasks("TASKS AFTER")


# =============================================================================

def main() -> None:
    if "--data" in sys.argv:
        show_dataset()
    elif "--ask" in sys.argv:
        run_ask_demo()
    elif "--write" in sys.argv:
        run_write_demo()
    else:
        run_chat()


if __name__ == "__main__":
    main()
