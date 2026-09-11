"""
The OpenAI connection and the conversation turn.

This module talks to the model. It has no idea what any tool does — it only
knows how to hand a name and arguments to `tools` and pass the result back.
That is rule 1 as a module boundary rather than a convention.

A turn returns one of two things:

    (reply, None)      the model answered, possibly after read-only tools ran
    (None, pending)    the model proposed a WRITE; nothing has happened yet and
                       the caller must get a human answer before continuing

`messages` is passed in and mutated, so the caller owns conversation history.
Phase 4's chat loop needs that; Phase 3 just uses a fresh list each time.
"""

import json
import os
import sys
from typing import Any

import data
import tools

MODEL = os.environ.get("AI_OPERATOR_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = f"""You are the AI Operator, an assistant inside a CRM used by a UK
residential estate agency.

You are assisting {data.CURRENT_USER}, an agent at the agency.

Read pronouns carefully, because they change the answer:

  "me", "my", "I", "mine"        -> {data.CURRENT_USER} only.
                                    Pass assigned_to="{data.CURRENT_USER}".
  "we", "us", "our", "the team", -> the whole agency, every lead regardless of owner.
  "the agency", "anyone"            Do NOT set assigned_to at all.

"Which of my leads haven't I contacted?" is scoped to {data.CURRENT_USER}.
"Who haven't we contacted?" is not — it covers everyone. If you are unsure which was
meant, ask rather than guess; silently narrowing the scope produces an answer that
looks right and is incomplete.

You have two modes.

Ask — you look things up and answer. Read-only.

Run a task — you change something.

  To change anything you MUST emit the tool call. This is not optional and there is
  no other route. Describing a change in words does nothing at all.

  Do NOT write "shall I go ahead?", "please confirm", or "is that correct?" about a
  write. The system shows the user a summary and collects their yes or no. If you
  ask in prose instead of emitting the call, nothing has been proposed, nothing is
  pending, and nothing will ever happen — however clearly you described it.

  Never tell the user a record was created unless a tool result in this conversation
  says it was. Saying "done" without a tool result is a false statement about the
  state of their CRM, which is worse than refusing.

  If you are missing a required argument, emit nothing and ask for that one argument.

Hard rules:

1. You have no direct access to the CRM. You request a tool; the system runs it and
   returns the result. Only state facts that came back from a tool call.
2. Never invent a lead, a name, a number or a date. If a search returns match_count
   0, say we have no such lead. Do not guess or fill gaps.
3. If a tool returns an error, relay what went wrong in plain language. Do not retry
   silently and do not pretend it worked.
4. Be brief and factual. Plain sentences, no markdown formatting, no bullet lists,
   no padding.

Today is {data.TODAY.strftime('%A %d %B %Y')} ({data.TODAY.isoformat()}).
When a user says "Friday", "tomorrow" or "next week", work the actual date out from
that and double-check the weekday matches before you call a tool. Getting a date
wrong is one of the easiest mistakes to make and one of the hardest for a user to
spot.
"""


def get_client():
    """Create the OpenAI client, failing loudly and usefully if we can't."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai is not installed. Run:  pip install -r requirements.txt")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY is not set.\n"
            "  CMD:         set OPENAI_API_KEY=sk-...\n"
            "  PowerShell:  $env:OPENAI_API_KEY='sk-...'\n"
            "  bash:        export OPENAI_API_KEY=sk-..."
        )
    return OpenAI()


def new_conversation() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _call_model(client, messages: list[dict]):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools.TOOL_SCHEMAS,
        tool_choice="auto",
    )


def _tool_message(call_id: str, name: str, payload: Any) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, default=str),
    }


def _answer_remaining(messages: list[dict], remaining, reason: str) -> None:
    """Give every unprocessed tool call in a batch a response message.

    The API requires that each tool_call in an assistant message is answered by a
    matching tool message. The model can emit several calls at once — a lookup and
    a write together, for instance — and we deliberately stop processing the batch
    early when a write needs confirming, or when one fails validation.

    Without this, the unprocessed calls would be left unanswered, the conversation
    history would be malformed, and the NEXT model call on that history would fail
    — one turn later, at a point unrelated to the cause. Filling them in keeps the
    history well-formed and tells the model plainly why they were skipped.
    """
    for call in remaining:
        messages.append(_tool_message(
            call.id,
            call.function.name,
            {"status": "not_processed", "reason": reason},
        ))


# A turn may need several model steps: look a lead up, then act on what it found.
# Bounded so a model that loops on itself stops rather than spinning forever —
# rule 4 again, a limit that reports itself is better than one that hangs.
MAX_STEPS = 5


def run_turn(client, messages: list[dict], user_input: str, verbose: bool = True):
    """One user turn. Returns (reply, pending_write).

    The model may need more than one step — a natural request like "create a task
    to call Aisha Khan" means search_leads() first, then create_task() with the id
    it found. So we loop: call the model, run whatever read tools it asks for, feed
    the results back, and let it decide again.

    The loop exits the moment a WRITE is proposed. Nothing is executed; the pending
    write is returned and the model is not consulted again until a human answers.
    """
    messages.append({"role": "user", "content": user_input})

    for _step in range(MAX_STEPS):
        try:
            response = _call_model(client, messages)
        except Exception as exc:
            return f"[model call failed: {exc}]", None

        message = response.choices[0].message

        # No tool call: the model is answering in words. Turn is done.
        if not message.tool_calls:
            reply = message.content or "(no reply)"
            messages.append({"role": "assistant", "content": reply})
            return reply, None

        # Record what the model proposed, before anything acts on it.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in message.tool_calls
            ],
        })

        # The model may emit several calls in one response — a lookup and a write
        # together, say. We stop the batch early in two cases below, so we must
        # answer whatever is left before doing so. See _answer_remaining().
        batch = list(message.tool_calls)

        for index, call in enumerate(batch):
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                messages.append(_tool_message(
                    call.id, name, {"error": "arguments were not valid JSON"}))
                continue

            if verbose:
                print(f"    [model proposed: {name}({args})]")

            # ---- WRITE: hold it, do not run it ----------------------------
            if tools.is_write(name):
                try:
                    pending = tools.propose_write(name, args)
                except tools.ToolError as exc:
                    # Validation failed. The user is never shown a confirmation
                    # for something that was going to fail — rule 4 before rule 2.
                    if verbose:
                        print(f"    [validation rejected it: {exc}]")
                    messages.append(_tool_message(call.id, name, {"error": str(exc)}))
                    _answer_remaining(
                        messages, batch[index + 1:],
                        "skipped: an earlier tool call in this batch failed validation")
                    break  # back to the loop; model explains the refusal

                if verbose:
                    print("    [HELD — not executed, awaiting confirmation]")
                messages.append(_tool_message(
                    call.id, name,
                    {"status": "awaiting_user_confirmation", "proposed": pending.summary}))
                _answer_remaining(
                    messages, batch[index + 1:],
                    "skipped: a write in this batch is awaiting user confirmation")
                return None, pending

            # ---- READ: safe to run now ------------------------------------
            try:
                result = tools.execute_read(name, args)
                messages.append(_tool_message(call.id, name, result))
                if verbose:
                    print(f"    [our code executed it -> {result}]")
            except tools.ToolError as exc:
                messages.append(_tool_message(call.id, name, {"error": str(exc)}))
                if verbose:
                    print(f"    [tool failed: {exc}]")

        # Loop round: the model now sees the tool results and decides what next.

    # Ran out of steps without settling. Say so rather than hanging or guessing.
    give_up = (
        f"I couldn't complete that in {MAX_STEPS} steps without going in circles. "
        "Try asking for one thing at a time."
    )
    messages.append({"role": "assistant", "content": give_up})
    return give_up, None


def resolve_pending(messages: list[dict], pending: tools.PendingWrite, approved: bool) -> str:
    """Execute or discard a held write, and record the outcome in history.

    The only caller of tools.execute_confirmed(). `approved` comes from a human,
    never from the model.
    """
    if not approved:
        messages.append({"role": "user", "content": f"Cancelled: {pending.summary}"})
        messages.append({"role": "assistant", "content": "Cancelled. Nothing was changed."})
        return "Cancelled. Nothing was changed."

    try:
        result = tools.execute_confirmed(pending)
    except tools.ToolError as exc:
        # Same user/assistant shape as the other two outcomes, so history stays
        # uniform however the write ended.
        messages.append({"role": "user", "content": f"Confirmed: {pending.summary}"})
        messages.append({"role": "assistant", "content": f"Failed: {exc}"})
        return f"I couldn't do that — {exc}"

    messages.append({"role": "user", "content": f"Confirmed: {pending.summary}"})
    messages.append({
        "role": "assistant",
        "content": f"Done. {json.dumps(result, default=str)}",
    })
    return f"Done — {pending.summary}. Created {result['created']['id']}."
