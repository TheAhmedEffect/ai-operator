# AI Operator — CRM chat assistant

A working slice of an AI Operator for a CRM: a command-line chat interface backed by
a real language model, with three Ask tools and one Run a task tool operating on an
invented dataset. Writes are held at a confirmation gate until a human approves them.

Built for the NexCell Autumn Cohort AI Challenge, September 2026.

---

## Running it

```bash
pip install -r requirements.txt

# CMD
set OPENAI_API_KEY=sk-...
# PowerShell
$env:OPENAI_API_KEY='sk-...'
# bash
export OPENAI_API_KEY=sk-...

python main.py
```

| Command | What it does |
|---|---|
| `python main.py` | The assistant — interactive chat |
| `python main.py --data` | Print the mock CRM |
| `python main.py --ask` | Four fixed questions exercising the read-only tools |
| `python main.py --write` | Three scenarios exercising the confirmation gate |
| `pytest -v` | 38 regression tests, ~0.1s — **no API key or model call needed** |

Model defaults to `gpt-4o-mini`; override with `AI_OPERATOR_MODEL`.

### Tests

`test_operator.py` runs offline and deterministically. It asserts the guarantees
rather than trusting them: that Ask tools never mutate, that proposing a write
executes nothing, that a cancelled write leaves the data untouched, that a confirmed
write creates exactly one record matching what was approved, that unknown lead ids
and invalid arguments are refused before any confirmation prompt, and that a held
write's payload cannot be altered between the question and the action.

It does **not** test whether the model picks the right tool — that needs a live call.
The split is deliberate: everything that can be asserted deterministically is, and
the rest is documented as a live-only check.

### A session that exercises everything

```
How many leads do I have?
Tell me about the one interested in Hackney
Create a task to call her on Monday
no
Actually yes, create that task
yes
tasks
```

This is the transcript I tested against. It proves four things: an Ask question never
reaches a write tool; "her" resolves across turns from conversation history; a
cancelled write changes nothing; and a confirmed write creates exactly one record,
linked to a lead id the model had actually seen in tool output.

---

## What I built

**Model:** OpenAI `gpt-4o-mini`, via the function-calling API.

I used plain function calling rather than LangChain or LangGraph, deliberately. I'd
expect to work in LangGraph on the real product, but for a one-week prototype a
framework would have hidden the exact thing this challenge is about — the boundary
between what the model decides and what my code does. With raw function calling that
boundary is a line of code you can point at, and I'd rather be able to point at it.

**Tools**

| Tool | Mode | Does |
|---|---|---|
| `get_insights()` | Ask | New leads this week, unassigned, never contacted, totals, open tasks |
| `search_leads(query)` | Ask | Keyword search by name, status or property interest |
| `list_leads(sort_by, order, limit, status)` | Ask | Ordered listing — newest, oldest, least recently contacted |
| `create_task(title, due, related_to)` | **Run a task** | Creates a task — **held for confirmation** |

`search_leads` and `list_leads` are deliberately separate rather than one tool with
sort parameters bolted on. Keyword search and ordered enumeration are different
questions, and each schema can then say clearly when it is the right one to reach for.

**Data:** five leads, two seeded tasks, in memory in `data.py`. All invented. No real
client, portal or production data anywhere. `TODAY` is pinned to a fixed date so
`get_insights()` is deterministic and a reviewer can check the arithmetic by hand.

**Files**

```
data.py    the mock CRM
tools.py   tool implementations, validation, and the confirmation gate
llm.py     the OpenAI conversation; knows nothing about what tools do
main.py    the chat loop and the phase demos
```

---

## The safety pattern

### Rule 1 — model talks, code operates

This is a module boundary, not a convention. `tools.py` can change data but never
speaks to the model. `llm.py` speaks to the model but cannot change data — it only
knows how to pass a name and arguments to `tools.execute_read()` and hand the result
back. Neither module can do the other's job.

### Rule 2 — confirm before you act

The obvious implementation is an `if` before executing: remember to ask. I didn't
build it that way, because a guarantee that depends on remembering isn't a guarantee.

Tools live in two separate registries:

```python
READ_TOOLS  = {"get_insights": ..., "search_leads": ...}
WRITE_TOOLS = {"create_task": ...}
```

and exactly one function in the project can call anything in `WRITE_TOOLS`:

```python
def execute_confirmed(pending: PendingWrite):
    return WRITE_TOOLS[pending.tool](**pending.args)
```

It requires a `PendingWrite`. A `PendingWrite` is only constructible by
`propose_write()`, which validates arguments and returns **without executing
anything**. The chat loop then refuses to proceed until a human answers.

So **forgetting to confirm is not a mistake this program can make**. There is no code
path from a model response to a mutation that doesn't pass through a person. That
holds even if someone adds a second write tool next month without reading this file,
which a well-placed `if` statement would not.

`PendingWrite` is a frozen dataclass. The arguments shown to the user are the exact
arguments executed — the model gets no second turn to alter the payload between the
question being asked and the action being taken.

One more detail: **validation runs before the confirmation prompt**, not after. If the
model proposes a task attached to a lead we don't hold, `validate_create_task` rejects
it and the user is never shown a confirmation for something that was going to fail.
Rule 4 fires before rule 2 gets a turn.

### Rule 3 — never invent what you don't have

Enforced in tool output rather than left to the prompt. `search_leads` returns an
explicit `match_count: 0` with an empty list — a positive statement that we looked and
found nothing, not silence for the model to fill. `validate_create_task` refuses any
`related_to` that isn't in the dataset, so the model cannot attach work to a lead it
imagined.

Tested: asking about a lead named "Michael Fenwick" returns *"We have no lead named
Michael Fenwick in the CRM"* rather than a plausible invented status.

### Rule 4 — fail loud

Every failure raises `ToolError` with a readable reason that goes back to the model to
relay. Unknown tool names raise rather than being ignored, so a hallucinated tool is
visible. The turn loop is bounded at five steps and reports that it gave up rather
than spinning.

### Why "model talks, server operates" matters

A language model is a good proposer and a bad executor. It's good at reading "remind me
to call her Monday" and working out that this means `create_task` with a title, a date
and a lead id. It has no way of knowing whether that lead exists, whether "Monday"
resolved correctly, or whether this user is allowed to write to this record — and it
will produce a confident, well-formed call regardless.

The failure that matters in a CRM isn't the model being wrong. It's the model being
wrong **and something changing anyway**, or the model being wrong **and sounding
right**. The first is prevented structurally, by the registry split. The second is why
rules 3 and 4 live in tool output rather than in the prompt.

One caveat I'd rather state than have found: the registry split guarantees that **no
data changes without a human yes**. It does not guarantee that the model *accurately
reports* what happened — bug 7 below is an instance of exactly that gap, where the
assistant can describe a write it never proposed. Those are separate properties. One is
enforced in code; the other currently rests on the prompt, and I say so in the write-up
rather than claiming a guarantee I don't have.

I've built this pattern before, in an asynchronous alert-routing agent where two of the
four correctness invariants were enforced by database constraints rather than by
application checks — a duplicate notification was a write the schema refused, not a bug
I hoped to catch in review. Same instinct: if a rule matters, make breaking it
structurally impossible rather than merely discouraged.

---

## Two bugs I found while testing, and what they changed

Both are more interesting than the features.

**1. I assumed a single tool step, which silently dropped writes.**

Asked *"create a task to call Aisha Khan on Friday"*, the model sensibly called
`search_leads` first to find her id. My code executed the search, made one follow-up
call, and read `message.content` — which is `None` when the model returns another tool
call rather than text. The proposed `create_task` was discarded and the user got
`(no reply)`.

A silently dropped write is exactly the failure mode this project exists to prevent, and
it was in my own control flow. The fix was to make the turn a bounded loop: call the
model, run whatever read tools it asks for, feed results back, let it decide again,
break the moment a write is proposed.

**2. The confirmation prompt was unreadable, so the gate caught nothing.**

The model resolved "Friday" to `2026-09-15` — a Tuesday. The validator accepted it
because it checks the *format* is a valid date, not that the date means what was asked.
And the prompt said *"due 2026-09-15"*, which is technically accurate and completely
unreviewable at a glance.

The gate was working perfectly and protecting nothing. A confirmation is only a safety
net if a human can actually read it, so the summary now renders
*"due Tuesday 15 September 2026"* — which makes a wrong weekday obvious instead of
invisible. The model is also now told today's weekday, which stopped it happening again.

The general lesson: a safety mechanism that a human can't meaningfully evaluate is
theatre. Worth checking the other three rules against that standard too.

**3. The gate then worked, and I approved the wrong thing anyway.**

Testing again after that fix, I asked for a task due "Monday". The model produced
`2026-09-13` and the prompt read *"due **Sunday** 13 September 2026"* — the error was
now stated plainly, in English, immediately above the confirmation. I typed yes.

Two findings in one. The fix worked: the same mistake that was invisible the day before
was now impossible to miss. And it made no difference, because I didn't read it. That is
the honest limit of confirm-before-write as a pattern — it converts a silent failure into
a visible one, which is a real improvement, but a human who rubber-stamps prompts is
still a human who rubber-stamps prompts.

The same prompt had produced the correct Monday on the previous run, so the model is also
simply inconsistent at date arithmetic. Rather than write a stricter prompt, I removed the
arithmetic: the `due` field's schema now embeds a literal table of the next fortnight with
weekday names, and tells the model not to calculate. Date maths is something models are
unreliable at and computers are perfect at — the fix is to stop asking.

That moves the failure mode from "confidently wrong date" to "date isn't in my table, so
I'll ask you for one", which is the right direction: from silent error to loud question.

**4. A right answer for the wrong reason.**

Asked *"tell me about the most recent lead"*, the assistant named the correct person. It
had no tool that could order anything, so it called `search_leads("new")` and reasoned
over the three results — and the newest lead happened to have status `new`, so the answer
came out right. A lead created yesterday and already contacted would have been missed
entirely, silently, with the same confident phrasing.

This one is easy to miss precisely because the output looks correct. It only shows up if
you read the tool trace rather than the reply, which is a good argument for printing the
trace at all.

The fix was a new `list_leads` tool with explicit sorting, not a better prompt — the model
was improvising around a missing capability, and no amount of instruction fixes a gap in
the tool surface. Its schema also tells the model plainly not to derive ordering from a
search result.

What I take from this one: **evaluate the tool calls, not just the answers.** Three of the
four bugs here produced output that looked entirely reasonable.

**5. The assistant did not know who it was talking to.**

Asked *"who's assigned to me?"*, it listed every lead and replied: *"There are no leads
currently assigned to you. The leads assigned are all under Zisan Ahmed."*

The model did nothing wrong. There was no notion of a current user anywhere in the
system, so "me" was unresolvable, and it accurately reported that some third party held
those leads. A missing **concept**, not a missing tool — and a more fundamental one, since
"my leads", "my tasks" and every permission check in a real CRM depend on identity.

Fixed by adding `CURRENT_USER` to the data layer, surfacing it in the system prompt, and
adding an `assigned_to` filter to `list_leads` so ownership questions are answered by a
filter rather than by the model reading a full list.

The same change fixed something quieter. Asked *"who haven't we contacted?"*, the tool had
returned all five leads and the model picked out the three with no contact date. Correct,
but that is the model doing filtering the tool should do — fine on five records, slow and
unreliable on five hundred, and wrong without announcing it. `list_leads` now takes a
`contacted` filter, and the schema tells the model to use it rather than eyeball a list.

The pattern across bugs 4 and 5 is the same: **when the tool surface has a hole, the model
papers over it with reasoning, and the output looks fine.** The reasoning is often correct
on a small dataset, which is exactly what makes it dangerous.

**6. Fixing bug 5 caused bug 6.**

Having told the model that "me" and "my" mean the current user, I asked *"who haven't we
contacted yet?"* — and got one lead back instead of three. It had applied the
`assigned_to` filter to a question about the whole agency.

The clearest way to see it is two adjacent turns that produced **identical tool calls**:

```
"Who haven't we contacted yet?"            -> assigned_to='Zisan Ahmed', contacted=False
"Which of my leads haven't I contacted?"   -> assigned_to='Zisan Ahmed', contacted=False
```

The second is right. The first is wrong — and before I added identity context, the first
had been right. My fix made the model over-eager to scope to the current user, and it
could no longer tell "we" from "I".

What makes this the worst of the six: the answer was **shorter but not obviously wrong**.
A missing lead doesn't announce itself. Rule 3 stops the assistant inventing data; nothing
was stopping it quietly omitting some.

The prompt now distinguishes first-person singular from first-person plural explicitly,
and tells the model to ask rather than guess when the scope is ambiguous. But the honest
lesson is about the fix, not the bug: **narrowing a filter is a silent failure mode, and a
prompt change can introduce one as easily as a code change.** This is precisely the kind
of regression an evaluation harness catches and manual testing does not — I found it by
chance, re-running an old question after an unrelated change.

There is also a third instance of the bug-4 pattern here. Asked *"who's assigned to
Dave?"*, the model called `search_leads("Dave")`, got zero matches, and answered that no
leads are assigned to anyone named Dave. True, but not established: `search_leads` matches
on name, status and property interest and **not** on assignee, so a zero result there says
nothing about ownership. Right answer, invalid reasoning, third time. That tool's
description now states what it does not cover, not just what it does.

**7. The model stopped calling the write tool and started narrating instead.**

Late in testing, two of three write scenarios produced no tool call at all. The model
described what it would do and asked for confirmation *in prose*:

> "I will create a task titled ... due tomorrow. Please confirm this task creation."

No proposal. Nothing held. Nothing pending. And if the user answers "yes", the model may
reply "done" — having called nothing, changed nothing, and told them otherwise.

This one reframed how I think about the whole design, because of what it does and does not
break. **The safety property held perfectly**: nothing mutated without approval, because
nothing mutated at all. What broke was **honesty** — the assistant's account of what
happened.

Those are two different guarantees and I had been treating them as one. The registry split
makes it structurally impossible to change data without a human yes. It does nothing
whatsoever to stop the model *claiming* it changed data. A phantom write leaves the CRM
correct and the user wrong.

The cause was mine. Fixing bug 3 had added "say you need an explicit date rather than
guessing" to the `due` schema — an instruction to answer in prose, which bled into how the
model handled writes generally. **The second time a fix caused a regression**, and the
second time I found it by re-running something that used to work.

I've rewritten the Run-a-task section of the system prompt to be explicit that describing a
change accomplishes nothing, that asking for permission in prose leaves nothing pending,
and that claiming a record exists without a tool result saying so is a false statement
about the user's data. The date instruction is reworded to not invite prose.

But I want to be straight about what that fix is: **it is a prompt change, in a project
whose argument is that prompts are weaker than structure.** The schema already said "do not
ask the user for permission yourself" and the model ignored it. I have made the instruction
harder to miss; I have not made it impossible to ignore.

The structural version — and what I would build next — is a post-response check: if an
assistant turn claims a mutation occurred and no tool result in that turn supports it, that
reply never reaches the user. That moves the guarantee from "the model was told not to"
into code, where the other three rules already live. It needs care to avoid false positives
on ordinary phrasing, which is why I have not bolted it on hours before a deadline.

---

## What I'd build next

**Immediately — the remaining tools.** `get_calendar(range)` and
`update_lead_status(lead_id, status)`. The second is the interesting one: it's the first
real test of whether the confirmation gate generalises. It should need *no changes* to
`main.py` or `llm.py` — register it in `WRITE_TOOLS`, add a validator and a summariser,
done. If it needs more than that, the abstraction was wrong and I'd want to know now.

**Then, in rough order of how much they'd change the design:**

- **Identity and permissions.** Auth is explicitly out of scope for this challenge, but
  bug 5 showed where it plugs in, so it is worth being precise about two things that get
  conflated.

  **Authentication — who are you?** Currently a hardcoded `CURRENT_USER` in `data.py`.
  In production that comes from the session, and the constant is the seam: nothing else
  in the codebase reads the current user from anywhere else, so replacing it is a
  one-line change rather than a refactor.

  **Authorisation — what may you do?** This is the more interesting half and it does not
  exist at all. Right now any user can propose any write on any record. Real scoping
  means a check *before* the human confirmation, inside `propose_write()`: is this person
  permitted to propose this, on this record, at all?

  The ordering matters. Asking "do you approve?" before asking "are you allowed?" shows
  users a confirmation for actions they could never perform, and trains them to click
  through prompts — which, as bug 3 demonstrated, they already do. The permission check
  belongs next to the validator, in code, for exactly the reason the confirm gate does:
  a rule that matters should be structurally impossible to skip rather than merely
  documented.

  A toy password check would have been worse than none here — it looks like security
  without being security, which is the opposite of what the rest of this design argues
  for.
- **Audit logging.** Every proposal, confirmation and refusal, with arguments as
  confirmed, append-only. The interesting log line isn't the successful write — it's the
  one where someone said no, or where validation refused before anyone was asked.
- **Confirmations that survive a restart.** A held `PendingWrite` lives in memory. In a
  real deployment someone might confirm minutes later on another device, and what
  executes must be exactly what they were shown.
- **Extend the harness to the model layer.** `test_operator.py` covers everything that can
  be asserted without a live call — the safety guarantees, the filters, the validation
  refusals, the immutability of a held payload. What it cannot cover is tool *selection*:
  whether "who haven't we contacted" produces an unscoped call and "which of my leads"
  produces a scoped one. That was bug 6, it was a prompt change that caused it, and I
  found it by chance re-running an old question rather than by testing.

  The next step is a small recorded-fixture suite: a set of prompts, the tool-call
  sequence each should produce, run against the live model on a schedule rather than on
  every commit. Prompts are code; the half of them I can test deterministically now is,
  and the other half needs a different mechanism rather than no mechanism.
- **Then** voice input, attachments, prompt library. None of them change whether the
  thing is safe.

---

## Known limits

- Confirmation is yes/no on the next turn. It doesn't handle *"yes but move it to
  Tuesday"* — a real version would let you amend a held proposal.
- One pending write at a time, by design. Batching needs a different confirmation UX
  and I'd rather not guess at it.
- Relative dates are handled by giving the model a lookup table rather than letting it
  calculate, which fixed the wrong-weekday bug — but it only covers a fortnight ahead.
  Anything further out needs an explicit date. Proper parsing in code is the real answer.
- `list_leads` covers ordering by four fields. Anything else — grouping, counting by
  status, date ranges — would need either more parameters or more tools, and I'd want to
  see which questions people actually ask before guessing which.
- Replies sometimes come back as bullet lists despite the prompt asking for plain
  sentences. Cosmetic, and I'd rather under-constrain the phrasing than over-engineer it.
- `data.TODAY` is pinned so results are reproducible. Swap for `date.today()` to let it
  drift.
