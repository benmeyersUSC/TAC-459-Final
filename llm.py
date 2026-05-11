from openai import OpenAI

_RESPONSE_SYSTEM = (
    "You are a customer support agent for a SaaS company. You write replies that are "
    "ready to send: empathetic, specific, and grounded in what is actually known. "
    "You never invent technical details, root causes, timelines, or commitments. "
    "When you don't know something, you say what will happen next instead of "
    "what has already been done. Never use filler like 'I hope this finds you well.' "
    "Distinguish between common acronyms (ETA, VPN, SSO — keep as-is) "
    "and technical jargon the customer wouldn't recognize (translate to plain language). "
    "Don't expand acronyms a customer would understand; do clarify the ones they wouldn't."
)

_BRIEFING_SYSTEM = (
    "You are helping a support agent triage their own ticket queue at the start "
    "of a shift. You write like a sharp colleague leaning over their shoulder — "
    "concrete, specific, no corporate filler. You reference real tickets by their "
    "content, not by ID."
)

# ── Category-specific guidance ────────────────────────────────────────────────
# Short hints that nudge tone and shape per category. Kept terse on purpose;
# the model already knows what these categories mean.
_CATEGORY_GUIDANCE = {
    "Access":                "Login, permissions, locked accounts. Tone: reassuring, security-aware.",
    "Administrative rights": "Privilege escalation requests. Tone: process-oriented; mention approval if relevant.",
    "HR Support":            "People issues — payroll, benefits, leave. Tone: warm, discreet, never dismissive.",
    "Hardware":              "Physical device problems. Tone: practical; ask for specifics if missing.",
    "Internal Project":      "Internal tooling or project work. Tone: collegial, less customer-service.",
    "Miscellaneous":         "Doesn't fit a clear bucket. Tone: ask a clarifying question if the ticket is vague.",
    "Purchase":              "Procurement, licenses, billing. Tone: clear about next steps and approvals.",
    "Storage":               "Quota, file access, capacity. Tone: practical; mention quota or cleanup options.",
}

 
def _category_hint(category: str) -> str:
    return _CATEGORY_GUIDANCE.get(category, "")


def draft_ticket_response(ticket_text: str,urgency_tier: str,urgency_score: float,category: str,api_key: str,similar_examples: list = None) -> str:
    client = OpenAI(api_key=api_key)
    cat_hint = _category_hint(category)
    cat_line = f"\nCategory guidance: {cat_hint}" if cat_hint else ""

    if urgency_tier.startswith(("P0", "P1")):
        tone_note = "Tone: decisive and calm. Match the urgency without sounding panicked. No over-apologizing."
    else:
        tone_note = "Tone: warm and efficient. Don't manufacture urgency the ticket doesn't have."

    examples_block = ""
    if similar_examples:
        examples_lines = ["\nFor reference, here's how we've responded to similar tickets in the past:\n"]
        for i, ex in enumerate(similar_examples, 1):
            examples_lines.append(
                f"--- Past ticket {i} (similarity {ex.get('similarity', 0):.2f}) ---\n"
                f"Ticket: \"{ex['ticket']}\"\n"
                f"Our response: \"{ex['response']}\"\n"
            )
        examples_lines.append(
            "\nMatch the style, tone, and level of specificity of these past responses. "
            "Don't copy them — adapt the same approach to the new ticket below."
        )
        examples_block = "\n".join(examples_lines)

    prompt = f"""Draft the body of an email response to this support ticket.{examples_block}

New ticket: "{ticket_text}"
Category: {category}
Priority tier: {urgency_tier} (urgency score: {urgency_score:.2f} / 1.00){cat_line}

Output rules:
- Email body only. No subject line, no salutation, no sign-off.
- Plain prose. No bullet lists unless the ticket itself asks for steps.
- 3–5 sentences.

Style:
- Acknowledge the specific issue in the first sentence — show you read it.
- State the concrete next step (what the customer should do, or what support will do next). Keep commitments safe: "we're investigating" is safe; "we've identified the bug" is not.
- {tone_note}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _RESPONSE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()
 


def generate_queue_briefing(df, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    # Pre-compute distribution so the model doesn't have to count.
    tier_counts = df['Tier'].value_counts().to_dict()
    cat_counts = df['Category'].value_counts().to_dict() if 'Category' in df.columns else {}
    tier_summary = ", ".join(f"{k}: {v}" for k, v in sorted(tier_counts.items()))
    cat_summary = ", ".join(f"{k}: {v}" for k, v in cat_counts.items())
    lines = [
        f"#{row['ID']} [{row['Tier']}] [{row.get('Category', '?')}] (score {row['Urgency Score']:.2f}) {row['Ticket']}"
        for _, row in df.iterrows()
    ]
    queue_text = "\n".join(lines)
 
    prompt = f"""Brief the agent on their current queue. They're about to start triage.
 
Queue size: {len(df)}
Tier distribution: {tier_summary}
Category distribution: {cat_summary}
 
Tickets:
{queue_text}
 
Write the briefing in this shape:
 
**The shape of your queue.** One or two sentences on the overall load and what dominates it (urgency-wise or category-wise). Don't just restate the counts — interpret them.
 
**Start here.** Name the 2–3 specific tickets that should go first. Reference each by ID with a short description of the issue (5–10 words), then say *why* in half a sentence. Format: "#4 (P0, server outage) — affecting all users, blocks everything else."
 
**Patterns to flag.** If two or more tickets share a likely root cause, theme, or affected system, surface that. If nothing obvious repeats, say so in one line and move on — don't manufacture a pattern.
 
**One thing to watch.** A single sentence on something easy to miss: a P2 that's actually time-sensitive, a category that's quietly piling up, an ambiguous ticket that needs clarification before work starts.
 
Hard rules:
- Reference tickets by ID with a short description, never by quoting ticket text.
- Under 220 words total.
- No corporate filler. No "leverage," no "stakeholders," no "ensure optimal outcomes."
- If the queue is small (under 5 tickets), compress to two sections instead of four — don't pad."""
 
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _BRIEFING_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=550,
    )
    return resp.choices[0].message.content.strip()

# CHANGE: new function for note-driven response revision — agent types quick notes, LLM rewrites the draft
def revise_ticket_response(
    original_draft: str,
    agent_notes: str,
    original_ticket: str,
    urgency_tier: str,
    category: str,
    api_key: str,
) -> str:
    """Revise an existing draft response based on the agent's quick notes.
    The agent supplies natural-language instructions (e.g. 'team is on it, ETA 2 hours');
    the model rewrites the draft to incorporate them while preserving tone."""
    client = OpenAI(api_key=api_key)

    cat_hint = _category_hint(category)
    cat_line = f"\nCategory guidance: {cat_hint}" if cat_hint else ""

    if urgency_tier.startswith(("P0", "P1")):
        tone_note = "Tone: decisive and calm. Match the urgency without sounding panicked."
    else:
        tone_note = "Tone: warm and efficient."

    prompt = f"""You are revising an existing draft response based on quick notes from the support agent who reviewed it.

Original ticket: "{original_ticket}"
Category: {category}
Priority tier: {urgency_tier}{cat_line}

Current draft:
\"\"\"
{original_draft}
\"\"\"

Agent's notes (use these to revise the draft — they may be telegraphic or shorthand):
\"\"\"
{agent_notes}
\"\"\"

Output rules:
- Return the revised email body only. No subject, no salutation, no sign-off.
- Plain prose. 3–5 sentences.
- Incorporate every concrete fact in the agent's notes (timelines, status updates, who's working on it, what was confirmed).
- Treat the agent's notes as authoritative — they have context you don't. If they say "team is on it, ETA 2 hours," include that as a commitment.
- Common acronyms (ETA, VPN, SSO, FAQ, etc.) — keep as acronyms; expanding them sounds awkward.
- Technical/internal acronyms (specific error codes, internal tool names, jargon the customer wouldn't recognize) — translate into plain language the customer can understand. The agent's notes use shorthand for speed; the customer-facing reply shouldn't.
- Preserve the structure and tone of the original draft where possible. Don't rewrite from scratch.
- {tone_note}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _RESPONSE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()
