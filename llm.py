from openai import OpenAI

_RESPONSE_SYSTEM = (
    "You are a professional customer support agent for a SaaS company. "
    "Your responses are empathetic, specific, and action-oriented. "
    "Never use filler phrases like 'I hope this email finds you well.'"
)

_BRIEFING_SYSTEM = (
    "You are a senior customer support team lead. "
    "You write clear, actionable operational briefings that help agents prioritize their shift effectively."
)


def draft_ticket_response(ticket_text: str, urgency_tier: str, urgency_score: float, api_key: str) -> str:
    """Generate a draft email response body for a single support ticket."""
    client = OpenAI(api_key=api_key)

    prompt = f"""Draft a professional response to the following customer support ticket.

Ticket: "{ticket_text}"
Priority tier: {urgency_tier} (urgency score: {urgency_score:.2f} / 1.00)

Instructions:
- Write the email body only — no subject line, no "Dear [name]", no sign-off.
- Acknowledge the customer's specific issue in the first sentence.
- State clearly what action is being taken or will be taken.
- Match urgency in tone: P0/P1 tickets should feel immediate and decisive; P2/P3 can be warmer and less urgent.
- 3–5 sentences maximum."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _RESPONSE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def generate_queue_briefing(df, api_key: str) -> str:
    """Generate a shift briefing summary for the full active ticket queue."""
    client = OpenAI(api_key=api_key)

    lines = [
        f"[{row['Tier']}] [{row.get('Category', '?')}] (score {row['Urgency Score']:.2f}) {row['Ticket']}"
        for _, row in df.iterrows()
    ]
    queue_text = "\n".join(lines)

    prompt = f"""Generate a concise shift briefing for a SaaS customer support team based on the current active ticket queue.

Active tickets ({len(df)} total):
{queue_text}

Structure your briefing as follows:
1. **Overall Load**: Urgency distribution summary (how many P0/P1/P2/P3).
2. **Immediate Attention**: Top 2–3 tickets that need to be handled first and why.
3. **Themes**: Any recurring complaint patterns worth flagging.
4. **Team Focus**: One or two sentences on where the team should direct energy this shift.

Keep the entire briefing under 200 words. Be specific — reference actual ticket content, not generic advice."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _BRIEFING_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()
