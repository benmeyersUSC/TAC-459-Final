import json
import os
import re
import requests
import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from dotenv import load_dotenv
import llm
import storage
import embeddings

# Load .env so OPENAI_API_KEY and ADMIN_PASSWORD flow into os.environ at startup
load_dotenv()

st.set_page_config(
    page_title="Intelligent Ticket Prioritizer",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

HF_REPO    = "benmeyersUSC/tac459-ticket-models"
MODELS_DIR = Path("models")


# ── Model definitions ─────────────────────────────────────────────────────────

class BertUrgencyRegressor(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.pooler_output)
        return torch.sigmoid(self.regressor(cls)).squeeze(1)


class BertCategoryClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.pooler_output)
        return self.classifier(cls)


def _ensure_file(filename, display_name=None):
    """Stream-download a file from HF Hub into models/ with a live progress bar.
    Returns immediately (no network call) if the file already exists on disk."""
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists():
        return str(dest)

    url   = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
    label = display_name or filename
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        bar   = st.progress(0, text=f"Downloading {label}...")
        done  = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    bar.progress(
                        min(done / total, 1.0),
                        text=f"{label}: {done/1e6:.0f} / {total/1e6:.0f} MB",
                    )
        bar.empty()
    return str(dest)


@st.cache_resource
def load_urgency_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertUrgencyRegressor(dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer, device


@st.cache_resource
def load_category_model(model_path, labels_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(labels_path) as f:
        label_map = json.load(f)
    label_names = [label_map[str(i)] for i in range(len(label_map))]
    model = BertCategoryClassifier(num_classes=len(label_names), dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer, label_names, device


# ── Styling helpers ───────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    'Access':                '#1565C0',
    'Administrative rights': '#6A1B9A',
    'HR Support':            '#2E7D32',
    'Hardware':              '#E65100',
    'Internal Project':      '#00838F',
    'Miscellaneous':         '#546E7A',
    'Purchase':              '#558B2F',
    'Storage':               '#4527A0',
}

TIER_STYLES = {
    'P0': ('background-color: #ff4b4b', 'white'),
    'P1': ('background-color: #ffa500', 'white'),
    'P2': ('background-color: #ffd700', 'black'),
    'P3': ('background-color: #90ee90', 'black'),
}


def tier_color(val):
    bg, fg = TIER_STYLES.get(val[:2], ('', ''))
    return f'{bg}; color: {fg}' if bg else ''


def category_color(val):
    color = CATEGORY_COLORS.get(val, '#546E7A')
    return f'background-color: {color}; color: white'


def score_to_tier(score, thresholds):
    if score >= thresholds['p0']:
        return 'P0 - Critical'
    elif score >= thresholds['p1']:
        return 'P1 - High'
    elif score >= thresholds['p2']:
        return 'P2 - Medium'
    return 'P3 - Low'


def parse_upload(uploaded_file):
    """Return (tickets, warnings). One ticket per line — file-based parsing."""
    raw = uploaded_file.read()
    warnings = []

    text = None
    for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        st.error("Could not decode the file. Please save it as UTF-8 and try again.")
        st.stop()

    tickets, skipped = [], 0
    for line in text.splitlines():
        line = line.strip().rstrip(',;|"\'').strip()
        if len(line.split()) < 3:
            if line:
                skipped += 1
            continue
        tickets.append(line)

    if skipped:
        warnings.append(
            f"{skipped} line(s) skipped — too short to be a ticket (fewer than 3 words)."
        )
    return tickets, warnings


def _parse_blocks(text):
    """Parse pasted text into tickets. Blocks are separated by blank lines.
    Within each block, single newlines are joined into one ticket."""
    blocks = re.split(r'\n\s*\n', text)
    tickets, skipped = [], 0
    for block in blocks:
        joined = ' '.join(line.strip() for line in block.splitlines() if line.strip())
        joined = joined.strip().rstrip(',;|"\'').strip()
        if len(joined.split()) < 3:
            if joined:
                skipped += 1
            continue
        tickets.append(joined)
    return tickets, skipped


def predict_urgency(text, model, tokenizer, device, max_len=128):
    text = ' '.join(text.strip().split())
    enc = tokenizer(text, truncation=True, padding='max_length',
                    max_length=max_len, return_tensors='pt')
    with torch.no_grad():
        score = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    return round(1.0 - score.item(), 4)


def predict_category(text, model, tokenizer, label_names, device, max_len=128):
    text = ' '.join(text.strip().split())
    enc = tokenizer(text, truncation=True, padding='max_length',
                    max_length=max_len, return_tensors='pt')
    with torch.no_grad():
        logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    probs = torch.softmax(logits, dim=1).squeeze()
    predicted = label_names[probs.argmax().item()]
    prob_dict = {label_names[i]: float(probs[i].item()) for i in range(len(label_names))}
    return predicted, prob_dict


def category_badge_html(category):
    color = CATEGORY_COLORS.get(category, '#546E7A')
    return (
        f"<span style='background:{color}; color:white; padding:2px 10px; "
        f"border-radius:4px; font-size:0.8em; font-weight:600'>{category}</span>"
    )


# ── CSS + session state ───────────────────────────────────────────────────────

st.markdown("""
<style>
[data-baseweb="tag"] {
    background-color: #455a64 !important;
}
[data-baseweb="tag"] span { color: white !important; }
</style>
""", unsafe_allow_html=True)

# Load resolved log from disk on startup so it survives across sessions
_persisted_log = storage.load_resolved_log()
_persisted_ticket_texts = {entry['ticket'] for entry in _persisted_log}

for key, default in [
    ('resolved_tickets', _persisted_ticket_texts),
    ('resolved_log', _persisted_log),
    ('draft', ''),
    ('briefing', ''),
    ('last_selected_ticket', None),
    ('is_admin', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## Settings")
st.sidebar.caption(f"Weights: [{HF_REPO}](https://huggingface.co/{HF_REPO})")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

with st.sidebar.expander("🔐 Admin login", expanded=False):
    if st.session_state.is_admin:
        st.success("Logged in as admin")
        if st.button("Log out", key="admin_logout"):
            st.session_state.is_admin = False
            st.rerun()
    else:
        admin_pw = st.text_input("Admin password:", type="password", key="admin_pw_input")
        if st.button("Log in", key="admin_login"):
            if admin_pw == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Incorrect password")

# Admin-only: tier thresholds (agents read the frozen values the admin set)
if st.session_state.is_admin:
    st.sidebar.markdown("### ⚙️ Tier thresholds (admin)")
    p0_thresh = st.sidebar.slider("P0 (Critical) cutoff", 0.0, 1.0,
                                  st.session_state.get('p0_thresh', 0.75), 0.05,
                                  key='p0_thresh')
    p1_thresh = st.sidebar.slider("P1 (High) cutoff", 0.0, 1.0,
                                  st.session_state.get('p1_thresh', 0.50), 0.05,
                                  key='p1_thresh')
    p2_thresh = st.sidebar.slider("P2 (Medium) cutoff", 0.0, 1.0,
                                  st.session_state.get('p2_thresh', 0.25), 0.05,
                                  key='p2_thresh')
else:
    p0_thresh = st.session_state.get('p0_thresh', 0.75)
    p1_thresh = st.session_state.get('p1_thresh', 0.50)
    p2_thresh = st.session_state.get('p2_thresh', 0.25)

thresholds = {'p0': p0_thresh, 'p1': p1_thresh, 'p2': p2_thresh}

# Admin-only: OpenAI API key (persisted in session state so agents can use AI features)
if st.session_state.is_admin:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 OpenAI API Key (admin)")
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get('api_key', os.environ.get("OPENAI_API_KEY", "")),
        help="Required for AI response drafting, queue briefings, and draft revisions. "
             "Defaults to the OPENAI_API_KEY in .env if set.",
        key="api_key_input",
        label_visibility="collapsed",
    )
    st.session_state['api_key'] = api_key
else:
    api_key = st.session_state.get('api_key', os.environ.get("OPENAI_API_KEY", ""))


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🎫 Intelligent Ticket Prioritizer")

# Input area: file upload + paste textarea, combined into one queue
uploaded_file = st.file_uploader(
    "Upload a ticket file (.txt or .csv)",
    type=["txt", "csv"],
)

with st.expander("Accepted file formats"):
    st.markdown(
        "**One ticket per line** — `.txt` or `.csv`. "
        "Trailing commas, semicolons, and quotes are stripped automatically.\n\n"
        "```\n"
        "Server is completely down, nobody can log in.\n"
        "Please update my email address when you get a chance.\n"
        "VPN drops every 5 minutes, blocking all remote work.\n"
        "```\n\n"
        "Lines with fewer than 3 words are ignored."
    )

tickets = None
parse_warnings = []

file_tickets = []
if uploaded_file is not None:
    file_tickets, parse_warnings = parse_upload(uploaded_file)

with st.expander("➕ Add additional tickets (paste below)"):
    st.caption(
        "Optional — paste extra tickets here, separated by blank lines. "
        "Each ticket can span multiple lines. They'll be appended to whatever you uploaded above."
    )
    pasted_text = st.text_area(
        "Additional tickets:",
        height=150,
        placeholder="Paste tickets, separated by blank lines. A ticket can span multiple lines.",
        key="paste_input",
        label_visibility="collapsed",
    )

pasted_tickets = []
if pasted_text.strip():
    pasted_tickets, skipped = _parse_blocks(pasted_text)
    if skipped:
        parse_warnings.append(
            f"{skipped} pasted block(s) skipped — too short to be a ticket (fewer than 3 words)."
        )

# Combine file + paste; dedupe while preserving order
if file_tickets or pasted_tickets:
    seen = set()
    combined = []
    for t in file_tickets + pasted_tickets:
        if t not in seen:
            seen.add(t)
            combined.append(t)
    tickets = combined

if not tickets:
    st.markdown(
        "<div style='text-align:center; padding:60px 20px; color:#888; "
        "border:2px dashed #ddd; border-radius:12px; margin-top:20px'>"
        "<div style='font-size:3em; margin-bottom:12px'>📭</div>"
        "<div style='font-size:1.1em; font-weight:500; margin-bottom:6px'>"
        "No tickets loaded yet</div>"
        "<div style='font-size:0.9em'>Upload a .txt or .csv file above to get started, "
        "or paste tickets directly. Each ticket will be scored for urgency "
        "and classified by category.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


if tickets:
    for w in parse_warnings:
        st.warning(w)

    with st.expander(f"Preview — {len(tickets)} tickets parsed"):
        for i, t in enumerate(tickets[:10], 1):
            st.caption(f"{i}. {t}")
        if len(tickets) > 10:
            st.caption(f"... and {len(tickets) - 10} more")

    st.info(f"{len(tickets)} tickets loaded. Running inference...")

    # Phase 1 — download weights if not on disk (live progress bar)
    urg_path    = _ensure_file("bert_urgency_refined.pth",    "Urgency model (420 MB)")
    cat_path    = _ensure_file("bert_categorizer.pth",        "Category model (420 MB)")
    labels_path = _ensure_file("bert_categorizer_labels.json")

    # Phase 2 — load into memory (cached for the session)
    with st.spinner("Initializing models..."):
        urg_model, urg_tok, urg_device = load_urgency_model(urg_path)
        cat_model, cat_tok, cat_label_names, cat_device = load_category_model(cat_path, labels_path)

    st.caption("Scoring urgency...")
    prog1 = st.progress(0)
    scores = []
    for i, ticket in enumerate(tickets):
        scores.append(predict_urgency(ticket, urg_model, urg_tok, urg_device))
        prog1.progress((i + 1) / len(tickets))

    st.caption("Classifying categories...")
    prog2 = st.progress(0)
    categories = []
    cat_probs = []
    for i, ticket in enumerate(tickets):
        predicted, probs = predict_category(ticket, cat_model, cat_tok, cat_label_names, cat_device)
        categories.append(predicted)
        cat_probs.append(probs)
        prog2.progress((i + 1) / len(tickets))

    df = pd.DataFrame({
        'Ticket': tickets,
        'Category': categories,
        'Category Probs': cat_probs,
        'Urgency Score': scores,
    })
    df['Tier'] = df['Urgency Score'].apply(lambda s: score_to_tier(s, thresholds))
    df.sort_values('Urgency Score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.insert(0, 'ID', range(1, len(df) + 1))

    st.success("Done!")

    # Remove resolved tickets from the active view
    active_df = df[~df['Ticket'].isin(st.session_state.resolved_tickets)].copy()

    # ── Filters ───────────────────────────────────────────────────────────────

    st.markdown("---")
    selected_cats = st.multiselect(
        "Filter by category:",
        options=sorted(cat_label_names),
        default=sorted(cat_label_names),
        key="cat_filter",
    )
    if selected_cats:
        active_df = active_df[active_df['Category'].isin(selected_cats)].copy()

    tier_options = ['P0 - Critical', 'P1 - High', 'P2 - Medium', 'P3 - Low']
    selected_tiers = st.multiselect(
        "Filter by urgency tier:",
        options=tier_options,
        default=tier_options,
        key="tier_filter",
    )
    if selected_tiers:
        active_df = active_df[active_df['Tier'].isin(selected_tiers)].copy()

    active_df.index = range(1, len(active_df) + 1)

    # Truncated summary column for the table (full text shown on row select)
    active_df['Summary'] = active_df['Ticket'].apply(
        lambda t: t if len(t) <= 80 else t[:77] + "..."
    )

    # ── Tier summary metrics (after both filters) ─────────────────────────────

    counts = active_df['Tier'].value_counts()
    c0, c1, c2, c3, c4 = st.columns(5)
    c0.metric("🔴 P0 Critical", int(counts.get('P0 - Critical', 0)))
    c1.metric("🟠 P1 High",     int(counts.get('P1 - High', 0)))
    c2.metric("🟡 P2 Medium",   int(counts.get('P2 - Medium', 0)))
    c3.metric("🟢 P3 Low",      int(counts.get('P3 - Low', 0)))
    c4.metric("✅ Resolved",    len(st.session_state.resolved_tickets))

    # ── Queue Briefing ────────────────────────────────────────────────────────

    st.markdown("---")
    brief_col, _ = st.columns([1, 4])
    with brief_col:
        if st.button("📋 Get Queue Briefing",
                     type="primary",
                     disabled=not api_key or active_df.empty,
                     help="Requires an OpenAI API key (set by admin)."):
            with st.spinner("Generating briefing..."):
                st.session_state.briefing = llm.generate_queue_briefing(active_df, api_key)

    if not api_key:
        st.caption("_AI features disabled — admin must set an OpenAI API key in the sidebar._")

    if st.session_state.briefing:
        with st.expander("📋 Queue Briefing", expanded=True):
            st.markdown(st.session_state.briefing)
            if st.button("Clear briefing"):
                st.session_state.briefing = ''
                st.rerun()

    # ── Ranked ticket table ───────────────────────────────────────────────────

    st.markdown("---")
    if active_df.empty:
        if st.session_state.resolved_tickets and len(df) == len(st.session_state.resolved_tickets):
            st.success("🎉 All tickets have been resolved!")
        elif not selected_cats:
            st.info("No categories selected. Pick one or more above to see tickets.")
        elif not selected_tiers:
            st.info("No tiers selected. Pick one or more above to see tickets.")
        else:
            st.info("No tickets match your filters.")
        selected_row = None
        selected_rows = []
    else:
        st.caption("Click a row to select it. Hold Cmd/Ctrl to select multiple for bulk actions.")
        display_cols = ['ID', 'Summary', 'Category', 'Urgency Score', 'Tier']
        styled = (
            active_df[display_cols]
            .style
            .map(tier_color, subset=['Tier'])
            .map(category_color, subset=['Category'])
        )
        event = st.dataframe(
            styled,
            use_container_width=True,
            on_select="rerun",
            selection_mode="multi-row",
        )

        csv = active_df.to_csv(index_label='Rank')
        st.download_button("Download ranked CSV", csv, "ranked_tickets.csv", "text/csv")

        selected_rows = event.selection.rows
        selected_row = active_df.iloc[selected_rows[-1]] if selected_rows else None

        # Bulk actions panel — appears when 2+ rows are selected
        if len(selected_rows) >= 2:
            st.markdown("---")
            selected_tickets_df = active_df.iloc[selected_rows]
            st.markdown(
                f"<div style='padding:10px 14px; background:#EFF6FF; border-left:4px solid #3B82F6; "
                f"border-radius:4px; margin-bottom:8px'>"
                f"<strong>{len(selected_rows)} tickets selected</strong> — "
                f"choose a bulk action below, or use the response panel for the most recent one."
                f"</div>",
                unsafe_allow_html=True,
            )

            bulk_col1, bulk_col2, _ = st.columns([1, 1, 2])

            with bulk_col1:
                if st.button("📝 Draft all responses",
                             type="primary",
                             disabled=not api_key,
                             help="Generate AI drafts for every selected ticket.",
                             key="bulk_draft_btn"):
                    drafts = {}
                    progress = st.progress(0, text="Drafting responses...")
                    for i, (_, row) in enumerate(selected_tickets_df.iterrows()):
                        try:
                            similar = embeddings.find_similar_tickets(
                                row['Ticket'],
                                st.session_state.resolved_log,
                                top_k=3,
                            )
                            draft = llm.draft_ticket_response(
                                row['Ticket'], row['Tier'], row['Urgency Score'],
                                row['Category'], api_key,
                                similar_examples=similar,
                            )
                            drafts[row['Ticket']] = draft
                        except Exception as e:
                            drafts[row['Ticket']] = f"[Error generating draft: {e}]"
                        progress.progress((i + 1) / len(selected_tickets_df),
                                         text=f"Drafting... ({i + 1}/{len(selected_tickets_df)})")
                    progress.empty()
                    st.session_state.bulk_drafts = drafts
                    st.success(f"✅ Drafted {len(drafts)} responses. Click any selected ticket to review and send.")

            with bulk_col2:
                if st.button("✅ Resolve all (no response)",
                             type="secondary",
                             help="Mark all selected tickets as resolved without drafting a response. "
                                  "Use for spam/duplicates.",
                             key="bulk_resolve_btn"):
                    for _, row in selected_tickets_df.iterrows():
                        entry = {
                            'ticket':   row['Ticket'],
                            'tier':     row['Tier'],
                            'category': row['Category'],
                            'score':    row['Urgency Score'],
                            'response': '',
                            'sent_to':  '',
                        }
                        storage.append_resolved(entry)
                        st.session_state.resolved_log.append(entry)
                        st.session_state.resolved_tickets.add(row['Ticket'])
                    st.session_state.draft = ''
                    st.session_state.last_selected_ticket = None
                    st.success(f"✅ Resolved {len(selected_tickets_df)} tickets.")
                    st.rerun()

    # ── Inline respond panel ──────────────────────────────────────────────────

    st.markdown("---")

    if selected_row is None:
        st.caption("👆 Click any row in the table above to select a ticket and respond.")
    else:
        tier     = selected_row['Tier']
        score    = selected_row['Urgency Score']
        category = selected_row['Category']

        # Clear draft when a different ticket is selected
        if st.session_state.last_selected_ticket != selected_row['Ticket']:
            st.session_state.draft = ''
            st.session_state.last_selected_ticket = selected_row['Ticket']

        # If a bulk draft exists for this ticket, populate it automatically
        bulk_drafts = st.session_state.get('bulk_drafts', {})
        if selected_row['Ticket'] in bulk_drafts and not st.session_state.draft:
            st.session_state.draft = bulk_drafts[selected_row['Ticket']]

        tier_bg = {'P0': '#ff4b4b', 'P1': '#ffa500', 'P2': '#ffd700', 'P3': '#90ee90'}.get(tier[:2], '#eee')
        tier_fg = 'white' if tier.startswith(('P0', 'P1')) else 'black'
        st.markdown(
            f"<div style='padding:12px 16px; border-radius:8px; background:{tier_bg}; "
            f"color:{tier_fg}; margin-bottom:12px'>"
            f"<strong>{tier}</strong> &nbsp;"
            f"{category_badge_html(category)}"
            f"&nbsp;|&nbsp; Urgency: {score:.2f}<br/><br/>"
            f"{selected_row['Ticket']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # "Why was this classified as X?" — BERT confidence across all 8 categories
        ticket_cat_probs = selected_row.get('Category Probs', {})
        if ticket_cat_probs:
            with st.expander(f"🔍 Why was this classified as {category}?", expanded=False):
                sorted_probs = sorted(ticket_cat_probs.items(), key=lambda x: x[1], reverse=True)
                top_conf = sorted_probs[0][1]
                runner_up = sorted_probs[1][1] if len(sorted_probs) > 1 else 0

                if top_conf > 0.85:
                    confidence_note = "**High confidence** — the model is sure about this classification."
                elif top_conf > 0.6:
                    confidence_note = "**Moderate confidence** — likely correct, but worth a glance."
                else:
                    margin = top_conf - runner_up
                    if margin < 0.15:
                        confidence_note = f"**Low confidence** — borderline call between {sorted_probs[0][0]} and {sorted_probs[1][0]}."
                    else:
                        confidence_note = "**Low confidence** — the model isn't very sure."
                st.markdown(confidence_note)
                st.markdown("")

                for cat_name, prob in sorted_probs:
                    bar_pct = int(prob * 100)
                    is_predicted = (cat_name == category)
                    bar_color = "#3B82F6" if is_predicted else "#CBD5E1"
                    label_weight = "600" if is_predicted else "400"
                    st.markdown(
                        f"<div style='margin-bottom:6px'>"
                        f"<div style='display:flex; justify-content:space-between; "
                        f"font-size:0.85em; font-weight:{label_weight}; margin-bottom:2px'>"
                        f"<span>{cat_name}</span><span>{prob:.2f}</span>"
                        f"</div>"
                        f"<div style='background:#F1F5F9; border-radius:4px; height:8px; overflow:hidden'>"
                        f"<div style='background:{bar_color}; height:100%; width:{bar_pct}%'></div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        action_col, resolve_col = st.columns(2)

        with action_col:
            if st.button("✍️ Draft AI Response",
                         type="primary",
                         disabled=not api_key,
                         help="Requires an OpenAI API key."):
                with st.spinner("Generating draft..."):
                    similar = embeddings.find_similar_tickets(
                        selected_row['Ticket'],
                        st.session_state.resolved_log,
                        top_k=3,
                    )
                    st.session_state.draft = llm.draft_ticket_response(
                        selected_row['Ticket'], tier, score, category, api_key,
                        similar_examples=similar,
                    )
                    st.session_state['last_retrieved'] = similar

        with resolve_col:
            if st.button("✅ Mark as Resolved", type="secondary"):
                draft_version = st.session_state.get('draft_version', 0)
                response_used = (
                    st.session_state.get(f'draft_textarea_v{draft_version}', '')
                    or st.session_state.draft
                )
                sent_to = st.session_state.get('to_email_input', '')
                entry = {
                    'ticket':   selected_row['Ticket'],
                    'tier':     tier,
                    'category': category,
                    'score':    score,
                    'response': response_used,
                    'sent_to':  sent_to,
                }
                storage.append_resolved(entry)
                st.session_state.resolved_log.append(entry)
                st.session_state.resolved_tickets.add(selected_row['Ticket'])
                st.session_state.draft = ''
                st.session_state.last_selected_ticket = None
                if 'bulk_drafts' in st.session_state:
                    st.session_state.bulk_drafts.pop(selected_row['Ticket'], None)
                st.rerun()

        if st.session_state.draft:
            # RAG status banner — always shown so it's obvious whether the vector DB was used
            retrieved = st.session_state.get('last_retrieved', [])
            if retrieved:
                top_sim = retrieved[0]['similarity']
                st.success(
                    f"📚 **RAG active** — pulled {len(retrieved)} similar past ticket(s) "
                    f"from the vector store (top similarity {top_sim:.2f}). "
                    f"Expand below to see what was retrieved."
                )
            else:
                resolved_count = sum(
                    1 for e in st.session_state.resolved_log if e.get('response')
                )
                if resolved_count == 0:
                    st.info(
                        "📚 **RAG inactive** — no resolved tickets with responses in the log yet. "
                        "Resolve a few drafts to start building the vector store."
                    )
                else:
                    st.info(
                        f"📚 **RAG ran** — searched {resolved_count} past ticket(s) "
                        f"but none scored above the 0.4 similarity threshold."
                    )

            if retrieved:
                with st.expander(f"📚 Show {len(retrieved)} retrieved tickets", expanded=True):
                    for i, ex in enumerate(retrieved, 1):
                        st.markdown(
                            f"<div style='padding:8px 12px; background:#F8FAFC; "
                            f"border-left:3px solid #3B82F6; border-radius:4px; margin-bottom:8px'>"
                            f"<div style='font-size:0.85em; color:#64748B; margin-bottom:4px'>"
                            f"Past ticket #{i} — similarity {ex['similarity']:.2f} · "
                            f"{ex.get('category', 'Unknown')} · {ex.get('tier', 'Unknown')}"
                            f"</div>"
                            f"<div style='font-size:0.9em; margin-bottom:6px'>"
                            f"<strong>Ticket:</strong> {ex['ticket']}"
                            f"</div>"
                            f"<div style='font-size:0.9em; color:#475569'>"
                            f"<strong>Our response:</strong> {ex['response']}"
                            f"</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("**Edit response before sending:**")
            draft_version = st.session_state.get('draft_version', 0)
            st.text_area(
                label="Response body",
                value=st.session_state.draft,
                height=160,
                label_visibility="collapsed",
                key=f"draft_textarea_v{draft_version}",
            )

            # Note-driven revision
            with st.expander("✏️ Revise with quick notes"):
                st.caption(
                    "Type short notes about what you want changed — facts, status updates, "
                    "next steps. The AI will rewrite the draft above to incorporate them. "
                    "Examples: \"team is on it, ETA 2 hrs\" / \"I confirmed the issue, escalating to engineering\" / "
                    "\"already resolved, just confirming with customer\"."
                )
                agent_notes = st.text_input(
                    "Your notes:",
                    placeholder="e.g. confirmed issue, eng team has fix, ETA 30 min",
                    key="agent_notes_input",
                )
                revise_col, _ = st.columns([1, 3])
                with revise_col:
                    if st.button("🔄 Revise draft",
                                 type="primary",
                                 disabled=not (api_key and agent_notes.strip()),
                                 help="Requires an OpenAI API key and notes."):
                        with st.spinner("Revising..."):
                            current_draft = (
                                st.session_state.get(f'draft_textarea_v{draft_version}', '')
                                or st.session_state.draft
                            )
                            revised = llm.revise_ticket_response(
                                original_draft=current_draft,
                                agent_notes=agent_notes,
                                original_ticket=selected_row['Ticket'],
                                urgency_tier=tier,
                                category=category,
                                api_key=api_key,
                            )
                            st.session_state.draft = revised
                            st.session_state['draft_version'] = draft_version + 1
                            st.rerun()

            st.text_input("Recipient email address:", key="to_email_input")
            to_email = st.session_state.get('to_email_input', '')

            send_col, _ = st.columns([1, 3])
            with send_col:
                if st.button("📤 Send Email", type="primary", disabled=not to_email.strip()):
                    st.success(f"Email sent to **{to_email}**.")

    # ── Resolved Tickets Log (admin-only) ─────────────────────────────────────

    if st.session_state.is_admin and st.session_state.resolved_log:
        st.markdown("---")
        log = st.session_state.resolved_log
        with st.expander(f"✅ Resolved Tickets ({len(log)}) — admin view", expanded=False):
            for entry in reversed(log):
                t_bg, t_fg = [s.replace('background-color: ', '') for s in
                              TIER_STYLES.get(entry['tier'][:2], ('#eee', 'black'))]
                cat_label = entry.get('category', '')
                tier_badge = (
                    f"<span style='background:{t_bg}; color:{t_fg}; padding:2px 8px; "
                    f"border-radius:4px; font-size:0.8em; font-weight:bold'>{entry['tier']}</span>"
                )
                with st.container():
                    st.markdown(
                        f"{tier_badge} &nbsp;{category_badge_html(cat_label)}&nbsp; "
                        f"<span style='font-size:0.95em'>{entry['ticket']}</span>",
                        unsafe_allow_html=True,
                    )
                    detail_cols = st.columns([2, 1])
                    with detail_cols[0]:
                        if entry['response']:
                            st.markdown("**Response sent:**")
                            st.markdown(
                                f"<div style='background:#f8f8f8; border-left:3px solid #ccc; "
                                f"padding:8px 12px; border-radius:4px; font-size:0.9em; "
                                f"white-space:pre-wrap'>{entry['response']}</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("_No response drafted._")
                    with detail_cols[1]:
                        st.markdown(f"**Urgency score:** {entry['score']:.2f}")
                        if entry['sent_to']:
                            st.markdown(f"**Sent to:** {entry['sent_to']}")
                        else:
                            st.caption("_Not emailed._")
                    st.markdown("<hr style='margin:8px 0; border-color:#eee'>", unsafe_allow_html=True)

            if st.button("🗑️ Clear all resolved history", type="secondary"):
                storage.clear_resolved_log()
                st.session_state.resolved_log = []
                st.session_state.resolved_tickets = set()
                st.rerun()
