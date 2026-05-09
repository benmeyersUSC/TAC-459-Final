import json
import os
import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download
import torch.nn as nn
import llm

HF_REPO = "benmeyersUSC/tac459-ticket-models"


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


@st.cache_resource
def load_urgency_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_path = hf_hub_download(repo_id=HF_REPO, filename="bert_urgency_refined.pth")
    model = BertUrgencyRegressor(dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer, device


@st.cache_resource
def load_category_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_path  = hf_hub_download(repo_id=HF_REPO, filename="bert_categorizer.pth")
    labels_path = hf_hub_download(repo_id=HF_REPO, filename="bert_categorizer_labels.json")
    with open(labels_path) as f:
        label_map = json.load(f)
    label_names = [label_map[str(i)] for i in range(len(label_map))]
    model = BertCategoryClassifier(num_classes=len(label_names), dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer, label_names, device


# ── Styling helpers ───────────────────────────────────────────────────────────

# Distinct color per category — used in table cells and badges
CATEGORY_COLORS = {
    'Access':                '#1565C0',   # deep blue
    'Administrative rights': '#6A1B9A',   # purple
    'HR Support':            '#2E7D32',   # green
    'Hardware':              '#E65100',   # deep orange
    'Internal Project':      '#00838F',   # cyan/teal
    'Miscellaneous':         '#546E7A',   # blue-grey
    'Purchase':              '#558B2F',   # light green
    'Storage':               '#4527A0',   # deep indigo
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
    return label_names[probs.argmax().item()]


def category_badge_html(category):
    color = CATEGORY_COLORS.get(category, '#546E7A')
    return (
        f"<span style='background:{color}; color:white; padding:2px 10px; "
        f"border-radius:4px; font-size:0.8em; font-weight:600'>{category}</span>"
    )


# ── Page config & session state ───────────────────────────────────────────────

st.set_page_config(page_title="Ticket Prioritizer", layout="wide")

# Override Streamlit's default red multiselect tags to neutral blue-grey
st.markdown("""
<style>
[data-baseweb="tag"] {
    background-color: #455a64 !important;
}
[data-baseweb="tag"] span { color: white !important; }
</style>
""", unsafe_allow_html=True)

for key, default in [
    ('resolved_tickets', set()),
    ('resolved_log', []),
    ('draft', ''),
    ('briefing', ''),
    ('last_selected_ticket', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## Models")
st.sidebar.caption(f"Weights: [{HF_REPO}](https://huggingface.co/{HF_REPO})")

st.sidebar.markdown("### Tier thresholds")
p0_thresh = st.sidebar.slider("P0 (Critical) cutoff", 0.0, 1.0, 0.75, 0.05)
p1_thresh = st.sidebar.slider("P1 (High) cutoff",     0.0, 1.0, 0.50, 0.05)
p2_thresh = st.sidebar.slider("P2 (Medium) cutoff",   0.0, 1.0, 0.25, 0.05)
thresholds = {'p0': p0_thresh, 'p1': p1_thresh, 'p2': p2_thresh}

st.sidebar.markdown("---")
st.sidebar.markdown("## AI Assistant")
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", ""),
    help="Required for AI response drafting and queue briefings.",
)


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🎫 Intelligent Ticket Prioritizer")
st.caption("Upload a .txt file with one ticket per line. Ranked by urgency, classified by category.")

uploaded_file = st.file_uploader("Drop your tickets file here", type=["txt"])

if uploaded_file is not None:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    tickets = [l.strip() for l in lines if l.strip()]

    st.info(f"Loaded {len(tickets)} tickets. Running inference...")

    with st.spinner("Loading models..."):
        urg_model, urg_tok, urg_device = load_urgency_model()
        cat_model, cat_tok, cat_label_names, cat_device = load_category_model()

    st.caption("Scoring urgency...")
    prog1 = st.progress(0)
    scores = []
    for i, ticket in enumerate(tickets):
        scores.append(predict_urgency(ticket, urg_model, urg_tok, urg_device))
        prog1.progress((i + 1) / len(tickets))

    st.caption("Classifying categories...")
    prog2 = st.progress(0)
    categories = []
    for i, ticket in enumerate(tickets):
        categories.append(predict_category(ticket, cat_model, cat_tok, cat_label_names, cat_device))
        prog2.progress((i + 1) / len(tickets))

    df = pd.DataFrame({'Ticket': tickets, 'Category': categories, 'Urgency Score': scores})
    df['Tier'] = df['Urgency Score'].apply(lambda s: score_to_tier(s, thresholds))
    df.sort_values('Urgency Score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1

    st.success("Done!")

    # Remove resolved tickets
    active_df = df[~df['Ticket'].isin(st.session_state.resolved_tickets)].copy()

    # ── Category filter ───────────────────────────────────────────────────────

    st.markdown("---")
    selected_cats = st.multiselect(
        "Filter by category:",
        options=sorted(cat_label_names),
        default=sorted(cat_label_names),
        key="cat_filter",
    )
    if selected_cats:
        active_df = active_df[active_df['Category'].isin(selected_cats)].copy()
    active_df.index = range(1, len(active_df) + 1)

    # ── Tier summary metrics (computed after category filter) ─────────────────

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
                     disabled=not api_key or active_df.empty,
                     help="Requires an OpenAI API key in the sidebar."):
            with st.spinner("Generating briefing..."):
                st.session_state.briefing = llm.generate_queue_briefing(active_df, api_key)

    if not api_key:
        st.caption("_Add an OpenAI API key in the sidebar to enable AI features._")

    if st.session_state.briefing:
        with st.expander("📋 Queue Briefing", expanded=True):
            st.markdown(st.session_state.briefing)
            if st.button("Clear briefing"):
                st.session_state.briefing = ''
                st.rerun()

    # ── Ranked ticket table ───────────────────────────────────────────────────

    st.markdown("---")
    if active_df.empty:
        st.success("🎉 All tickets have been resolved!")
        selected_row = None
    else:
        st.caption("Click a row to select it, then respond or resolve below.")
        display_cols = ['Ticket', 'Category', 'Urgency Score', 'Tier']
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
            selection_mode="single-row",
        )

        csv = active_df.to_csv(index_label='Rank')
        st.download_button("Download ranked CSV", csv, "ranked_tickets.csv", "text/csv")

        # Resolve the selected row from the click event
        selected_rows = event.selection.rows
        selected_row = active_df.iloc[selected_rows[0]] if selected_rows else None

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

        action_col, resolve_col = st.columns(2)

        with action_col:
            if st.button("✍️ Draft AI Response",
                         disabled=not api_key,
                         help="Requires an OpenAI API key in the sidebar."):
                with st.spinner("Generating draft..."):
                    st.session_state.draft = llm.draft_ticket_response(
                        selected_row['Ticket'], tier, score, api_key
                    )

        with resolve_col:
            if st.button("✅ Mark as Resolved"):
                response_used = st.session_state.get('draft_textarea', '') or st.session_state.draft
                sent_to = st.session_state.get('to_email_input', '')
                st.session_state.resolved_log.append({
                    'ticket':   selected_row['Ticket'],
                    'tier':     tier,
                    'category': category,
                    'score':    score,
                    'response': response_used,
                    'sent_to':  sent_to,
                })
                st.session_state.resolved_tickets.add(selected_row['Ticket'])
                st.session_state.draft = ''
                st.session_state.last_selected_ticket = None
                st.rerun()

        if st.session_state.draft:
            st.markdown("**Edit response before sending:**")
            st.text_area(
                label="Response body",
                value=st.session_state.draft,
                height=160,
                label_visibility="collapsed",
                key="draft_textarea",
            )
            st.text_input("Recipient email address:", key="to_email_input")
            to_email = st.session_state.get('to_email_input', '')

            send_col, _ = st.columns([1, 3])
            with send_col:
                if st.button("📤 Send Email", disabled=not to_email.strip()):
                    st.success(f"Email sent to **{to_email}**.")

    # ── Resolved Tickets Log ──────────────────────────────────────────────────

    if st.session_state.resolved_log:
        st.markdown("---")
        log = st.session_state.resolved_log
        with st.expander(f"✅ Resolved Tickets ({len(log)})", expanded=False):
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
