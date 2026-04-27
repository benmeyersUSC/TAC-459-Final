
import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Model def
class BertUrgencyRegressor(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.pooler_output
        cls = self.dropout(cls)
        score = torch.sigmoid(self.regressor(cls))
        return score.squeeze(1)

# use streamlit caching to only save model once per session!
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertUrgencyRegressor(dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer, device

# for displaying color of ranked tickets
def score_to_tier(score, thresholds):
    if score >= thresholds['p0']:
        return 'P0 - Critical'
    elif score >= thresholds['p1']:
        return 'P1 - High'
    elif score >= thresholds['p2']:
        return 'P2 - Medium'
    else:
        return 'P3 - Low'

# aesthetics
def tier_color(val):
    if val.startswith('P0'):
        return 'background-color: #ff4b4b; color: white'
    elif val.startswith('P1'):
        return 'background-color: #ffa500; color: white'
    elif val.startswith('P2'):
        return 'background-color: #ffd700; color: black'
    else:
        return 'background-color: #90ee90; color: black'

# inference
def predict_urgency(text, model, tokenizer, device, max_len=128):
    text = ' '.join(text.strip().split())
    encoding = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=max_len, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        score = model(input_ids, attention_mask)
    return round(1.0 - score.item(), 4)  # flip: higher = more urgent, stays in [0,1]



# UI
st.set_page_config(page_title="Ticket Prioritizer", layout="wide")
st.title("🎫 Intelligent Ticket Prioritizer")
st.caption("Upload a .txt file with one ticket per line. Model ranks by predicted urgency.")

MODEL_PATH = st.sidebar.text_input(
    "Model path (.pth)", value="bert_urgency_refined.pth"
)

st.sidebar.markdown("### Tier thresholds")
p0_thresh = st.sidebar.slider("P0 (Critical) cutoff", 0.0, 1.0, 0.75, 0.05)
p1_thresh = st.sidebar.slider("P1 (High) cutoff", 0.0, 1.0, 0.50, 0.05)
p2_thresh = st.sidebar.slider("P2 (Medium) cutoff", 0.0, 1.0, 0.25, 0.05)
thresholds = {'p0': p0_thresh, 'p1': p1_thresh, 'p2': p2_thresh}

uploaded_file = st.file_uploader("Drop your tickets file here", type=["txt"])

if uploaded_file is not None:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    # clean tickets
    tickets = [l.strip() for l in lines if l.strip()]  # drop blank lines

    st.info(f"Loaded {len(tickets)} tickets. Running inference...")

    # load the model
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model(MODEL_PATH)

    # accumulate predictions with each ticket
    progress = st.progress(0)
    scores = []
    for i, ticket in enumerate(tickets):
        scores.append(predict_urgency(ticket, model, tokenizer, device))
        progress.progress((i + 1) / len(tickets))

    # and make DF from tickets and their urgency
    df = pd.DataFrame({'Ticket': tickets, 'Urgency Score': scores})
    df['Tier'] = df['Urgency Score'].apply(lambda s: score_to_tier(s, thresholds))
    df.sort_values('Urgency Score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1  # rank starts at 1

    st.success("Done! Tickets ranked below.")

    # some aesthetics
    counts = df['Tier'].value_counts()
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("🔴 P0 Critical", int(counts.get('P0 - Critical', 0)))
    c1.metric("🟠 P1 High", int(counts.get('P1 - High', 0)))
    c2.metric("🟡 P2 Medium", int(counts.get('P2 - Medium', 0)))
    c3.metric("🟢 P3 Low", int(counts.get('P3 - Low', 0)))

    styled = df.style.map(tier_color, subset=['Tier'])
    st.dataframe(styled, use_container_width=True)

    csv = df.to_csv(index_label='Rank')
    st.download_button("Download ranked CSV", csv, "ranked_tickets.csv", "text/csv")
