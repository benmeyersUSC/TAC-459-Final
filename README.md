# TAC-459 — Intelligent Ticket Prioritizer

Fine-tuned BERT models for support ticket urgency scoring and category classification, with an LLM layer for response drafting and queue briefings.

## Setup

Requires Python 3.10+ and roughly 2 GB of free disk space for model weights.

```bash
git clone https://github.com/benmeyersUSC/TAC-459-Final.git
cd TAC-459-Final
pip install -r requirements.txt
streamlit run app.py
```

On first launch the app downloads two fine-tuned BERT models (~420 MB each) from HuggingFace Hub directly to a local `models/` directory. A progress bar is shown in the browser. Subsequent launches skip the download entirely.

## Usage

1. Upload a `.txt` file with one support ticket per line.
2. The app scores each ticket for urgency (0–1) and classifies it into one of eight categories (Hardware, Access, HR Support, etc.).
3. Use the category multiselect to filter the queue. Tier counts update to match.
4. Click any row in the table to select a ticket, then draft an AI-assisted response or mark it resolved.
5. Enter your OpenAI API key in the sidebar to enable response drafting and queue briefings.

## Model weights

Hosted on HuggingFace Hub: [benmeyersUSC/tac459-ticket-models](https://huggingface.co/benmeyersUSC/tac459-ticket-models)

Training notebooks are in `training_notebooks/`.
