# TAC-459 — Intelligent Ticket Prioritizer

## For the grader — submission contents

| Deliverable | File |
| --- | --- |
| Final roadmap | [`ROADMAP_3.pdf`](ROADMAP_3.pdf) |
| Slide deck | [`SLIDES.pdf`](SLIDES.pdf) |
| Final demo video | [`FINAL_DEMO_VIDEO.mp4`](FINAL_DEMO_VIDEO.mp4) |
| MVP demo video | [`MVP_DEMO_VIDEO.mp4`](MVP_DEMO_VIDEO.mp4) |
| Runnable code | see below — entry point is `app.py` |
| Sample input | `TICKET_EXAMPLE.txt`, `LONG_TICKET_EXAMPLES.txt` |

**Source files (all in repo root):**
- `app.py` — Streamlit entry point (run this)
- `llm.py` — OpenAI prompts: draft, briefing, revise
- `embeddings.py` — sentence-transformers RAG over resolved tickets
- `storage.py` — JSON persistence for the resolved-ticket log
- `requirements.txt` — Python dependencies
- `.env.example` — template for the environment file (see Setup below)
- `training_notebooks/` — Colab notebooks for fine-tuning both BERT models

**Running instructions are in the "Setup" section directly below.** One quick step: copy `.env.example` to `.env` and paste in an OpenAI API key.

---

Two fine-tuned BERT models for support ticket urgency scoring and category classification, paired with an LLM layer for response drafting and queue briefings. Drafts are grounded in past resolved tickets via a local vector-store RAG.

## Setup

Requires Python 3.10+ and roughly 2 GB of free disk space for model weights.

```bash
git clone https://github.com/benmeyersUSC/TAC-459-Final.git
cd TAC-459-Final
python3 -m pip install -r requirements.txt
cp .env.example .env
# open .env in any editor and replace `sk-your-key-here` with your OpenAI API key
streamlit run app.py
```

The OpenAI key is required for the LLM features (response drafts, queue briefings, draft revisions). The app will still run without it — the BERT urgency + category models work independently — but the AI buttons will be disabled.

On first launch the app downloads two fine-tuned BERT models (~420 MB each) from HuggingFace Hub into a local `models/` directory. A progress bar is shown in the browser. Subsequent launches skip the download.

## Sample ticket files

Two sample files ship with the repo:

- `TICKET_EXAMPLE.txt` — short, terse tickets (good for a fast walkthrough of urgency scoring + category classification)
- `LONG_TICKET_EXAMPLES.txt` — longer, more realistic tickets (good for showcasing the LLM draft, queue briefing, and RAG features)

Drag either file into the upload box, or paste tickets directly into the "Add additional tickets" expander.

## Usage

1. Upload one of the sample files (or paste tickets).
2. The app scores each ticket for urgency (0–1) and classifies it into one of eight categories.
3. Filter by category and/or urgency tier (P0–P3) with the multiselects.
4. Click any row to select a ticket. Click multiple rows to unlock bulk actions.
5. **📋 Get Queue Briefing** — generates a shift briefing of the active queue.
6. **✍️ Draft AI Response** — generates a customer-facing reply, grounded in similar past resolved tickets (RAG). A green "RAG active" banner indicates whether the vector store contributed examples.
7. **🔄 Revise with notes** — type shorthand like "team is on it, ETA 2 hrs" and the LLM rewrites the draft.
8. **✅ Mark as Resolved** — saves the ticket + response to `resolved_tickets.json` for future RAG retrieval. The more you resolve, the better future drafts get.

## Admin mode

Click "🔐 Admin login" in the sidebar. Default password: `admin123` (set via `.env`).

Admin-only features:
- Tier threshold sliders (P0/P1/P2/P3 cutoffs)
- OpenAI API key field (in case the bundled key is rate-limited)
- Resolved-tickets log + clear-history button

## Architecture

- `app.py` — Streamlit UI
- `llm.py` — OpenAI prompts: draft, briefing, revise
- `embeddings.py` — sentence-transformers RAG over the resolved-ticket log
- `storage.py` — JSON persistence for resolved tickets
- `training_notebooks/` — Colab notebooks for fine-tuning both BERT models
- Model weights: [benmeyersUSC/tac459-ticket-models on HuggingFace Hub](https://huggingface.co/benmeyersUSC/tac459-ticket-models)
