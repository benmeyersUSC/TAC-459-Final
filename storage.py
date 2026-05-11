# CHANGE: new module for persistence — keeps all read/write logic in one place

import json
from datetime import datetime
from pathlib import Path

# Where the persistent log lives. Lives in repo root for now; can be moved later.
STORAGE_PATH = Path("resolved_tickets.json")


def load_resolved_log():
    """Read the resolved-ticket log from disk. Returns [] if no file exists yet."""
    if not STORAGE_PATH.exists():
        return []
    try:
        with open(STORAGE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # If the file is corrupt or unreadable, start fresh rather than crashing the app.
        return []


def save_resolved_log(log):
    """Write the full resolved-ticket log to disk, overwriting whatever was there."""
    try:
        with open(STORAGE_PATH, "w") as f:
            json.dump(log, f, indent=2, default=str)
    except OSError:
        # If disk write fails, don't crash — the in-memory state is still intact.
        pass


def append_resolved(entry):
    """Append a single resolved-ticket entry to the log on disk.
    Adds a 'resolved_at' timestamp and an embedding vector for RAG retrieval."""
    if 'resolved_at' not in entry:
        entry['resolved_at'] = datetime.now().isoformat()
    # CHANGE: embed the ticket text so RAG retrieval can find similar past tickets
    if 'embedding' not in entry:
        try:
            import embeddings
            entry['embedding'] = embeddings.embed_text(entry['ticket']).tolist()
        except Exception:
            # If embedding fails (e.g. no network for first-time download), still save the entry
            entry['embedding'] = None
    log = load_resolved_log()
    log.append(entry)
    save_resolved_log(log)
    return entry


def get_resolved_ticket_texts():
    """Return a set of all resolved ticket texts. Used to filter the active queue."""
    return {entry['ticket'] for entry in load_resolved_log()}


def clear_resolved_log():
    """Wipe the persistent log. Useful for the admin panel later, or for testing."""
    if STORAGE_PATH.exists():
        STORAGE_PATH.unlink()
