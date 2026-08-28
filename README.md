# Synthetic Panel

Testing how well LLM-generated survey respondents reproduce real human survey data.

A staged interview walks you through designing a survey — who takes it, what it
measures, the items, and the human benchmark to validate against. The panel of
synthetic respondents assembles live alongside the conversation. Running an item
asks each respondent independently and shows how far the resulting distribution
sits from real human data.

## Running locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then add a free key from aistudio.google.com/apikey
.venv/bin/uvicorn app:app --reload --port 8010
```

## After cloning: enable the secret-scanning hook

Git does not run repo-provided hooks until you point it at them. Once per clone:

```bash
git config core.hooksPath .githooks
```

This blocks commits containing API keys or `.env` files. Without it, nothing stops
a key from being committed.

## Layout

```
app.py              FastAPI — the interview prompt, /api/chat, /api/respond
static/index.html   the whole client: landing, conversation, panel, stage views
.githooks/          pre-commit secret scanner
```

## Notes

Each synthetic respondent is asked in its own API call. Batching them into one
request lets the personas see each other's answers and converge, which would
invalidate the divergence measurement this tool exists to make.
