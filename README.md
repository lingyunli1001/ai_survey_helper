# Synthetic Panel

Pretest a survey on synthetic respondents — language models conditioned on
demographic profiles — before you spend money fielding it on real people.

Two ways in:

- **Design one from scratch.** A staged interview walks you through the
  respondent, the construct, the item pool, and a human benchmark to compare
  against. The panel of synthetic respondents assembles live alongside the
  conversation.
- **Bring your own.** Paste a questionnaire or upload a file (`.docx`, `.pdf`,
  `.csv`, `.txt`). It is parsed into items grouped by facet, and the interview
  picks up at Stage 1 to set up the panel around them.

Either way you get:

- **A format per question, not one scale for everything.** The model picks
  between 5-point agreement, 5-point frequency, yes/no, **multiple choice** with
  its own options, and **ordered custom options** (income bands, a bespoke
  scale) — whichever actually fits the question. An imported survey keeps the
  answer options it already had.
- **Wording review** on every item — double-barrels, leading phrasing,
  unbalanced options, vague terms, presupposition — each with a concrete
  rewrite, shown under the question.
- **A synthetic run.** Each respondent answers the whole questionnaire in one
  API call, in a shuffled order, the way a real respondent would — personas are
  never batched together. The result view shows the distribution plus
  per-question diagnostics. Ordered items get ceiling and floor effects,
  midpoint pile-up, near-zero variance (the homogeneity artefact) and subgroup
  splits by mean; multiple-choice items get concentration, unused options, and
  subgroup splits by top choice — mean and standard deviation are omitted there,
  since averaging unordered options means nothing.
- **Export.** The questionnaire (Markdown or CSV, with the review notes) and the
  run results (a Markdown report or CSV).

The human benchmark is collected during design as a reference point; comparing
the synthetic distribution against real crosstabs is still done by eye.

## Running locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
cp .env.example .env        # then add a free key from aistudio.google.com/apikey
.venv/bin/uvicorn app:app --reload --port 8010     # Windows: .venv\Scripts\uvicorn
```

Then open http://localhost:8010.

## After cloning: enable the secret-scanning hook

Git does not run repo-provided hooks until you point it at them. Once per clone:

```bash
git config core.hooksPath .githooks
```

This blocks commits containing API keys or `.env` files. Without it, nothing stops
a key from being committed.

## Layout

```
app.py              FastAPI backend
  /api/chat         the staged design interview (streamed)
  /api/import       parse a pasted or uploaded questionnaire
  /api/review       wording review of the drafted items
  /api/respond      one call per respondent, each answering the whole pool
static/index.html   the whole client: landing, conversation, panel, stage views,
                    diagnostics, export
.githooks/          pre-commit secret scanner
```

## Notes

Personas are never batched into one request. That would let them see each
other's answers and converge, destroying the divergence the run is meant to
surface. Each respondent gets its own call; a shared pacer keeps every endpoint
inside the free tier's 15-requests-a-minute limit.

Synthetic respondents are not a substitute for human data. A tight,
low-variance distribution is usually the model being uniform, not the
population agreeing — the diagnostics call that out rather than hide it.
