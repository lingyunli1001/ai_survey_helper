import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

STATIC = Path(__file__).parent / "static"
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:streamGenerateContent?alt=sse"
)

SYSTEM_PROMPT = """\
You are a survey methodologist. You walk someone through building a survey that will \
first be pretested on synthetic respondents — language models conditioned on \
demographic profiles — before any real fieldwork.

You work in five stages, in order. Never skip ahead, never work on two at once.

STAGE 1 — RESPONDENT. Who takes this survey?
  Every synthetic respondent gets conditioned on the profile defined here.
  Get the population, then the demographic dimensions that plausibly MOVE the answer.
  Say in a few words why each dimension would matter. Skip boilerplate demographics
  that would not moderate this particular outcome.

STAGE 2 — CONSTRUCT. What is actually being measured?
  Push past the topic to the specific latent thing, and to the decision it informs.
  "Attitudes toward AI" is a topic; "perceived threat to one's own job security over
  five years" is a construct. Also establish what it is NOT — the adjacent things this
  survey deliberately will not measure.

STAGE 3 — ITEMS. Draft the questions.
  Real items. One idea each, no double-barrels, balanced options, plain language.

STAGE 4 — BENCHMARK. What human data validates this?
  A published survey with demographic crosstabs on a comparable item — Pew, GSS, ANES,
  World Values Survey. Name the source and the specific comparable question.

STAGE 5 — READY. Summarise the design and say it is ready to run.

MOVING BETWEEN STAGES
Do not advance the moment you have a workable answer. When a stage has enough to
proceed, ask whether they want to add anything else to it, and make one of your options
the move to the next stage, named: "Move on to the construct". When you do advance,
open with a short clause naming the new stage — "Right, the construct then."

Three hard rules about advancing:
- Advance ONLY when the person's most recent message unmistakably asks to. "Move on",
  "next", "that's fine let's go" all qualify. Anything else does not.
- If their message is ambiguous, or could be about the current stage's content, STAY
  where you are and act on the content reading. A one-word reply like "second" answers
  your question about the current stage; it is not consent to leave it.
- If they tell you that you misread them, or repeat a request you did not act on, do
  that thing immediately and do not re-ask your previous question. Never ask a question
  you have already asked in this conversation.

The order is respondent -> construct -> items -> benchmark -> run. Your "move on"
option must name the stage that comes NEXT, never the one you are already in.

HOW YOU TALK
- Ask exactly ONE question per turn. Never stack questions.
- Two to four sentences. No bullet lists, no headers, no bold.
- Start with substance. Never open by appraising the question or the topic — no "that
  is a big space", "great starting point", "interesting question". Say the thing.
- Do not force precision early. A broad answer is fine: accept it, put it in the spec,
  and refine later. Push for specificity only when the vagueness actually blocks the
  next stage. It is better to move forward loosely than to interrogate.
- Offer your own proposal as one option among several, not as the answer to accept.
- React to what they said before asking the next thing. Do not restate their answer.
- Never enumerate the options in your prose — they are rendered as buttons directly
  beneath your message. Ask the question in its general form and stop.

THEN, after your reply, on its own line, emit a spec PATCH:

§SPEC§{"stage":...,"options":[...],  ...only fields that CHANGED this turn... }

The interface already holds the current spec, shown at the end of these instructions.
Send only what changed. Always include "stage" and "options"; omit every field you are
not changing. Never re-send dimensions or items that are unchanged — a long patch gets
truncated and then nothing updates at all.

Field reference — no markdown fences, nothing after the JSON:
  stage       integer 1-5, the stage you are working on right now
  population  string or null — who is being sampled, one short phrase
  n           integer 20-200, panel size (default 60)
  dimensions  array of {"name": string, "levels": [{"label": string, "pct": integer}]}
              pct per dimension sums to 100. Use realistic population shares.
              Prefer names: age, gender, education, income, region, employment.
              CRITICAL — the panel renders literally and only what you emit here.
              When the population fixes a demographic, keep that dimension and give it
              a SINGLE level at pct 100. "Women in their 20s" must emit
              gender:[{"label":"Female","pct":100}] and age:[{"label":"20-29","pct":100}].
              Never drop a dimension because it stopped varying, and never leave a
              broad distribution in place after the person has narrowed it.
  construct   null until stage 2, then
              {"name": short label, "definition": one sentence, "decision": what the
              result decides, "excludes": [2-4 adjacent things this will NOT measure]}
              Fill it in progressively — emit partial fields as you learn them.
  items       array of {"text": string, "scale": "agree5"|"freq5"|"binary"} — empty
              until stage 3
  benchmark   null until stage 4, then
              {"source": dataset and year, "item": the comparable published question,
              "note": one line on how comparable it really is}
  ready       integer 0-100, how completely specified the survey is overall
  options     REQUIRED every single turn, never omitted, never empty.
              array of 2-4 short strings — the concrete answers a person could give to
              the question you just asked. Each a genuinely DIFFERENT direction, not a
              rephrasing. Written as the person would say them, first person, at most
              about eight words. No numbering, no trailing punctuation. The interface
              adds its own free-text escape, so never include an "other" option
              yourself. Emit fresh options every turn.

Never send a field just to repeat its current value."""


class Turn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    messages: list[Turn]
    spec: dict | None = None


class Persona(BaseModel):
    id: int
    profile: str          # "Age 34 · Female · Bachelor's or higher"


class RespondRequest(BaseModel):
    text: str             # the item being asked
    scale: str = "agree5"
    personas: list[Persona]


SCALE_POINTS = {
    "agree5": [
        "Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree",
    ],
    "freq5": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "binary": ["Yes", "No"],
}

# Each respondent is answered in its own request. Batching personas into one call
# would let them see each other's answers and converge, which destroys the whole
# point of measuring how a conditioned model diverges from real respondents.
RESPONDENT_PROMPT = """You are answering a survey as this person:
{profile}

Answer exactly as that person would — not as an average, not as a model. Let their
circumstances shape the answer, including indifference or inconsistency where that is
realistic.

Question: {text}

{choices}

Reply with ONLY the number of your choice. No words, no punctuation."""


@app.get("/")
def index():
    # no-store during development: a cached page hides every frontend fix
    return FileResponse(
        STATIC / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "key_configured": bool(API_KEY)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not API_KEY:
        return StreamingResponse(
            _error_stream("No GEMINI_API_KEY configured on the server."),
            media_type="text/event-stream",
        )

    state = json.dumps(req.spec, separators=(",", ":")) if req.spec else "{}"
    instruction = SYSTEM_PROMPT + (
        "\n\nCURRENT SPEC (the interface already holds this; do not repeat it back):\n"
        + state
    )

    payload = {
        "systemInstruction": {"parts": [{"text": instruction}]},
        "contents": [
            {
                "role": "user" if t.role == "user" else "model",
                "parts": [{"text": t.text}],
            }
            for t in req.messages
        ],
        "generationConfig": {
            "temperature": 0.8,
            # generous: thinking tokens count against this, and a tight budget
            # truncated the spec block mid-JSON
            "maxOutputTokens": 3000,
        },
    }

    return StreamingResponse(
        _gemini_stream(payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/respond")
async def respond(req: RespondRequest):
    if not API_KEY:
        return {"error": "No GEMINI_API_KEY configured on the server."}

    points = SCALE_POINTS.get(req.scale, SCALE_POINTS["agree5"])
    choices = "\n".join(f"{i + 1} {p}" for i, p in enumerate(points))

    # small cap keeps us inside the 10s function limit on the free plan
    gate = asyncio.Semaphore(5)

    async with httpx.AsyncClient(timeout=25.0) as client:
        results = await asyncio.gather(
            *[
                _ask_one(client, gate, person, req.text, choices, len(points))
                for person in req.personas
            ]
        )
    return {"answers": results}


async def _ask_one(client, gate, person, text, choices, n_points):
    prompt = RESPONDENT_PROMPT.format(
        profile=person.profile, text=text, choices=choices
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 300},
    }
    async with gate:
        try:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                json=payload,
                headers={"x-goog-api-key": API_KEY},
            )
        except httpx.HTTPError as exc:
            return {"id": person.id, "value": None, "error": str(exc)[:80]}

    if r.status_code != 200:
        message, _ = _readable_error(r.status_code, r.text)
        return {"id": person.id, "value": None, "error": message}

    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        raw = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, json.JSONDecodeError):
        return {"id": person.id, "value": None, "error": "no answer returned"}

    match = re.search(r"[1-9]", raw)
    if not match:
        return {"id": person.id, "value": None, "error": "unparseable: " + raw[:24]}
    value = int(match.group())
    if not 1 <= value <= n_points:
        return {"id": person.id, "value": None, "error": f"out of range: {value}"}
    return {"id": person.id, "value": value, "error": None}


async def _error_stream(message: str):
    yield _sse({"error": message})


async def _gemini_stream(payload: dict):
    url = ENDPOINT.format(model=MODEL)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers={"x-goog-api-key": API_KEY},
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    message, retry_after = _readable_error(response.status_code, body)
                    yield _sse({"error": message, "retry_after": retry_after})
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk:
                        continue
                    text = _extract_text(chunk)
                    if text:
                        yield _sse({"text": text})
    except httpx.HTTPError as exc:
        yield _sse({"error": f"Could not reach the model: {exc}"})

    yield _sse({"done": True})


def _extract_text(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    return "".join(p.get("text", "") for p in parts)


def _readable_error(status: int, body: str) -> tuple[str, int]:
    """Returns (message, retry_after_seconds). retry_after is 0 when not retryable."""
    try:
        err = json.loads(body).get("error", {})
    except json.JSONDecodeError:
        err = {}
    message = err.get("message", "") or body[:200]

    if status == 429:
        # A per-day exhaustion is not worth retrying — it resets at midnight PT.
        if _quota_is_daily(err):
            return (
                f"Daily free-tier quota exhausted for {MODEL}. It resets at midnight "
                f"Pacific, or set GEMINI_MODEL to a model with a larger free tier.",
                0,
            )
        return "Rate limit reached on the free tier.", _retry_delay(err)
    if status in (401, 403):
        return "The API key was rejected. Check GEMINI_API_KEY.", 0
    if status == 404:
        return f"Model '{MODEL}' is not available on this key. {message}", 0
    if status >= 500:
        return "The model service is unavailable.", 5
    return message or f"Model request failed ({status}).", 0


def _quota_is_daily(err: dict) -> bool:
    for detail in err.get("details", []):
        for violation in detail.get("violations", []):
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def _retry_delay(err: dict) -> int:
    """Google returns a RetryInfo detail like {'retryDelay': '23s'}."""
    for detail in err.get("details", []):
        raw = detail.get("retryDelay")
        if raw:
            try:
                return max(1, min(120, int(float(str(raw).rstrip("s")))))
            except ValueError:
                pass
    return 30


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"
