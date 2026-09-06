import asyncio
import json
import os
import random
import re
import time
from collections import deque
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
MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "8000"))
# Thinking shares maxOutputTokens with the visible reply. Left unset, flash-lite does
# no thinking at all and scales it dynamically with difficulty; sending a budget turns
# it ON. So -1 (omit) is the default, and the knob exists for models like gemini-3.6-*
# where thinking is always on and needs capping so the spec block still fits.
THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "-1"))
# The free tier allows 15 generate_content requests per minute per model, and every
# synthetic respondent is one request. Without pacing, a panel run burns the minute's
# quota in about two seconds and the rest of the run comes back as errors.
RPM = int(os.environ.get("GEMINI_RPM", "15"))
_recent: deque = deque()
_rate_lock = asyncio.Lock()


async def _throttle():
    """Block until sending one more request keeps us inside the per-minute quota."""
    async with _rate_lock:
        while True:
            now = time.monotonic()
            while _recent and now - _recent[0] >= 60.0:
                _recent.popleft()
            if len(_recent) < RPM:
                _recent.append(now)
                return
            # holding the lock while sleeping is deliberate: it queues callers in order
            # instead of releasing them all at once into the same exhausted window
            await asyncio.sleep(60.0 - (now - _recent[0]) + 0.05)
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:streamGenerateContent?alt=sse"
)
SENTINEL = "\u00a7SPEC\u00a7"

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

STAGE 3 — ITEMS. Draft the whole pool at once, then refine it.
  Your FIRST turn in this stage must already contain the finished pool: about TWENTY
  items in that turn's patch. Choose the facets yourself from the construct — never ask
  which facets to use, never ask permission to draft, never promise to draft next turn,
  and never add questions one at a time. The one question you ask on that turn is about
  revising the pool you have just written, not about whether to write it.
  Real items. One idea each, no double-barrels, balanced options, plain language.
  Twenty rephrasings of one question is a failure. Split the construct into 4-5 named
  facets and write 4-5 items per facet, so the pool spans the construct instead of
  circling one corner of it. Group the items facet by facet and tag each with its facet.
  Your prose names the facets and what you deliberately left out. Never list the items
  themselves — the panel already shows them.
  Every turn after that REFINES the pool, and your options are revision moves: more
  items on a facet, drop a facet, refocus it, plainer wording, sharper wording, and the
  move to the benchmark. When a revision rewrites existing items, reuse their ids.

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

ONE EXCEPTION, and it is narrow: on your first turn in stage 3 you do not ask before
drafting. You write the twenty-item pool outright, in that same turn's patch, choosing
the facets yourself. Do not offer to draft it, do not ask whether to draft it now, and
do not ask which facets to use.

That exception changes WHEN you draft and nothing else. Every rule above still binds on
that turn: two to four sentences, no lists, no bullets, no headers. The twenty items go
in the patch and ONLY in the patch — writing them out in your message is the single
worst thing you can do here, because the interface is already showing them in the panel
beside you. Say how many items you wrote and what the facets are called, in a sentence.
Nothing more.

Your reply ALWAYS comes first. Never begin a turn with the marker below and never send
a patch with nothing in front of it — a reply that starts with the marker shows the
person a blank message. Write the sentences, then the patch, in that order, every time,
including the turn where you draft the item pool.

The patch is REQUIRED on every single turn, with no exceptions. A turn with no patch
leaves the panel frozen and the person with no options to click, so it is a broken turn
even when the sentences read well. If nothing else changed, still send "stage" and
"options".

THEN, after your reply, on its own line, emit a spec PATCH:

§SPEC§{"stage":...,"options":[...],  ...only fields that CHANGED this turn... }

The interface already holds the current spec, shown at the end of these instructions.
Send only what changed. Always include "stage" and "options"; omit every field you are
not changing. Never re-send items that are unchanged, and send "dimensions" only when the
distribution actually changes — a long patch gets truncated and then nothing updates at
all.

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
  items       array of {"id": string, "text": string, "facet": string,
              "scale": "agree5"|"freq5"|"binary"} — empty until stage 3.
              "facet" is the short label of the sub-area the item covers; items sharing
              a facet are shown grouped under it.
              CRITICAL — items are merged BY ID, never replaced as a list. Send only the
              items you are adding or changing this turn.
              Give every item a short stable id: q1, q2, q3 and so on.
                - a NEW id appends a question
                - reusing an EXISTING id rewrites that question in place
                - {"id":"q2","remove":true} deletes it
              The current items and their ids are in the spec below. Reuse an id only
              when you mean to change that exact question, never to add a new one, and
              never renumber items that already exist.
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


class Item(BaseModel):
    id: str
    text: str
    scale: str = "agree5"


class RespondRequest(BaseModel):
    items: list[Item]
    personas: list[Persona]


SCALE_POINTS = {
    "agree5": [
        "Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree",
    ],
    "freq5": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "binary": ["Yes", "No"],
}

# One request per RESPONDENT, who answers the whole questionnaire in it — the same
# thing a real respondent does. Personas are never batched together: that would let
# them see each other's answers and converge, destroying the divergence measurement
# this tool exists to make. Item order is shuffled per respondent so that a persona
# anchoring on its first answer shows up as noise rather than a systematic pull.
RESPONDENT_PROMPT = """You are answering a survey as this person:
{profile}

Answer exactly as that person would — not as an average, not as a model. Let their
circumstances shape every answer, including indifference, ambivalence or inconsistency
where that is realistic for them. Do not try to be consistent across questions for its
own sake, and do not give the same rating to everything.

{questions}

Answer every question. Reply with one line per question: the question number, a space,
then the number of your choice. Nothing else — no words, no explanation."""


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
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }
    if THINKING_BUDGET >= 0:
        payload["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": THINKING_BUDGET
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
    if not req.items:
        return {"error": "No items to ask."}

    gate = asyncio.Semaphore(5)
    async with httpx.AsyncClient(timeout=180.0) as client:
        results = await asyncio.gather(
            *[_ask_one(client, gate, person, req.items) for person in req.personas]
        )
    return {"answers": results}


def _render_questions(items: list[Item], order: list[int]) -> str:
    """The questionnaire as the respondent sees it, in their own shuffled order."""
    blocks = []
    for shown, idx in enumerate(order, start=1):
        item = items[idx]
        points = SCALE_POINTS.get(item.scale, SCALE_POINTS["agree5"])
        choices = "   ".join(f"{i + 1} {p}" for i, p in enumerate(points))
        blocks.append(f"{shown}. {item.text}\n   {choices}")
    return "\n\n".join(blocks)


def _parse_answers(raw: str, items: list[Item], order: list[int]) -> dict:
    """Map "<question number> <choice>" lines back onto the unshuffled items."""
    out = {}
    for line in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[).:\-]?\s+(\d+)", line)
        if not m:
            continue
        shown, value = int(m.group(1)), int(m.group(2))
        if not 1 <= shown <= len(order):
            continue
        item = items[order[shown - 1]]
        points = SCALE_POINTS.get(item.scale, SCALE_POINTS["agree5"])
        if 1 <= value <= len(points):
            out[item.id] = value
    return out


async def _ask_one(client, gate, person, items: list[Item]):
    order = list(range(len(items)))
    random.shuffle(order)
    prompt = RESPONDENT_PROMPT.format(
        profile=person.profile, questions=_render_questions(items, order)
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            # one short line per item, plus room for a model that pads
            "maxOutputTokens": max(300, 60 * len(items) + 200),
        },
    }

    async with gate:
        for attempt in range(3):
            await _throttle()
            try:
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                    json=payload,
                    headers={"x-goog-api-key": API_KEY},
                )
            except httpx.HTTPError as exc:
                return {"id": person.id, "values": {}, "error": str(exc)[:80]}
            if r.status_code != 429:
                break
            message, retry_after = _readable_error(429, r.text)
            if not retry_after or attempt == 2:
                return {"id": person.id, "values": {}, "error": message}
            await asyncio.sleep(min(retry_after, 30))

    if r.status_code != 200:
        message, _ = _readable_error(r.status_code, r.text)
        return {"id": person.id, "values": {}, "error": message}

    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        raw = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, json.JSONDecodeError):
        return {"id": person.id, "values": {}, "error": "no answer returned"}

    values = _parse_answers(raw, items, order)
    if not values:
        return {"id": person.id, "values": {}, "error": "unparseable: " + raw[:24]}
    # items the respondent skipped are reported, not silently blank
    missing = [it.id for it in items if it.id not in values]
    return {
        "id": person.id,
        "values": values,
        "error": None,
        "missing": missing or None,
    }


async def _error_stream(message: str):
    yield _sse({"error": message})


def _patch_json(seen: str):
    """The patch object from a reply, or None if it is absent or not valid JSON.

    Brace-matched rather than taking the last "}" in the string, because the model
    sometimes writes prose after the patch.
    """
    at = seen.find(SENTINEL)
    if at < 0:
        return None
    text = re.sub(r"^```(json)?", "", seen[at + len(SENTINEL):].strip(), flags=re.I).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def _recover_patch(payload: dict, reply: str) -> str:
    """Ask for the patch a reply came back without.

    flash-lite drops the block every so often. Prompting has not made that reliable, so
    recover it instead: without a patch the panel freezes and the person is left with no
    options to click, which reads as the app having ignored them.
    """
    followup = {
        "systemInstruction": payload.get("systemInstruction"),
        "contents": list(payload.get("contents", []))
        + [
            {"role": "model", "parts": [{"text": reply}]},
            {
                "role": "user",
                "parts": [
                    {
                        "text": "That reply was missing its spec patch. Send only the "
                        "patch for it now — the marker, then the JSON object, and "
                        "nothing else. No prose before or after. It must include "
                        '"stage" and "options".'
                    }
                ],
            },
        ],
        # low temperature: this is a formatting repair, not a fresh answer
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": MAX_OUTPUT_TOKENS},
    }
    await _throttle()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                json=followup,
                headers={"x-goog-api-key": API_KEY},
            )
    except httpx.HTTPError:
        return ""
    if r.status_code != 200:
        return ""
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, json.JSONDecodeError):
        return ""
    at = text.find(SENTINEL)
    return text[at:] if at >= 0 else ""


async def _gemini_stream(payload: dict):
    url = ENDPOINT.format(model=MODEL)
    finish = ""
    seen = ""
    await _throttle()
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
                    text, reason = _extract_part(chunk)
                    if reason:
                        finish = reason
                    if text:
                        seen += text
                        yield _sse({"text": text})
    except httpx.HTTPError as exc:
        yield _sse({"error": f"Could not reach the model: {exc}"})

    # A turn whose patch is missing OR unparseable freezes the panel, so ask for it
    # again rather than letting the turn land broken. The client reads the LAST patch
    # in the stream, so appending a good one supersedes a malformed one.
    recovered = ""
    if seen.strip() and API_KEY and finish != "MAX_TOKENS" and _patch_json(seen) is None:
        recovered = "missing" if SENTINEL not in seen else "malformed"
        patch = await _recover_patch(payload, seen)
        if patch:
            seen += patch
            yield _sse({"text": patch})
        else:
            recovered += "-failed"

    # One line per turn, so a reply that renders wrong in the browser can be traced to
    # what the model actually sent: where the sentinel landed and how it finished.
    cut = seen.find(SENTINEL)
    print(
        f"[turn] finish={finish or '-'} chars={len(seen)} "
        f"sentinel={'no' if cut < 0 else cut} "
        f"prose={len(seen[:cut].strip()) if cut >= 0 else len(seen.strip())}"
        f"{' recovered=' + recovered if recovered else ''}",
        flush=True,
    )

    # A truncated reply is otherwise indistinguishable from a clean one: the stream
    # just stops mid-sentence and the half turn poisons the rest of the conversation.
    if finish and finish != "STOP":
        yield _sse({"truncated": True, "reason": finish})

    yield _sse({"done": True})


def _extract_part(raw: str) -> tuple[str, str]:
    """Returns (text, finish_reason) for one SSE chunk; finish_reason is "" until the last."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "", ""
    candidate = (data.get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    return text, candidate.get("finishReason") or ""


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
