#!/usr/bin/env python3
"""
decide_free.py - research-to-decision agent that runs on FREE services only.

  AI brain : Google Gemini API, free tier  (key from https://aistudio.google.com/apikey)
  Search   : DuckDuckGo via the `ddgs` package (no key, no account)

What it does:
  1. Asks you up to 3 short questions - only ones that would change the pick.
  2. Plans the options + criteria, then searches the web and reads the top pages.
  3. Keeps only what actually separates the options; drops the noise.
  4. Gives ONE recommendation, or says plainly there is no confident pick.

Anti-guessing rules enforced in code:
  - Every fact must cite a source ID ([S3]) from pages this script actually fetched.
    No source       -> marked UNKNOWN.
    Fake source ID  -> marked UNVERIFIED.
  - If the pick relies on such a fact, its confidence is lowered automatically.

Usage (PowerShell):
    $env:GEMINI_API_KEY="your-key"
    python decide_free.py "Best discount broker for intraday F&O in India"
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import pathlib
import re
import sys
import time

try:
    from google import genai
    from ddgs import DDGS
    import requests
except ImportError:
    sys.exit("Missing packages. Run:  pip install -r requirements_free.txt")


DEFAULT_MODEL = os.environ.get("DECIDE_MODEL", "gemini-3.8-flash")
TODAY = dt.date.today().isoformat()
PAGE_CHARS = 3500        # how much text to keep from each fetched page
UA = {"User-Agent": "Mozilla/5.0 (decide-agent; personal research)"}


def log(msg: str) -> None:
    print(f"  > {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Gemini helper: ask for JSON, parse it, retry on rate limits
# --------------------------------------------------------------------------- #

def ask_json(client, model: str, prompt: str, retries: int = 4) -> dict:
    prompt += "\n\nReply with ONLY valid JSON. No markdown, no code fences, no commentary."
    last_err = None
    for attempt in range(retries):
        try:
            r = client.interactions.create(
                model=model,
                input=prompt,
                response_format={"type": "text", "mime_type": "application/json"},
            )
            text = (r.output_text or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)   # strip fences just in case
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            log("model returned broken JSON, retrying...")
        except Exception as e:  # rate limit / temporary errors
            msg = str(e)
            last_err = e
            if "API key" in msg or "API_KEY" in msg or "401" in msg or "403" in msg:
                sys.exit("Gemini rejected the key. Delete gemini_key.txt in this folder, run again, "
                         "and paste a fresh key from https://aistudio.google.com/apikey")
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg:
                wait = 20 * (attempt + 1)
                log(f"free-tier limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini call failed after {retries} tries: {last_err}")


# --------------------------------------------------------------------------- #
# Search + page reading (free)
# --------------------------------------------------------------------------- #

def web_search(query: str, n: int, region: str) -> list[dict]:
    try:
        return DDGS().text(query, region=region, max_results=n) or []
    except Exception as e:
        log(f"search failed for '{query}': {e}")
        return []


def fetch_text(url: str) -> str:
    """Download a page and return its readable text (rough but dependency-free)."""
    try:
        r = requests.get(url, headers=UA, timeout=12)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return ""
        t = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", " ", r.text)
        t = re.sub(r"(?s)<[^>]+>", " ", t)
        t = html.unescape(re.sub(r"\s+", " ", t)).strip()
        return t[:PAGE_CHARS]
    except Exception:
        return ""


def gather_sources(queries, per_query, region, read_pages) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    seen_urls = set()
    for q in queries:
        log(f"searching: {q}")
        for r in web_search(q, per_query, region):
            url = r.get("href") or r.get("url") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sid = f"S{len(sources) + 1}"
            body = r.get("body", "")
            if read_pages:
                page = fetch_text(url)
                if page:
                    body = page
            sources[sid] = {"title": r.get("title", ""), "url": url, "text": body}
        time.sleep(1)  # be polite to the free search endpoint
    return sources


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def clarify(client, model, question: str) -> str:
    out = ask_json(client, model, f"""Today is {TODAY}. Someone needs to decide: "{question}"
Ask at most 3 short questions, ONLY ones whose answer would change which option you'd
recommend (budget, hard constraints, must-haves, how they'll use it). If the question
is specific enough, ask none.
JSON shape: {{"questions": ["...", "..."]}}""")
    qs = [q for q in out.get("questions", []) if isinstance(q, str) and q.strip()][:3]
    if not qs:
        return ""
    print("\nA few quick questions (press Enter to skip any):", file=sys.stderr)
    answers = []
    for q in qs:
        a = input(f"  - {q}\n    > ").strip()
        if a:
            answers.append(f"Q: {q}\nA: {a}")
    return "\n".join(answers)


def plan(client, model, question, context, max_queries) -> dict:
    return ask_json(client, model, f"""Today is {TODAY}. Plan research for this decision:
"{question}"
{"Person's situation:" + chr(10) + context if context else ""}

1. List the 3-5 options a well-informed person would seriously consider.
2. List the 3-6 criteria that ACTUALLY decide this choice: options must differ on them
   and they must affect the outcome. Skip marketing fluff and things all options share.
3. Write up to {max_queries} web search queries that will find CURRENT facts for those
   options x criteria. Prefer queries that surface official pages (pricing, charges,
   documentation, regulators) over "top 10" blogs. Include the year {TODAY[:4]} where
   freshness matters.

JSON shape: {{"options": ["..."], "criteria": ["..."], "queries": ["..."]}}""")


def decide(client, model, question, context, sources: dict[str, dict]) -> dict:
    src_block = "\n\n".join(
        f"[{sid}] {s['title']}\nURL: {s['url']}\n{s['text']}" for sid, s in sources.items()
    )
    return ask_json(client, model, f"""Today is {TODAY}. Turn the sources below into ONE decision.
Decision: "{question}"
{"Person's situation:" + chr(10) + context if context else ""}

HARD RULES
- Use ONLY facts stated in the sources. Do NOT use your own knowledge for any value.
- Every fact cites exactly one source id like "S4". If no source states it:
  status "unknown", value "unknown", source "".
- If sources disagree: status "conflicting", value states both (e.g. "Rs 20 (S2) vs Rs 0 (S7)").
- Prefer official sources over blogs; note when a value may be outdated.
- Keep only criteria that differ between options and affect the outcome. List what you
  dropped in ignored_as_noise (short phrases).
- Recommend exactly one option. If unknowns could plausibly flip the answer, confidence
  is "low" and would_change_if names that unknown. If nothing can be responsibly
  recommended, choice is "NONE".
- Plain language, no hype.

JSON shape:
{{
  "criteria": [{{"name": "...", "why_it_matters": "...", "weight": "high|medium|low"}}],
  "options": [{{"name": "...", "one_line": "...",
               "facts": [{{"criterion": "...", "value": "...",
                          "status": "verified|conflicting|unknown", "source": "S1"}}]}}],
  "eliminated": [{{"name": "...", "reason": "..."}}],
  "recommendation": {{"choice": "...", "reason": "2-4 sentences", "confidence": "high|medium|low",
                     "would_change_if": "...", "runner_up": "..."}},
  "uncertainties": ["..."],
  "ignored_as_noise": ["..."]
}}

SOURCES
{src_block}""")


def enforce_evidence(d: dict, sources: dict[str, dict]) -> list[str]:
    """Downgrade facts whose source id doesn't exist. Attach real URLs to the rest."""
    issues = []
    for opt in d.get("options", []):
        for f in opt.get("facts", []):
            sid = str(f.get("source", "")).strip().strip("[]").upper()
            label = f"{opt.get('name')} / {f.get('criterion')}"
            f["source_url"] = ""
            if f.get("status") == "unknown":
                continue
            if not sid:
                issues.append(f"{label}: claimed without a source - treated as unknown")
                f["status"], f["value"] = "unknown", f"unknown (claimed: {f.get('value')})"
            elif sid not in sources:
                issues.append(f"{label}: cited a source that doesn't exist - not trusted")
                f["status"] = "unverified"
            else:
                f["source_url"] = sources[sid]["url"]

    rec = d.setdefault("recommendation", {})
    pick = str(rec.get("choice", "")).strip().lower()
    if pick and any(i.lower().startswith(pick + " /") for i in issues):
        rec["confidence"] = {"high": "medium", "medium": "low"}.get(rec.get("confidence"), "low")
        issues.append(f"Confidence lowered: the pick ({rec.get('choice')}) relied on a fact that couldn't be verified")
    return issues


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _norm(v: str) -> str:
    """Normalise a value for 'is it the same?' checks: '15.6 Inches' == '15.6-inch'."""
    v = re.sub(r"\(.*?\)|\[.*?\]", "", str(v).lower())
    v = re.sub(r"[^a-z0-9.]", "", v)
    return re.sub(r"(es|s)$", "", v)


def split_same(opts: list, crits: list) -> tuple[list, list[str]]:
    """Criteria where every option has the same verified value can't decide anything.
    Return (criteria that differ, 'Name (value)' strings for the identical ones)."""
    keep, same = [], []
    for c in crits:
        facts = [next((x for x in o.get("facts", []) if x.get("criterion") == c.get("name")), None) for o in opts]
        vals = {_norm(f["value"]) for f in facts if f and f.get("status") == "verified"}
        all_verified = len(opts) > 1 and all(f and f.get("status") == "verified" for f in facts)
        if all_verified and len(vals) == 1:
            same.append(f"{c.get('name')} ({facts[0]['value']})")
        else:
            keep.append(c)
    return keep, same


MARK = {"verified": "", "conflicting": " (conflicting)", "unknown": " ?", "unverified": " (unverified)"}


def render(question: str, d: dict, issues: list[str]) -> str:
    rec = d.get("recommendation", {})
    cited: list[str] = []

    def cite(url: str) -> str:
        if not url:
            return ""
        if url not in cited:
            cited.append(url)
        return f" [{cited.index(url) + 1}]"

    out = [f"# {question}", f"_Researched {TODAY}_", ""]
    if str(rec.get("choice", "")).strip().upper() in ("", "NONE"):
        out += ["## No confident recommendation", "", rec.get("reason", ""), ""]
    else:
        flag = "  -- LOW CONFIDENCE, read the unknowns below" if rec.get("confidence") == "low" else ""
        out += [f"## Pick: {rec['choice']}", f"**Confidence:** {rec.get('confidence')}{flag}", "",
                rec.get("reason", ""), ""]
        if rec.get("runner_up"):
            out += [f"**Runner-up:** {rec['runner_up']}", ""]
    out += [f"**This would change if:** {rec.get('would_change_if', '')}", ""]

    unknowns = list(d.get("uncertainties", [])) + issues
    if unknowns:
        out += ["## What I couldn't verify", ""] + [f"- {u}" for u in unknowns] + [""]

    opts, crits = d.get("options", []), d.get("criteria", [])
    crits, same = split_same(opts, crits)
    if opts and crits:
        out += ["## What actually matters", ""]
        out.append("| Criterion | " + " | ".join(o.get("name", "?") for o in opts) + " |")
        out.append("|---|" + "---|" * len(opts))
        order = {"high": 0, "medium": 1, "low": 2}
        for c in sorted(crits, key=lambda c: order.get(c.get("weight"), 3)):
            cells = []
            for o in opts:
                f = next((x for x in o.get("facts", []) if x.get("criterion") == c.get("name")), None)
                if not f:
                    cells.append("unknown ?")
                else:
                    val = str(f.get("value", "")).replace("|", "/")
                    cells.append(val + MARK.get(f.get("status"), "") + cite(f.get("source_url", "")))
            out.append(f"| **{c.get('name')}** ({c.get('weight')}) | " + " | ".join(cells) + " |")
        out += ["", "_? = not found. Nothing in this table was filled in from memory._", ""]
        out += ["**Why these criteria:**"] + [f"- {c.get('name')}: {c.get('why_it_matters')}" for c in crits] + [""]
    if same:
        out += ["**Same for all options (so it doesn't decide anything):** " + ", ".join(same), ""]

    if d.get("eliminated"):
        out += ["## Dropped early", ""] + [f"- **{e.get('name')}** - {e.get('reason')}" for e in d["eliminated"]] + [""]
    if d.get("ignored_as_noise"):
        out += ["## Ignored as noise", "", ", ".join(map(str, d["ignored_as_noise"])), ""]
    if cited:
        out += ["## Sources", ""] + [f"{i}. {u}" for i, u in enumerate(cited, 1)] + [""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Free research agent: one clear recommendation.")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--no-questions", action="store_true", help="Skip clarifying questions")
    ap.add_argument("--queries", type=int, default=8, help="Number of web searches (default 8)")
    ap.add_argument("--per-query", type=int, default=4, help="Results read per search (default 4)")
    ap.add_argument("--region", default="in-en", help="Search region (default in-en; 'wt-wt' = global)")
    ap.add_argument("--fast", action="store_true", help="Use search snippets only, don't open pages")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="decisions")
    args = ap.parse_args()

    question = " ".join(args.question).strip() or input("What are you deciding? > ").strip()
    if not question:
        sys.exit("No question given.")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    key_file = pathlib.Path(__file__).with_name("gemini_key.txt")
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        print("First time: paste your free Gemini API key (get one at https://aistudio.google.com/apikey)")
        key = input("Key > ").strip().strip('"')
        if not key:
            sys.exit("No key given.")
        key_file.write_text(key, encoding="utf-8")
        print("Saved. You won't be asked again. (Delete gemini_key.txt to change it.)\n")

    client = genai.Client(api_key=key)

    context = "" if args.no_questions else clarify(client, args.model, question)
    log("planning...")
    p = plan(client, args.model, question, context, args.queries)
    queries = [q for q in p.get("queries", []) if isinstance(q, str)][: args.queries]
    if not queries:
        sys.exit("Couldn't plan any searches. Try rephrasing the question.")

    sources = gather_sources(queries, args.per_query, args.region, not args.fast)
    log(f"{len(sources)} sources collected")
    if not sources:
        sys.exit("No search results came back (DuckDuckGo may be rate-limiting). Wait a minute and retry.")

    log("comparing and deciding...")
    d = decide(client, args.model, question, context, sources)
    issues = enforce_evidence(d, sources)
    report = render(question, d, issues)

    print("\n" + report)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60] or "decision"
    path = outdir / f"{TODAY}_{slug}.md"
    path.write_text(report, encoding="utf-8")
    log(f"saved {path}")


if __name__ == "__main__":
    main()
