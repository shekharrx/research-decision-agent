#!/usr/bin/env python3
"""
decide.py - research-to-decision agent.

Ask it a decision question. It will:
  1. Ask you up to 3 short questions, only if your answers would change the pick.
  2. Search the web for the real options and the facts that decide between them.
  3. Throw away the noise (spec-sheet filler, marketing, things that don't differ).
  4. Give ONE recommendation - or say plainly that there is no confident pick.

Anti-guessing rules enforced in code, not just in the prompt:
  - Every fact must carry a source URL that actually came back from a search.
    No URL            -> the fact is downgraded to UNKNOWN.
    URL never fetched -> the fact is downgraded to UNVERIFIED.
  - Downgraded facts are listed under "What I couldn't verify".
  - If confidence is low, the report leads with a warning instead of a pick.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python decide.py "Best discount broker for intraday F&O in India"
    python decide.py "Which laptop under 70k for Python + light ML?" --no-questions
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
from urllib.parse import urlsplit, urlunsplit

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Run:  pip install -r requirements.txt")


DEFAULT_MODEL = os.environ.get("DECIDE_MODEL", "claude-sonnet-5-5")
TODAY = dt.date.today().isoformat()


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

CLARIFY_SYSTEM = f"""Today is {TODAY}. You help someone make a decision fast.
Look at their question. Ask at most 3 short questions, and ONLY questions whose
answer would change which option you recommend (budget, hard constraints, must-haves,
how they'll actually use it). If the question is already specific enough, ask none.
Never ask about preferences that don't change the outcome."""

RESEARCH_SYSTEM = f"""Today is {TODAY}. You are a decision researcher. Your output is
raw notes for a later step - be factual and dense, not polished.

Do this:
1. Identify the 3-5 options a well-informed person would seriously consider. Drop
   obvious non-contenders early and say in one line why.
2. Decide the 3-6 criteria that ACTUALLY decide this choice for this person.
   A criterion matters only if the options differ on it AND it affects the outcome.
   Ignore things that are the same across options, marketing claims, and vanity specs.
3. Search for the current facts on each option x criterion. Prefer official sources
   (pricing pages, regulators, documentation, filings) over blogs and affiliate
   "top 10" lists. Note each fact's date when you can - stale prices are a trap.
4. When sources disagree, record BOTH values and both sources. Do not pick one.
5. When you cannot find a fact, write "NOT FOUND". Never fill gaps from memory.

Write notes as: OPTION / CRITERION / VALUE / source. Keep each fact tied to the
search result it came from."""

DECIDE_SYSTEM = f"""Today is {TODAY}. Turn research notes into one decision.

Hard rules:
- Use ONLY facts present in the notes. If a value isn't in the notes, its status is
  "unknown" and value is "unknown". Do not fill it from your own knowledge.
- source_url must be a URL that appears in the notes, copied exactly.
- If two sources disagree, status is "conflicting" and the value states both.
- Keep only criteria that differ between options and affect the outcome. Everything
  else you dropped goes in ignored_as_noise (short phrases).
- Recommend exactly one option. If the unknowns could plausibly flip the answer,
  set confidence to "low" and say exactly which unknown would flip it.
- If no option can be responsibly recommended, set choice to "NONE".
- Write plainly. No hype words."""


# --------------------------------------------------------------------------- #
# Tool schemas (used to force structured output)
# --------------------------------------------------------------------------- #

ASK_TOOL = {
    "name": "ask_questions",
    "description": "Questions for the user. Empty list if none are needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
        },
        "required": ["questions"],
    },
}

FACT = {
    "type": "object",
    "properties": {
        "criterion": {"type": "string"},
        "value": {"type": "string", "description": "Short. 'unknown' if not in notes."},
        "status": {"type": "string", "enum": ["verified", "conflicting", "unknown"]},
        "source_url": {"type": "string", "description": "Exact URL from notes, or ''."},
    },
    "required": ["criterion", "value", "status", "source_url"],
}

DECISION_TOOL = {
    "name": "submit_decision",
    "description": "Submit the final structured decision.",
    "input_schema": {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "weight": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["name", "why_it_matters", "weight"],
                },
            },
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "one_line": {"type": "string"},
                        "facts": {"type": "array", "items": FACT},
                    },
                    "required": ["name", "one_line", "facts"],
                },
            },
            "eliminated": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["name", "reason"],
                },
            },
            "recommendation": {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "description": "Option name, or 'NONE'."},
                    "reason": {"type": "string", "description": "2-4 sentences."},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "would_change_if": {"type": "string"},
                    "runner_up": {"type": "string"},
                },
                "required": ["choice", "reason", "confidence", "would_change_if"],
            },
            "uncertainties": {"type": "array", "items": {"type": "string"}},
            "ignored_as_noise": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["criteria", "options", "recommendation", "uncertainties"],
    },
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(f"  > {msg}", file=sys.stderr, flush=True)


def norm_url(url: str) -> str:
    """Normalise a URL so the same page compares equal (no fragment/trailing slash)."""
    try:
        p = urlsplit(url.strip())
        return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), p.query, ""))
    except ValueError:
        return url.strip()


def forced_tool_call(client, model, system, user_text, tool, max_tokens=8000) -> dict:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": user_text}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError(f"Model did not return {tool['name']}")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def clarify(client, model, question: str) -> str:
    out = forced_tool_call(client, model, CLARIFY_SYSTEM, question, ASK_TOOL, 1000)
    qs = [q for q in out.get("questions", []) if q.strip()][:3]
    if not qs:
        return ""
    print("\nA few quick questions (press Enter to skip any):", file=sys.stderr)
    answers = []
    for q in qs:
        a = input(f"  - {q}\n    > ").strip()
        if a:
            answers.append(f"Q: {q}\nA: {a}")
    return "\n".join(answers)


def research(client, model, question, context, max_searches, country):
    """Run the web-search turn. Returns (notes_text, set_of_urls_actually_seen)."""
    tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": max_searches}
    if country:
        tool["user_location"] = {"type": "approximate", "country": country}

    prompt = f"Decision to research:\n{question}"
    if context:
        prompt += f"\n\nWhat the person told me about their situation:\n{context}"

    messages = [{"role": "user", "content": prompt}]
    blocks = []
    for _ in range(8):  # pause_turn continuations
        resp = client.messages.create(
            model=model, max_tokens=16000, system=RESEARCH_SYSTEM,
            tools=[tool], messages=messages,
        )
        blocks.extend(resp.content)
        if resp.stop_reason == "pause_turn":
            # Send the paused assistant turn back unchanged so the server continues it.
            messages = [messages[0], {"role": "assistant", "content": resp.content}]
            continue
        break

    seen: set[str] = set()
    notes: list[str] = []
    searches = 0
    for b in blocks:
        if b.type == "server_tool_use":
            searches += 1
            log(f"searched: {b.input.get('query', '')}")
        elif b.type == "web_search_tool_result":
            if isinstance(b.content, list):
                for r in b.content:
                    seen.add(norm_url(r.url))
            else:
                log(f"search error: {getattr(b.content, 'error_code', 'unknown')}")
        elif b.type == "text":
            text = b.text
            for c in getattr(b, "citations", None) or []:
                url = getattr(c, "url", None)
                if url:
                    seen.add(norm_url(url))
                    text += f" [source: {url}]"
            notes.append(text)

    log(f"{searches} searches, {len(seen)} distinct sources")
    return "".join(notes).strip(), seen


def decide(client, model, question, context, notes) -> dict:
    user = (
        f"Decision: {question}\n\n"
        + (f"Person's situation:\n{context}\n\n" if context else "")
        + f"Research notes:\n<notes>\n{notes}\n</notes>"
    )
    return forced_tool_call(client, model, DECIDE_SYSTEM, user, DECISION_TOOL, 12000)


def enforce_evidence(decision: dict, seen_urls: set[str]) -> list[str]:
    """Downgrade any fact whose source wasn't actually retrieved. Returns issues found."""
    issues = []
    for opt in decision.get("options", []):
        for f in opt.get("facts", []):
            url = (f.get("source_url") or "").strip()
            label = f"{opt['name']} / {f['criterion']}"
            if f["status"] == "unknown":
                continue
            if not url:
                issues.append(f"{label}: claimed without a source - treated as unknown")
                f["status"], f["value"] = "unknown", f"unknown (claimed: {f['value']})"
            elif norm_url(url) not in seen_urls:
                issues.append(f"{label}: source wasn't in the search results - not trusted")
                f["status"] = "unverified"

    # If the recommended option itself leaned on a downgraded fact, lower confidence.
    rec = decision.get("recommendation", {})
    pick = rec.get("choice", "").strip().lower()
    if pick and any(i.lower().startswith(pick + " /") for i in issues):
        rec["confidence"] = {"high": "medium", "medium": "low"}.get(rec.get("confidence"), "low")
        issues.append(f"Confidence lowered: the pick ({rec['choice']}) relied on a fact that couldn't be verified")
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
    rec = d["recommendation"]
    sources: list[str] = []

    def cite(url: str) -> str:
        if not url:
            return ""
        if url not in sources:
            sources.append(url)
        return f" [{sources.index(url) + 1}]"

    out = [f"# {question}", f"_Researched {TODAY}_", ""]

    # Headline
    if rec["choice"].strip().upper() == "NONE":
        out += ["## No confident recommendation", "", rec["reason"], ""]
    else:
        flag = "  -- LOW CONFIDENCE, read the unknowns below" if rec["confidence"] == "low" else ""
        out += [f"## Pick: {rec['choice']}", f"**Confidence:** {rec['confidence']}{flag}", "",
                rec["reason"], ""]
        if rec.get("runner_up"):
            out += [f"**Runner-up:** {rec['runner_up']}", ""]
    out += [f"**This would change if:** {rec['would_change_if']}", ""]

    # Uncertainty first - before the table, so it can't be skimmed past
    unknowns = list(d.get("uncertainties", [])) + issues
    if unknowns:
        out += ["## What I couldn't verify", ""] + [f"- {u}" for u in unknowns] + [""]

    # Comparison table: criteria rows x option columns
    opts = d["options"]
    crits, same = split_same(opts, d["criteria"])
    if opts and crits:
        out += ["## What actually matters", ""]
        out.append("| Criterion | " + " | ".join(o["name"] for o in opts) + " |")
        out.append("|---|" + "---|" * len(opts))
        weight_order = {"high": 0, "medium": 1, "low": 2}
        for c in sorted(crits, key=lambda c: weight_order.get(c["weight"], 3)):
            cells = []
            for o in opts:
                f = next((x for x in o["facts"] if x["criterion"] == c["name"]), None)
                if not f:
                    cells.append("unknown ?")
                else:
                    val = f["value"].replace("|", "/")
                    cells.append(val + MARK.get(f["status"], "") + cite(f.get("source_url", "")))
            out.append(f"| **{c['name']}** ({c['weight']}) | " + " | ".join(cells) + " |")
        out += ["", "_? = not found. Nothing in this table was filled in from memory._", ""]
        out += ["**Why these criteria:**"] + [f"- {c['name']}: {c['why_it_matters']}" for c in crits] + [""]
    if same:
        out += ["**Same for all options (so it doesn't decide anything):** " + ", ".join(same), ""]

    if d.get("eliminated"):
        out += ["## Dropped early", ""] + [f"- **{e['name']}** - {e['reason']}" for e in d["eliminated"]] + [""]
    if d.get("ignored_as_noise"):
        out += ["## Ignored as noise", "", ", ".join(d["ignored_as_noise"]), ""]
    if sources:
        out += ["## Sources", ""] + [f"{i}. {u}" for i, u in enumerate(sources, 1)] + [""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Research a decision and get one clear pick.")
    ap.add_argument("question", nargs="*", help="The decision, e.g. 'best broker for F&O in India'")
    ap.add_argument("--no-questions", action="store_true", help="Skip the clarifying questions")
    ap.add_argument("--searches", type=int, default=12, help="Max web searches (default 12)")
    ap.add_argument("--country", default="IN", help="2-letter country to localise search ('' for none)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="decisions", help="Folder to save reports")
    ap.add_argument("--json", action="store_true", help="Also save the raw structured decision")
    args = ap.parse_args()

    question = " ".join(args.question).strip() or input("What are you deciding? > ").strip()
    if not question:
        sys.exit("No question given.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (get one at https://platform.claude.com).")

    client = anthropic.Anthropic()

    context = "" if args.no_questions else clarify(client, args.model, question)
    log("researching...")
    notes, seen = research(client, args.model, question, context, args.searches, args.country)
    if not seen:
        sys.exit("No search results came back, so there's nothing to base a decision on. "
                 "Check that web search is enabled for your API organisation.")
    log("comparing and deciding...")
    decision = decide(client, args.model, question, context, notes)
    issues = enforce_evidence(decision, seen)
    report = render(question, decision, issues)

    print("\n" + report)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60] or "decision"
    path = outdir / f"{TODAY}_{slug}.md"
    path.write_text(report, encoding="utf-8")
    if args.json:
        path.with_suffix(".json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    log(f"saved {path}")


if __name__ == "__main__":
    main()
