"""
Role Match: the Python backend for Jana Vanshika's portfolio.

A visitor pastes a job description. This server compares it with the content of
the portfolio (profile.json) and returns:
  * which of Vanshika's skills and topics the role mentions,
  * which technologies the role asks for that are NOT on her portfolio (honest gaps),
  * which of her projects / courses / experience are closest in meaning (TF-IDF + cosine similarity),
  * a transparent overall score.

Run locally:   python app.py        then open http://127.0.0.1:5000
It also serves the portfolio itself (../index.html and ../assets), so one command is enough.
"""
import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
SITE = BASE.parent                      # folder that holds index.html and assets/
PROFILE = json.loads((BASE / "profile.json").read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------- settings
MIN_WORDS = 5                           # reject inputs that are too short to say anything
MAX_CHARS = 6000                        # reject huge pastes (protects the server)
RATE_LIMIT = 20                         # requests ...
RATE_WINDOW = 60                        # ... per this many seconds, per visitor
MIN_SIMILARITY = 0.04                   # ignore profile items that barely relate to the role
SIM_FULL_MARKS = 0.25                   # cosine similarity at which the "meaning" part scores 100%
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")   # set to your site's address once hosted

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


# ----------------------------------------------------------------------------- keyword detection
def term_regex(term: str) -> re.Pattern:
    """Match a term as a whole word, so 'java' does not fire inside 'javascript' or 'git' inside 'github'."""
    return re.compile(r"(?<![a-z0-9+#])" + re.escape(term.lower()) + r"(?![a-z0-9+#])")


def build_entries():
    entries = []
    for s in PROFILE["skills"]:
        entries.append({"kind": "skill", "name": s["name"], "section": "#skills",
                        "detail": s["category"], "patterns": [term_regex(a) for a in s["aliases"]]})
    for t in PROFILE["topics"]:
        entries.append({"kind": "topic", "name": t["name"], "section": t["section"],
                        "detail": t["source"], "patterns": [term_regex(a) for a in t["aliases"]]})
    return entries


ENTRIES = build_entries()
GAP_PATTERNS = [(g, term_regex(g)) for g in PROFILE["gap_terms"]]


def find_matches(text: str):
    """Return profile skills/topics mentioned in `text`. A match inside a longer match is dropped
    (e.g. 'neural networks' inside 'graph neural networks')."""
    hits = []
    for e in ENTRIES:
        for pat in e["patterns"]:
            for m in pat.finditer(text):
                hits.append((m.start(), m.end(), e))
    hits = [h for h in hits
            if not any(o[2] is not h[2] and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0])
                       for o in hits)]
    seen, out = set(), []
    for _, _, e in sorted(hits, key=lambda h: h[0]):
        if e["name"] not in seen:
            seen.add(e["name"])
            out.append({"name": e["name"], "kind": e["kind"], "section": e["section"], "detail": e["detail"]})
    return out


def find_gaps(text: str, matched_names):
    gaps = []
    for term, pat in GAP_PATTERNS:
        if pat.search(text):
            gaps.append(term)
    return gaps


# ----------------------------------------------------------------------------- semantic similarity
DOCS = PROFILE["documents"]
VECTORIZER = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
DOC_MATRIX = VECTORIZER.fit_transform([d["title"] + ". " + d["text"] for d in DOCS])


def closest_documents(text: str, limit: int = 3):
    sims = cosine_similarity(VECTORIZER.transform([text]), DOC_MATRIX)[0]
    ranked = sorted(range(len(DOCS)), key=lambda i: sims[i], reverse=True)
    out = []
    for i in ranked[:limit]:
        if sims[i] >= MIN_SIMILARITY:
            out.append({"type": DOCS[i]["type"], "title": DOCS[i]["title"], "section": DOCS[i]["section"],
                        "similarity": round(float(sims[i]), 3)})
    return out, (float(sims.max()) if len(sims) else 0.0)


# ----------------------------------------------------------------------------- scoring
def label_for(score: int) -> str:
    if score >= 70:
        return "Strong overlap"
    if score >= 40:
        return "Good overlap"
    if score >= 15:
        return "Some overlap"
    return "Little overlap"


def analyse(text: str) -> dict:
    lowered = text.lower()
    matched = find_matches(lowered)
    gaps = find_gaps(lowered, {m["name"] for m in matched})
    closest, best_sim = closest_documents(text)

    meaning = min(1.0, best_sim / SIM_FULL_MARKS)                 # 0..1: how close in meaning
    asked = len(matched) + len(gaps)                              # technologies the role names that we recognise
    if asked:
        coverage = len(matched) / asked                           # 0..1: share of those technologies she lists
        score = round(100 * (0.7 * coverage + 0.3 * meaning))
        basis = (f"{len(matched)} of the {asked} technologies this role names appear on the portfolio "
                 f"(70% of the score), plus how close the wording is to her projects and courses (30%).")
    else:
        score = round(100 * 0.6 * meaning)                       # no recognisable keywords: lean on meaning only, capped
        basis = ("No specific technologies from my list were recognised, so this score only reflects how close "
                 "the wording is to her projects and courses, and it is capped at 60.")
    score = max(0, min(95, score))                              # a keyword estimate should never claim a perfect fit

    if matched:
        names = ", ".join(m["name"] for m in matched[:4])
        summary = f"The role mentions {names}{' and more' if len(matched) > 4 else ''}, which appear on the portfolio."
    else:
        summary = "None of the portfolio's listed skills are named directly in this role."
    if gaps:
        summary += f" It also asks for {', '.join(gaps[:3])}{' and more' if len(gaps) > 3 else ''}, which are not listed on the portfolio yet."

    return {"score": score, "label": label_for(score), "summary": summary, "basis": basis,
            "matched": matched, "gaps": gaps, "closest": closest}


# ----------------------------------------------------------------------------- tiny rate limiter (in memory)
_hits = defaultdict(deque)


def too_many_requests(visitor: str) -> bool:
    now = time.time()
    q = _hits[visitor]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


# ----------------------------------------------------------------------------- routes
@app.after_request
def add_cors(resp):
    if request.path.startswith("/api/"):
        resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.get("/api/health")
def health():
    return jsonify(status="ok", documents=len(DOCS), skills=len(PROFILE["skills"]))


@app.route("/api/match", methods=["POST", "OPTIONS"])
def match():
    if request.method == "OPTIONS":
        return ("", 204)
    if too_many_requests(request.remote_addr or "unknown"):
        return jsonify(error="Too many requests. Please wait a minute and try again."), 429
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    if not isinstance(text, str):
        return jsonify(error="Please send the job description as text."), 400
    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) < MIN_WORDS:
        return jsonify(error=f"Please paste a little more: at least {MIN_WORDS} words."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"That is a bit long. Please keep it under {MAX_CHARS} characters."), 400
    return jsonify(analyse(text))


# the portfolio itself, so `python app.py` is all you need locally
@app.get("/")
def home():
    return send_from_directory(SITE, "index.html")


@app.get("/assets/<path:name>")
def assets(name):
    return send_from_directory(SITE / "assets", name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
