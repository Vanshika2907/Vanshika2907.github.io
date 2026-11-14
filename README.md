# Role Match: Python backend

A visitor pastes a job description into the **Role Match** section of the portfolio. The browser sends it to this
Flask service, which compares it with the portfolio's own content and returns a score, matched skills, honest gaps and
the closest projects and courses.

## Run it on your computer

```
cd backend
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**. The same server also serves the portfolio, so nothing else is needed.
Scroll to *Role Match*, click **Try an example**, then **Check the fit**.

## How it works (for your viva)

| Step | What happens | Where |
|---|---|---|
| 1. Validate | Rejects text under 5 words or over 6000 characters, non-text input and bad JSON. | `match()` in `app.py` |
| 2. Rate limit | Max 20 requests per minute per visitor, held in memory. | `too_many_requests()` |
| 3. Keyword detection | Whole-word regular expressions find the skills and topics from `profile.json` (so `java` does not fire inside `javascript`). A list of other common technologies finds what the role asks for that the portfolio does not list. | `find_matches()`, `find_gaps()` |
| 4. Meaning comparison | **TF-IDF** turns the job description and each project, course and experience entry into vectors. **Cosine similarity** ranks the closest ones. Library: scikit-learn. | `closest_documents()` |
| 5. Score | `70% x (technologies matched / technologies the role names) + 30% x (closest similarity, scaled)`. Capped at 95. Labels: 70+ strong, 40+ good, 15+ some, below that little. | `analyse()` |

**Why it is honest:** it only uses what is in `profile.json`, which mirrors the portfolio. Anything the role asks for
that is not listed is shown as a gap, not hidden. The page says it is a keyword and wording comparison, not a hiring verdict.

**Limits to mention if asked:** it matches words and similar wording, not true understanding. For example it will not
know that "data wrangling" relates to Python. It does not remember or store what visitors paste. Possible upgrades:
sentence embeddings instead of TF-IDF, and a database to log anonymous scores.

## API

- `GET /api/health` returns `{"status": "ok", ...}`
- `POST /api/match` with `{"text": "<job description>"}` returns `score`, `label`, `summary`, `basis`, `matched`, `gaps`, `closest`.
  Errors return `{"error": "..."}` with status 400 (bad input) or 429 (too many requests).

## Keeping it up to date

When you add a skill, project or course to the page, add it to `profile.json` too (the `skills`, `topics` or
`documents` lists). No code changes are needed.

## Putting it online (optional)

GitHub Pages cannot run Python, so the backend needs its own free host (for example Render or PythonAnywhere):

1. Upload the project to GitHub (the whole folder, with `backend/` inside it).
2. Create a web service on the host. Root directory `backend`, install command `pip install -r requirements.txt`,
   start command `gunicorn app:app`.
3. Set the environment variable `ALLOWED_ORIGIN` to your portfolio address, for example `https://vanshika2907.github.io`.
4. In `index.html`, set `matchApi` in `CONFIG` to the address the host gives you.

Free hosts go to sleep when idle, so the first request after a quiet spell can take up to a minute. The page waits
20 seconds and then shows a friendly message.
