"""
Lightweight web UI server that talks to LM Studio (local OpenAI-compatible API)
and gives the model access to two tools:
 - web_search: DuckDuckGo search via ddgs
 - fetch_url: fetch and extract readable text from a URL

Key changes:
 - Removed OpenAI SDK usage and switched to plain HTTP calls to LM Studio. This
   avoids strict client-side validation errors like "The string did not match
   the expected pattern" and works reliably with community models.
 - Handles both LM Studio's inline tool markup (<|...|> tags) and standard
   OpenAI tool_calls responses.
 - Adds robust error handling so the frontend always gets a clean text answer.
"""

# pip install flask flask-cors ddgs requests beautifulsoup4
import json, re, uuid, os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

LMSTUDIO_BASE = os.environ.get("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1")  # must include /v1
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")            # any non-empty string
MODEL = os.environ.get("LMSTUDIO_MODEL", "openai/gpt-oss-20b")               # change to your loaded model

TOOLS = [
  {"type":"function","function":{
    "name":"web_search",
    "description":"Search the web for up-to-date results.",
    "parameters":{
      "type":"object",
      "properties":{
        "q":{"type":"string","description":"query terms"},
        "max_results":{"type":"integer","minimum":1,"maximum":5,"default":5}
      },
      "required":["q"],
      "additionalProperties": False
    }
  }},
  {"type":"function","function":{
    "name":"fetch_url",
    "description":"Fetch page text from a specific article URL.",
    "parameters":{
      "type":"object",
      "properties":{
        "url":{"type":"string"},
        "max_chars":{"type":"integer","default":6000}
      },
      "required":["url"],
      "additionalProperties": False
    }
  }}
]

SYSTEM = {"role":"system","content":(
  "You may use tools, but only when the user explicitly asks you to browse, "
  "search the web, or fetch a specific URL. If the user just chats, do not use "
  "any tools. Prefer web_search first for fresh info, and only call fetch_url "
  "after you have a specific article URL; never fetch homepages. When calling "
  "tools, you may either: (1) use tool_calls; or (2) emit LM-style inline tags "
  "like '<|assistant|>\n<|commentary to=functions.web_search|><|message|>{\"q\":\"...\"}'. After using tools, "
  "answer in plain text with a short set of bullets and clickable links. Do not "
  "emit channel tags in your final answer."
)}

def should_allow_tools(user_msg: str) -> bool:
    """Return True only if the user explicitly asked to browse/search/fetch.
    Triggers when the message contains a clear intent or includes a URL.
    """
    s = (user_msg or "").lower()
    explicit_triggers = [
        "search the web", "web search", "browse the web", "search online",
        "search on the web", "use web_search",
    ]
    if any(t in s for t in explicit_triggers):
        return True
    if "http://" in s or "https://" in s:
        return True
    # Default: do not allow tools
    return False

def _as_text(content):
    if isinstance(content, list):
        parts=[]
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])  # OpenAI content blocks
            else:
                parts.append(str(p))
        return "".join(parts)
    return content or ""

_TAG_RE = re.compile(r"(?:<\|[^>|]+\|>|\|start\||\|assistant\||\|message\||\|system\||\|user\|)")

def strip_lm_tags(s: str) -> str:
    if not s: return s
    return _TAG_RE.sub("", s)

def _normalize_tool_name(name: str) -> str:
    n = (name or "").strip().rstrip(".,;:|>")
    # Some models glue 'json' onto the function name (e.g., web_searchjson)
    if n.lower().endswith("json"):
        cand = n[:-4]
        if cand:
            n = cand
    return n

def extract_tool_markups(text):
    """Extract tool calls written in LM Studio inline style or relaxed text.

    Supports both of these shapes (and similar variants):
      <|assistant|><|commentary to=functions.web_search|><|message|>{"q":"foo"}
      assistantcommentary to=functions.web_searchjson{"q":"foo"}

    Returns a list of {name, arguments} dictionaries.
    """
    s = _as_text(text)
    if not s or "to=functions." not in s:
        return []

    out = []

    # 1) Classic tag form with explicit <|message|>{...}
    pat1 = r"to=functions\.([A-Za-z0-9_]+).*?(?:<\|message\|>|\|message\|>)(\{.*?\})(?=(?:<\||\|start\||$))"
    for m in re.finditer(pat1, s, flags=re.S):
        arg_str = (m.group(2) or "{}").strip()
        try:
            args = json.loads(arg_str)
        except Exception:
            try:
                args = json.loads(arg_str.replace("'","\""))
            except Exception:
                args = {}
        out.append({"name": _normalize_tool_name(m.group(1)), "arguments": args})

    if out:
        return out

    # 2) Relaxed form: ... to=functions.NAME ... json {...}
    #    We locate each occurrence of to=functions.NAME then parse the nearest JSON object
    text_s = s
    i = 0
    name_pat = re.compile(r"to=functions\.([A-Za-z0-9_]+)")
    while True:
        m = name_pat.search(text_s, i)
        if not m:
            break
        name = _normalize_tool_name(m.group(1))
        # Find the start of the next JSON object after the match
        jstart = text_s.find('{', m.end())
        if jstart == -1:
            i = m.end();
            continue
        # Parse a balanced JSON object from jstart
        depth = 0
        in_str = False
        esc = False
        jend = None
        for k in range(jstart, len(text_s)):
            ch = text_s[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        jend = k + 1
                        break
        if jend is None:
            i = m.end()
            continue
        arg_str = text_s[jstart:jend]
        try:
            args = json.loads(arg_str)
        except Exception:
            try:
                args = json.loads(arg_str.replace("'","\""))
            except Exception:
                args = {}
        out.append({"name": name, "arguments": args})
        i = jend

    return out

def do_web_search(q, k):
    hits = list(DDGS().text(q, max_results=k))
    out = []
    for h in hits[:k]:
        snip = (h.get("body") or "")
        if len(snip) > 140: snip = snip[:140] + "..."
        out.append({"title":h.get("title"), "href":h.get("href"), "snippet":snip})
    return out

def do_fetch_url(url, limit):
    r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)[:limit]
    return {"url":url, "text":text, "truncated_to":limit}

def render_fallback_summary(accum: dict) -> str:
    """Produce a readable plaintext summary if the model returns empty text."""
    web = accum.get("web") or []
    fetched = accum.get("fetched") or []
    lines = []
    if web:
        lines.append("Top results:")
        for r in web[:5]:
            title = (r.get("title") or "Untitled").strip()
            url = r.get("href") or ""
            snippet = (r.get("snippet") or "").strip()
            if snippet:
                lines.append(f"- {title} — {url}\n  {snippet}")
            else:
                lines.append(f"- {title} — {url}")
    if fetched:
        lines.append("\nFetched pages:")
        for f in fetched[:3]:
            url = f.get("url") or ""
            text = (f.get("text") or "").strip()
            if len(text) > 220:
                text = text[:220] + "..."
            lines.append(f"- {url}\n  {text}")
    if not lines:
        return "I couldn't generate a summary from tools. Try rephrasing or provide a URL."
    return "\n".join(lines)

def chat_once(messages, allow_tools=True):
    """Call LM Studio's OpenAI-compatible endpoint via HTTP.
    We bypass the OpenAI SDK to avoid client-side validation issues.
    """
    url = f"{LMSTUDIO_BASE}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 384,
    }
    if allow_tools:
        payload["tools"] = TOOLS
    headers = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=90)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach LM Studio at {LMSTUDIO_BASE}. "
            f"Ensure the LM Studio server is running and a model is loaded. "
            f"Original error: {e}"
        )
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"].get("message") or str(j["error"]))
    choice = (j.get("choices") or [{}])[0]
    return choice.get("message", {})

def tool_loop(user_msg, history):
    messages = history[:] or [SYSTEM]
    messages.append({"role":"user","content":user_msg})
    # Only allow tools when explicitly requested
    allow_tools = should_allow_tools(user_msg)
    # one tool round max to keep within budget
    msg = chat_once(messages, allow_tools=allow_tools)
    accum = {"web": None, "fetched": []}

    # Handle LM-style inline tags (primary path)
    calls = extract_tool_markups(msg.get("content")) if (msg and allow_tools) else []
    if calls:
        messages.append({"role":"assistant","content": _as_text(msg.get("content"))})
        for i, c in enumerate(calls):
            if c["name"] == "web_search":
                out = do_web_search(c["arguments"].get("q",""), min(5, c["arguments"].get("max_results",5)))
                accum["web"] = out
            elif c["name"] == "fetch_url":
                out = do_fetch_url(c["arguments"]["url"], int(c["arguments"].get("max_chars",6000)))
                accum["fetched"].append(out)
            else:
                out = {"error":"unknown tool"}
            messages.append({"role":"tool","tool_call_id":f"markup-{i}","name":c["name"],"content":json.dumps(out)})
        messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
        msg = chat_once(messages, allow_tools=False)
    elif allow_tools and ("to=functions." in _as_text(msg.get("content", ""))):
        # Heuristic: model tried to call a tool but we couldn't parse it. Nudge once.
        messages.append({"role":"assistant","content": _as_text(msg.get("content"))})
        messages.append({"role":"system","content":"If you need tools, emit a valid tool call (tool_calls) or LM-style inline tag with a JSON object. Otherwise, answer directly in plain text."})
        msg = chat_once(messages, allow_tools=True)

    # Handle OpenAI tool_calls
    elif (allow_tools and msg.get("tool_calls")):
        tool_calls = msg.get("tool_calls") or []
        messages.append({"role":"assistant","tool_calls": tool_calls})
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            name = fn.get("name")
            arg_str = fn.get("arguments") or "{}"
            try:
                args = json.loads(arg_str)
            except Exception:
                args = {}
            if name == "web_search":
                out = do_web_search(args.get("q",""), min(5, args.get("max_results",5)))
                accum["web"] = out
            elif name == "fetch_url":
                out = do_fetch_url(args.get("url"), int(args.get("max_chars",6000)))
                accum["fetched"].append(out)
            else:
                out = {"error":"unknown tool"}
            messages.append({"role":"tool","tool_call_id":tc.get("id","tc-0"),"name":name,"content":json.dumps(out)})
        messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
        msg = chat_once(messages, allow_tools=False)

    final = strip_lm_tags(_as_text(msg.get("content"))) if msg else ""
    # Fallback: If the assistant still emitted inline tool tags, process them now
    if allow_tools and "to=functions." in (final or ""):
        calls = extract_tool_markups(final)
        if calls:
            messages.append({"role":"assistant","content": final})
            for i, c in enumerate(calls):
                if c["name"] == "web_search":
                    out = do_web_search(c["arguments"].get("q",""), min(5, c["arguments"].get("max_results",5)))
                    accum["web"] = out
                elif c["name"] == "fetch_url":
                    out = do_fetch_url(c["arguments"].get("url"), int(c["arguments"].get("max_chars",6000)))
                    accum["fetched"].append(out)
                else:
                    out = {"error":"unknown tool"}
                messages.append({"role":"tool","tool_call_id":f"markup-fb-{i}","name":c["name"],"content":json.dumps(out)})
            messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
            msg = chat_once(messages, allow_tools=False)
            final = strip_lm_tags(_as_text(msg.get("content"))) if msg else final
    # If final is empty, synthesize a clean summary from accumulated results
    if not (final or "").strip() and (accum["web"] or accum["fetched"]):
        final = render_fallback_summary(accum)
    # Still empty? Nudge once for a plain-text answer without tools
    if not (final or "").strip():
        messages.append({"role":"assistant","content": ""})
        messages.append({"role":"system","content": "Answer now in plain text with short bullets or a short paragraph. No tags."})
        msg = chat_once(messages, allow_tools=False)
        final = strip_lm_tags(_as_text(msg.get("content"))) if msg else final
    # If tools were not allowed but the model emitted LM-style tags, replace with a nudge
    if not allow_tools and "to=functions." in (final or ""):
        final = (
            "I won't browse unless you ask. Say 'search the web for …' "
            "or provide a URL to fetch."
        )
    messages.append({"role":"assistant","content": final})
    return messages, final

# -------- Flask app --------
app = Flask(__name__, static_url_path="", static_folder="static")
CORS(app)

SESSIONS = {}  # in-memory chat state

@app.post("/chat")
def chat():
    try:
        data = request.get_json(force=True)
        thread = data.get("thread") or str(uuid.uuid4())
        user_msg = data.get("message","")
        if not user_msg:
            return jsonify({"thread": thread, "answer": "Please enter a message."})
        history = SESSIONS.get(thread, [])
        history, answer = tool_loop(user_msg, history)
        SESSIONS[thread] = history
        return jsonify({"thread": thread, "answer": answer})
    except Exception as e:
        # Always return a clean JSON response to keep UI stable
        return jsonify({"thread": data.get("thread") if isinstance(data, dict) else str(uuid.uuid4()),
                        "answer": f"Sorry, I hit an error: {e}"})

@app.get("/")
def root():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(port=7000, debug=False)
