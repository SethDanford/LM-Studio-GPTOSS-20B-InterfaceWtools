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
import json, re, uuid, os, time
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Flask, request, jsonify
from flask_cors import CORS
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

LMSTUDIO_BASE = os.environ.get("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1")  # must include /v1
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")            # any non-empty string
MODEL = os.environ.get("LMSTUDIO_MODEL", "openai/gpt-oss-20b")               # change to your loaded model
APP_TIMEZONE = os.environ.get("APP_TIMEZONE") or os.environ.get("TZ") or "Asia/Phnom_Penh"
_RESOLVED_MODEL = None
try:
    MAX_TOKENS = int(os.environ.get("LMSTUDIO_MAX_TOKENS", "2048"))
except Exception:
    MAX_TOKENS = 2048
try:
    ANSWER_RETRY_TOKENS = int(os.environ.get("LMSTUDIO_ANSWER_RETRY_TOKENS", "768"))
except Exception:
    ANSWER_RETRY_TOKENS = 768

def get_model_name():
    """Use LMSTUDIO_MODEL when set; otherwise fall back to an available chat model.

    The original app defaulted to GPT-OSS 20B. That is still the preferred
    default, but users often swap the loaded LM Studio model without setting an
    environment variable. In that case, asking LM Studio which model is loaded
    avoids sending chat requests to a stale model id.
    """
    global _RESOLVED_MODEL
    if _RESOLVED_MODEL:
        return _RESOLVED_MODEL
    if os.environ.get("LMSTUDIO_MODEL"):
        _RESOLVED_MODEL = MODEL
        return _RESOLVED_MODEL
    try:
        r = requests.get(f"{LMSTUDIO_BASE}/models", timeout=3)
        r.raise_for_status()
        ids = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
        if MODEL in ids:
            _RESOLVED_MODEL = MODEL
        else:
            chat_models = [mid for mid in ids if "embed" not in mid.lower()]
            _RESOLVED_MODEL = (chat_models or ids or [MODEL])[0]
    except Exception:
        _RESOLVED_MODEL = MODEL
    return _RESOLVED_MODEL

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
  "You may use tools when the user explicitly asks you to browse/search/fetch, "
  "provides a URL, or asks a question that clearly requires current external "
  "information such as today's weather, latest news, current prices, live status, "
  "schedules, or recent events. If the user just chats or asks about stable facts, "
  "do not use any tools. Prefer web_search first for fresh info, and only call fetch_url "
  "after you have a specific article URL; never fetch homepages. When calling tools, "
  "prefer real tool_calls. If tool_calls are not available, emit exactly one JSON object "
  "like {\"name\":\"web_search\",\"arguments\":{\"q\":\"...\"}}. The server also accepts "
  "LM-style inline tags from GPT-OSS-style chat templates. After using tools, "
  "answer in plain text with a short set of bullets and clickable links. Do not "
  "emit channel tags in your final answer."
)}

def current_local_datetime():
    try:
        tz = ZoneInfo(APP_TIMEZONE)
        return datetime.now(tz), APP_TIMEZONE
    except Exception:
        now = datetime.now().astimezone()
        return now, now.tzname() or "local time"

def current_system_message():
    now, zone = current_local_datetime()
    current = now.strftime("%A, %B %-d, %Y at %-I:%M %p")
    return {
        "role": "system",
        "content": (
            f"{SYSTEM['content']}\n\n"
            f"Current local date/time for this chat server: {current} ({zone}). "
            "Use this local date/time directly for date or time questions; do not browse for it."
        ),
    }

def with_current_system(history):
    messages = history[:] if history else []
    system = current_system_message()
    if messages and messages[0].get("role") == "system":
        messages[0] = system
    else:
        messages.insert(0, system)
    return messages

def _history_indicates_current_info(history) -> bool:
    if not history:
        return False
    terms = [
        "weather", "forecast", "temperature", "rain", "storm", "air quality",
        "news", "price", "score", "schedule", "status", "flight", "traffic",
    ]
    recent = []
    for m in reversed(history[-6:]):
        content = _as_text(m.get("content"))
        if content:
            recent.append(content.lower())
    text = " ".join(recent)
    return any(t in text for t in terms)

def should_allow_tools(user_msg: str, history=None) -> bool:
    """Return True when the message likely needs current external information.

    The app should browse for current/fresh requests even when the user did not
    literally say "search the web", while keeping ordinary chat and stable facts
    offline.
    """
    s = (user_msg or "").lower()
    if "http://" in s or "https://" in s:
        return True

    explicit_triggers = [
        "search the web", "web search", "browse the web", "search online",
        "search on the web", "use web_search", "look up", "lookup",
        "search for", "find online", "check online", "google",
        "up to date", "up-to-date",
    ]
    if any(t in s for t in explicit_triggers):
        return True

    answer_from_results_terms = [
        "search results", "just give me an answer", "give me an answer",
        "give an answer", "instead of", "just answer", "answer directly",
    ]
    if _history_indicates_current_info(history) and any(t in s for t in answer_from_results_terms):
        return True

    authoritative_source_terms = [
        "bureau of meteorology", "weather bureau", "bom.gov", "met office",
        "national weather service", "official forecast", "official weather",
    ]
    if any(t in s for t in authoritative_source_terms):
        return True
    if re.search(r"\bbom\b", s) and _history_indicates_current_info(history):
        return True
    if "meteorology" in s and any(t in s for t in ["check", "data", "weather", "forecast", "temperature"]):
        return True
    if any(t in s for t in ["check", "fetch", "pull", "get"]) and any(t in s for t in ["data", "source", "site", "website", "bureau", "official"]):
        if _history_indicates_current_info(history):
            return True

    current_info_topics = [
        "weather", "forecast", "temperature", "rain", "storm", "air quality",
        "news", "headline", "headlines", "breaking",
        "stock", "stocks", "share price", "crypto", "bitcoin", "exchange rate",
        "price", "prices", "score", "scores", "schedule", "traffic",
        "flight", "flights", "status", "open now", "hours today",
    ]
    fresh_terms = [
        "latest", "current", "recent", "today", "tomorrow", "yesterday", "this week",
        "this month", "right now", "at the moment", "currently", "at present",
        "presently", "now", "moment", "live", "new", "updated",
    ]
    if any(topic in s for topic in current_info_topics) and any(term in s for term in fresh_terms):
        return True

    weather_intents = ["weather", "forecast", "temperature", "rain", "storm", "air quality"]
    weather_markers = [
        " the weather", "weather in ", "weather for ", "weather at ", "weather near ",
        "forecast in ", "forecast for ", "temperature in ", "temperature for ",
        "will it rain", "is it raining", "rain in ", "air quality in ",
    ]
    if any(t in s for t in weather_intents) and any(m in s for m in weather_markers):
        return True

    lookup_verbs = ["find", "check", "what's happening", "what is happening"]
    if any(v in s for v in lookup_verbs) and any(t in s for t in fresh_terms):
        return True
    # Default: do not allow tools
    return False

def is_local_datetime_question(user_msg: str) -> bool:
    s = (user_msg or "").lower().strip()
    date_time_phrases = [
        "current date", "today's date", "todays date", "what date is it",
        "what is the date", "what day is it", "what is today",
        "current time", "what time is it", "what is the time",
    ]
    return any(p in s for p in date_time_phrases)

def answer_local_datetime(user_msg: str) -> str:
    now, zone = current_local_datetime()
    s = (user_msg or "").lower()
    date_text = now.strftime("%A, %B %-d, %Y")
    time_text = now.strftime("%-I:%M %p")
    if "time" in s and "date" not in s and "day" not in s:
        return f"The current local time is {time_text} ({zone})."
    if "time" in s:
        return f"The current local date and time is {date_text} at {time_text} ({zone})."
    return f"The current local date is {date_text} ({zone})."

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

def _json_loads_loose(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(raw.replace("'", "\""))

def _iter_json_values(text: str):
    """Yield JSON objects/arrays embedded in model text.

    Qwen-family models sometimes emit a bare object such as
    {"q":"latest news ..."} followed by a stray closing tag. Balanced parsing
    lets us recover the JSON object without caring about the trailing markup.
    """
    s = _as_text(text)
    if not s:
        return
    for start, opener in ((i, ch) for i, ch in enumerate(s) if ch in "{["):
        closer = "}" if opener == "{" else "]"
        stack = [closer]
        in_str = False
        esc = False
        for pos in range(start + 1, len(s)):
            ch = s[pos]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == "\"":
                    in_str = False
                continue
            if ch == "\"":
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or ch != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    raw = s[start:pos + 1]
                    try:
                        yield _json_loads_loose(raw)
                    except Exception:
                        pass
                    break

def _normalize_tool_arguments(name: str, args):
    if not isinstance(args, dict):
        args = {}
    name = _normalize_tool_name(name)
    if name == "web_search":
        q = args.get("q") or args.get("query") or args.get("search_query") or args.get("text")
        out = {"q": str(q or "").strip()}
        max_results = args.get("max_results", args.get("topn", args.get("k", 5)))
        try:
            out["max_results"] = max(1, min(5, int(max_results)))
        except Exception:
            out["max_results"] = 5
        return name, out
    if name == "fetch_url":
        out = {"url": str(args.get("url") or args.get("href") or "").strip()}
        try:
            out["max_chars"] = int(args.get("max_chars", 6000))
        except Exception:
            out["max_chars"] = 6000
        return name, out
    return name, args

def _coerce_json_tool_call(obj, default_name=None):
    if not isinstance(obj, dict):
        return None

    # OpenAI-ish JSON emitted as text:
    # {"name":"web_search","arguments":{"q":"..."}}
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if isinstance(name, dict):
        name = name.get("name")
    args = obj.get("arguments", obj.get("args", obj.get("parameters", obj.get("input"))))
    if isinstance(args, str):
        try:
            args = _json_loads_loose(args)
        except Exception:
            args = {}
    if name:
        name, args = _normalize_tool_arguments(str(name), args if isinstance(args, dict) else obj)
        if name in {"web_search", "fetch_url"}:
            return {"name": name, "arguments": args}

    # Compact forms:
    # {"web_search":{"q":"..."}} or {"fetch_url":{"url":"..."}}
    for candidate in ("web_search", "fetch_url"):
        nested = obj.get(candidate)
        if isinstance(nested, dict):
            name, args = _normalize_tool_arguments(candidate, nested)
            return {"name": name, "arguments": args}

    # Qwen observed form from the UI screenshot:
    # {"q":"latest news Strait of Hormuz"}</>
    if default_name == "web_search" and any(k in obj for k in ("q", "query", "search_query", "text")):
        name, args = _normalize_tool_arguments("web_search", obj)
        if args.get("q"):
            return {"name": name, "arguments": args}
    if default_name == "fetch_url" and any(k in obj for k in ("url", "href")):
        name, args = _normalize_tool_arguments("fetch_url", obj)
        if args.get("url"):
            return {"name": name, "arguments": args}
    return None

def extract_json_tool_calls(text, default_name=None):
    """Extract text-encoded JSON tool calls, including Qwen bare-arg output."""
    calls = []
    seen = set()
    def add_call(call):
        if not call:
            return
        key = json.dumps(call, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        calls.append(call)

    for obj in _iter_json_values(text):
        if isinstance(obj, list):
            for item in obj:
                call = _coerce_json_tool_call(item, default_name=default_name)
                add_call(call)
        elif isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
            for item in obj["tool_calls"]:
                call = _coerce_json_tool_call(item, default_name=default_name)
                add_call(call)
        else:
            call = _coerce_json_tool_call(obj, default_name=default_name)
            add_call(call)
    return calls

def extract_xml_tool_calls(text):
    """Extract Qwen-style XML-ish tool calls.

    Example:
      <tool_call>
      <function=fetch_url>
      <parameter=url>https://example.com</parameter>
      </function>
      </tool_call>
    """
    s = _as_text(text)
    if not s or "<function=" not in s:
        return []

    calls = []
    function_re = re.compile(
        r"<function=([A-Za-z0-9_]+)>\s*(.*?)\s*</function>",
        flags=re.S,
    )
    parameter_re = re.compile(
        r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>",
        flags=re.S,
    )
    for m in function_re.finditer(s):
        name = _normalize_tool_name(m.group(1))
        body = m.group(2)
        args = {}
        for p in parameter_re.finditer(body):
            key = p.group(1).strip()
            value = strip_lm_tags(p.group(2)).strip()
            args[key] = value
        if name in {"web_search", "fetch_url"}:
            name, args = _normalize_tool_arguments(name, args)
            calls.append({"name": name, "arguments": args})
    return calls

def extract_text_tool_calls(text, default_name=None):
    """Extract any non-OpenAI tool-call format we know how to recover."""
    return (
        extract_tool_markups(text)
        or extract_xml_tool_calls(text)
        or extract_json_tool_calls(text, default_name=default_name)
    )

def do_web_search(q, k):
    q = (q or "").strip()
    if not q:
        return [{"error": "web_search requires a non-empty query"}]
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

def run_tool_calls(calls, messages, accum, call_id_prefix):
    for i, c in enumerate(calls):
        name, args = _normalize_tool_arguments(c.get("name"), c.get("arguments", {}))
        if name == "web_search":
            try:
                out = do_web_search(args.get("q",""), args.get("max_results", 5))
            except Exception as e:
                out = [{"error": f"web_search failed: {e}"}]
            accum["web"] = out
        elif name == "fetch_url":
            try:
                out = do_fetch_url(args.get("url"), int(args.get("max_chars",6000)))
            except Exception as e:
                out = {"error": f"fetch_url failed: {e}"}
            accum["fetched"].append(out)
        else:
            out = {"error":"unknown tool"}
        messages.append({"role":"tool","tool_call_id":c.get("id") or f"{call_id_prefix}-{i}","name":name,"content":json.dumps(out)})

def _extract_weather_location(text):
    s = _as_text(text)
    patterns = [
        r"\b(?:in|for|at|near)\s+([A-Za-z][A-Za-z\s.'-]{1,60}?)(?:\s+(?:on|today|tomorrow|this|at|right|currently|please|could|will|be|like)\b|[?.!,]|$)",
        r"\b([A-Za-z][A-Za-z\s.'-]{1,60})\s+weather\b",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            loc = re.sub(r"\s+", " ", m.group(1)).strip(" .,'-")
            if loc:
                return loc.title()
    return ""

def _extract_weather_date(text):
    s = _as_text(text).lower()
    now, _ = current_local_datetime()
    month_name = now.strftime("%B")
    if "today" in s:
        return now.strftime("%B %-d %Y")
    if "tomorrow" in s:
        # Avoid importing timedelta just for this narrow case.
        from datetime import timedelta
        return (now + timedelta(days=1)).strftime("%B %-d %Y")
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+of\s+this\s+month\b", s)
    if m:
        return f"{month_name} {int(m.group(1))} {now.year}"
    m = re.search(r"\bon\s+the\s+(\d{1,2})(?:st|nd|rd|th)?\b", s)
    if m:
        return f"{month_name} {int(m.group(1))} {now.year}"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(\d{4})\b", s)
    if m:
        return f"{m.group(2).title()} {int(m.group(1))} {m.group(3)}"
    return ""

def rewrite_current_info_query(base_text):
    s = _as_text(base_text)
    low = s.lower()
    if any(t in low for t in ["weather", "forecast", "temperature", "rain"]):
        loc = _extract_weather_location(s)
        date = _extract_weather_date(s)
        if loc and date:
            return f"{loc} weather forecast {date}"
        if loc and any(t in low for t in ["current", "currently", "right now", "at the moment", "temperature"]):
            return f"current temperature weather {loc}"
        if loc:
            return f"{loc} weather forecast"
    return s.strip()

def build_search_query(user_msg, messages):
    s = (user_msg or "").strip()
    followup_markers = [
        "check", "data", "source", "site", "website", "bureau", "official",
        "answer", "instead", "search results", "that", "there", "it",
    ]
    if any(m in s.lower() for m in followup_markers):
        for m in reversed(messages[:-1]):
            if m.get("role") != "user":
                continue
            prev = _as_text(m.get("content")).strip()
            if prev and _history_indicates_current_info([m]):
                return rewrite_current_info_query(f"{prev} {s}")
    return rewrite_current_info_query(s)

def force_web_search(user_msg, messages, accum):
    """Run a search when a fresh-info prompt is allowed but the model skips tools."""
    q = build_search_query(user_msg, messages)
    try:
        out = do_web_search(q, 5)
    except Exception as e:
        out = [{"error": f"web_search failed: {e}"}]
    accum["web"] = out
    messages.append({
        "role": "system",
        "content": (
            "Web search results for the user's current-information request:\n"
            f"{json.dumps(out)}"
        ),
    })

def handle_tool_response(msg, messages, accum, call_id_prefix):
    if not msg:
        return None

    # Prefer standard OpenAI tool_calls when LM Studio/model provides them.
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        messages.append({"role":"assistant","tool_calls": tool_calls})
        calls = []
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            name = fn.get("name")
            arg_str = fn.get("arguments") or "{}"
            try:
                args = _json_loads_loose(arg_str)
            except Exception:
                args = {}
            calls.append({"id": tc.get("id"), "name": name, "arguments": args})
        run_tool_calls(calls, messages, accum, call_id_prefix)
        return True

    content = _as_text(msg.get("content"))
    calls = extract_text_tool_calls(content, default_name="web_search")
    if calls:
        messages.append({"role":"assistant","content": content})
        run_tool_calls(calls, messages, accum, call_id_prefix)
        return True
    return False

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

def chat_once(messages, allow_tools=True, metrics=None, max_tokens=None):
    """Call LM Studio's OpenAI-compatible endpoint via HTTP.
    We bypass the OpenAI SDK to avoid client-side validation issues.
    """
    url = f"{LMSTUDIO_BASE}/chat/completions"
    model_name = get_model_name()
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens or MAX_TOKENS,
    }
    if allow_tools:
        payload["tools"] = TOOLS
    headers = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}", "Content-Type": "application/json"}
    started = time.perf_counter()
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=90)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach LM Studio at {LMSTUDIO_BASE}. "
            f"Ensure the LM Studio server is running and a model is loaded. "
            f"Original error: {e}"
        )
    elapsed = time.perf_counter() - started
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"].get("message") or str(j["error"]))
    choice = (j.get("choices") or [{}])[0]
    if metrics is not None:
        usage = j.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        reasoning_tokens = int(details.get("reasoning_tokens") or 0)
        thinking_seconds = None
        if completion_tokens > 0 and reasoning_tokens > 0:
            thinking_seconds = elapsed * min(1.0, reasoning_tokens / completion_tokens)
        metrics.append({
            "model": j.get("model") or model_name,
            "elapsed_seconds": elapsed,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or 0),
            "reasoning_tokens": reasoning_tokens,
            "thinking_seconds": thinking_seconds,
            "tokens_per_second": (completion_tokens / elapsed) if elapsed > 0 and completion_tokens else None,
            "stop_reason": choice.get("finish_reason"),
        })
    return choice.get("message", {})

def summarize_metrics(model_metrics, turn_started, source="model"):
    response_seconds = time.perf_counter() - turn_started
    prompt_tokens = sum(m.get("prompt_tokens") or 0 for m in model_metrics)
    completion_tokens = sum(m.get("completion_tokens") or 0 for m in model_metrics)
    total_tokens = sum(m.get("total_tokens") or 0 for m in model_metrics)
    reasoning_tokens = sum(m.get("reasoning_tokens") or 0 for m in model_metrics)
    model_seconds = sum(m.get("elapsed_seconds") or 0 for m in model_metrics)
    thinking_values = [m.get("thinking_seconds") for m in model_metrics if m.get("thinking_seconds") is not None]
    tokens_per_second = (completion_tokens / model_seconds) if model_seconds > 0 and completion_tokens else None
    stop_reason = next((m.get("stop_reason") for m in reversed(model_metrics) if m.get("stop_reason")), None)
    return {
        "source": source,
        "model": next((m.get("model") for m in reversed(model_metrics) if m.get("model")), get_model_name()),
        "model_calls": len(model_metrics),
        "response_seconds": response_seconds,
        "model_seconds": model_seconds,
        "thinking_seconds": sum(thinking_values) if thinking_values else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "tokens_per_second": tokens_per_second,
        "stop_reason": stop_reason or source,
    }

def _shorten_text(text, limit=1800):
    s = _as_text(text).strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[truncated]"

def build_answer_retry_messages(messages, user_msg):
    context = []
    for m in messages[-10:]:
        role = m.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        content = _shorten_text(m.get("content"), 1400)
        if not content:
            continue
        if role == "system" and not (
            "Web search results" in content
            or "Fetched pages" in content
            or "Using the gathered" in content
        ):
            continue
        context.append(f"{role.upper()}:\n{content}")
    return [
        current_system_message(),
        {
            "role": "system",
            "content": (
                "Answer-only recovery mode. /no_think\n"
                "The previous model call produced no visible answer, likely because it spent "
                "the token budget on reasoning. Do not reason out loud, do not call tools, "
                "and do not return search-result lists unless no answer can be inferred. "
                "Use the context below to give the user the clearest concise answer possible. "
                "If the exact value or forecast is not present, state that briefly and say what "
                "the available sources support."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Latest user request:\n{user_msg}\n\n"
                f"Recent context:\n\n{chr(10).join(context)}\n\n"
                "Return only the final answer. /no_think"
            ),
        },
    ]

def tool_loop(user_msg, history, collect_meta=False):
    turn_started = time.perf_counter()
    model_metrics = []

    def finish(final, source="model"):
        meta = summarize_metrics(model_metrics, turn_started, source=source)
        if collect_meta:
            return messages, final, meta
        return messages, final

    def ask(allow_tools=True):
        return chat_once(messages, allow_tools=allow_tools, metrics=model_metrics)

    def retry_answer_only():
        retry_messages = build_answer_retry_messages(messages, user_msg)
        retry_msg = chat_once(
            retry_messages,
            allow_tools=False,
            metrics=model_metrics,
            max_tokens=ANSWER_RETRY_TOKENS,
        )
        return strip_lm_tags(_as_text(retry_msg.get("content"))) if retry_msg else ""

    messages = with_current_system(history)
    messages.append({"role":"user","content":user_msg})
    if is_local_datetime_question(user_msg):
        final = answer_local_datetime(user_msg)
        messages.append({"role":"assistant","content": final})
        return finish(final, source="local")
    allow_tools = should_allow_tools(user_msg, messages)
    msg = ask(allow_tools=allow_tools)
    accum = {"web": None, "fetched": []}

    if allow_tools and handle_tool_response(msg, messages, accum, "tool"):
        messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
        msg = ask(allow_tools=False)
    elif allow_tools and "to=functions." in _as_text(msg.get("content", "")):
        # Heuristic: model tried to call a tool but we couldn't parse it. Nudge once,
        # then process that second response instead of exposing tool markup to the UI.
        messages.append({"role":"assistant","content": _as_text(msg.get("content"))})
        messages.append({"role":"system","content":"If you need tools, emit a valid tool_call or a JSON object like {\"name\":\"web_search\",\"arguments\":{\"q\":\"...\"}}. Otherwise, answer directly in plain text."})
        msg = ask(allow_tools=True)
        if handle_tool_response(msg, messages, accum, "tool-retry"):
            messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
            msg = ask(allow_tools=False)
        else:
            force_web_search(user_msg, messages, accum)
            messages.append({"role":"system","content":"Using the gathered web search results, answer in plain text with bullet headlines and links. Do not call tools."})
            msg = ask(allow_tools=False)
    elif allow_tools:
        force_web_search(user_msg, messages, accum)
        messages.append({"role":"system","content":"Using the gathered web search results, answer in plain text with bullet headlines and links. Do not call tools."})
        msg = ask(allow_tools=False)

    final = strip_lm_tags(_as_text(msg.get("content"))) if msg else ""
    # Fallback: If the assistant still emitted recoverable tool text, process it.
    # Some Qwen templates keep emitting XML-ish tool blocks even after tool output;
    # keep this bounded so raw tool tags do not reach the UI.
    cleanup_rounds = 0
    while allow_tools and cleanup_rounds < 3:
        calls = extract_text_tool_calls(final, default_name="web_search")
        if not calls:
            break
        messages.append({"role":"assistant","content": final})
        run_tool_calls(calls, messages, accum, f"tool-fallback-{cleanup_rounds}")
        messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
        msg = ask(allow_tools=False)
        final = strip_lm_tags(_as_text(msg.get("content"))) if msg else final
        cleanup_rounds += 1
    if allow_tools and extract_text_tool_calls(final, default_name="web_search") and (accum["web"] or accum["fetched"]):
        final = render_fallback_summary(accum)
    # If final is empty, first try a compact answer-only retry from accumulated
    # results. If Qwen still returns no visible content, synthesize a clean
    # summary so the UI never shows a blank response.
    if not (final or "").strip() and (accum["web"] or accum["fetched"]):
        final = retry_answer_only()
        if not (final or "").strip():
            final = render_fallback_summary(accum)
    # Still empty? Nudge once for a plain-text answer without tools
    if not (final or "").strip():
        messages.append({"role":"assistant","content": ""})
        messages.append({"role":"system","content": "Answer now in plain text with short bullets or a short paragraph. No tags. /no_think"})
        msg = ask(allow_tools=False)
        final = strip_lm_tags(_as_text(msg.get("content"))) if msg else final
    if not (final or "").strip():
        final = retry_answer_only()
    # If our heuristic missed a fresh-info request but the model still emitted
    # a recoverable tool request, execute it instead of showing a permission nudge.
    if not allow_tools:
        calls = extract_text_tool_calls(final, default_name="web_search")
        if calls:
            messages.append({"role":"assistant","content": final})
            run_tool_calls(calls, messages, accum, "tool-late")
            messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
            msg = ask(allow_tools=False)
            final = strip_lm_tags(_as_text(msg.get("content"))) if msg else final
        elif "to=functions." in (final or "") or "<tool_call" in (final or ""):
            force_web_search(user_msg, messages, accum)
            messages.append({"role":"system","content":"Using the gathered web search results, answer in plain text with bullet headlines and links. Do not call tools."})
            msg = ask(allow_tools=False)
            final = strip_lm_tags(_as_text(msg.get("content"))) if msg else final
    if not (final or "").strip():
        final = retry_answer_only()
    if not (final or "").strip():
        final = "I couldn't get a response from the model. Try rephrasing or ask me to check a specific source."
    messages.append({"role":"assistant","content": final})
    return finish(final)

# -------- Flask app --------
app = Flask(__name__, static_url_path="", static_folder="static")
CORS(app)

SESSIONS = {}  # in-memory chat state

def get_session(thread):
    existing = SESSIONS.get(thread)
    if isinstance(existing, dict):
        existing.setdefault("history", [])
        existing.setdefault("display", [])
        return existing
    if isinstance(existing, list):
        session = {"history": existing, "display": []}
        SESSIONS[thread] = session
        return session
    session = {"history": [], "display": []}
    SESSIONS[thread] = session
    return session

def rebuild_history_from_display(session):
    rebuilt = [current_system_message()]
    for item in session.get("display", []):
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and content:
            rebuilt.append({"role": role, "content": content})
    session["history"] = rebuilt
    return rebuilt

@app.post("/chat")
def chat():
    try:
        data = request.get_json(force=True)
        thread = data.get("thread") or str(uuid.uuid4())
        user_msg = data.get("message","")
        if not user_msg:
            return jsonify({"thread": thread, "answer": "Please enter a message."})
        session = get_session(thread)
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        history, answer, meta = tool_loop(user_msg, session.get("history", []), collect_meta=True)
        session["history"] = history
        session["display"].append({"id": user_id, "role": "user", "content": user_msg})
        session["display"].append({"id": assistant_id, "role": "assistant", "content": answer, "meta": meta})
        return jsonify({
            "thread": thread,
            "answer": answer,
            "meta": meta,
            "user_id": user_id,
            "assistant_id": assistant_id,
        })
    except Exception as e:
        # Always return a clean JSON response to keep UI stable
        return jsonify({"thread": data.get("thread") if isinstance(data, dict) else str(uuid.uuid4()),
                        "answer": f"Sorry, I hit an error: {e}"})

@app.post("/message/delete")
def delete_message():
    data = request.get_json(force=True)
    thread = data.get("thread")
    message_id = data.get("id")
    if not thread or not message_id or thread not in SESSIONS:
        return jsonify({"ok": False, "error": "Unknown thread or message."}), 404
    session = get_session(thread)
    before = len(session.get("display", []))
    session["display"] = [m for m in session.get("display", []) if m.get("id") != message_id]
    rebuild_history_from_display(session)
    return jsonify({"ok": len(session["display"]) != before, "removed": [message_id]})

@app.post("/message/truncate")
def truncate_message():
    data = request.get_json(force=True)
    thread = data.get("thread")
    message_id = data.get("id")
    if not thread or not message_id or thread not in SESSIONS:
        return jsonify({"ok": False, "error": "Unknown thread or message."}), 404
    session = get_session(thread)
    display = session.get("display", [])
    idx = next((i for i, m in enumerate(display) if m.get("id") == message_id), None)
    if idx is None:
        return jsonify({"ok": False, "error": "Unknown message."}), 404
    removed = [m.get("id") for m in display[idx:]]
    session["display"] = display[:idx]
    rebuild_history_from_display(session)
    return jsonify({"ok": True, "removed": removed})

@app.get("/")
def root():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", "7000")), debug=False)
