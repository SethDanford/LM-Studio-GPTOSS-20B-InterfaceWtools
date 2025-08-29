# pip install flask flask-cors openai ddgs requests beautifulsoup4
import json, re, uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

LMSTUDIO_BASE = "http://127.0.0.1:1234/v1"   # LM Studio server
LMSTUDIO_API_KEY = "lm-studio"
MODEL = "openai/gpt-oss-20b"                 # change to your loaded model

client = OpenAI(base_url=LMSTUDIO_BASE, api_key=LMSTUDIO_API_KEY)

TOOLS = [
  {"type":"function","function":{
    "name":"web_search",
    "description":"Search web.",
    "parameters":{"type":"object","properties":{
      "q":{"type":"string"},
      "max_results":{"type":"integer","minimum":1,"maximum":5,"default":5}
    },"required":["q"],"additionalProperties":False}
  }},
  {"type":"function","function":{
    "name":"fetch_url",
    "description":"Get text from URL.",
    "parameters":{"type":"object","properties":{
      "url":{"type":"string","format":"uri"},
      "max_chars":{"type":"integer","default":6000}
    },"required":["url"],"additionalProperties":False}
  }}
]

SYSTEM = {"role":"system","content":
  "You can call tools. Use web_search first. Only call fetch_url on article URLs, not homepages. "
  "After tools, answer in plain text with short bullets and clickable links. Do not emit channel tags."
}

def extract_tool_markups(text):
    pat = r"to=functions\.([A-Za-z0-9_]+).*?(?:<\|message\|>|\|message\|>)(\{.*?\})(?=(?:<\||\|start\||$))"
    s = text if isinstance(text, str) else "".join(p.get("text","") if isinstance(p,dict) else str(p) for p in text)
    return [{"name":m.group(1), "arguments":json.loads(m.group(2))} for m in re.finditer(pat, s, flags=re.S)]

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

def chat_once(messages, allow_tools=True):
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS if allow_tools else None, temperature=0.2, max_tokens=384
    )
    return resp.choices[0].message

def tool_loop(user_msg, history):
    messages = history[:] or [SYSTEM]
    messages.append({"role":"user","content":user_msg})
    # one tool round max to keep within 4k
    msg = chat_once(messages, allow_tools=True)

    # Handle LM-style inline tags
    calls = extract_tool_markups(getattr(msg, "content", "")) if msg else []
    if calls:
        messages.append({"role":"assistant","content":getattr(msg,"content","")})
        for i, c in enumerate(calls):
            if c["name"] == "web_search":
                out = do_web_search(c["arguments"].get("q",""), min(5, c["arguments"].get("max_results",5)))
            elif c["name"] == "fetch_url":
                out = do_fetch_url(c["arguments"]["url"], int(c["arguments"].get("max_chars",6000)))
            else:
                out = {"error":"unknown tool"}
            messages.append({"role":"tool","tool_call_id":f"markup-{i}","name":c["name"],"content":json.dumps(out)})
        messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
        msg = chat_once(messages, allow_tools=False)

    # Handle OpenAI tool_calls
    elif getattr(msg, "tool_calls", None):
        messages.append({"role":"assistant","tool_calls": msg.tool_calls})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if tc.function.name == "web_search":
                out = do_web_search(args["q"], min(5, args.get("max_results",5)))
            elif tc.function.name == "fetch_url":
                out = do_fetch_url(args["url"], int(args.get("max_chars",6000)))
            else:
                out = {"error":"unknown tool"}
            messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.function.name,"content":json.dumps(out)})
        messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with bullet headlines and links. Do not call tools."})
        msg = chat_once(messages, allow_tools=False)

    messages.append({"role":"assistant","content":msg.content or ""})
    return messages, (msg.content or "")

# -------- Flask app --------
app = Flask(__name__, static_url_path="", static_folder="static")
CORS(app)

SESSIONS = {}  # in-memory chat state

@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    thread = data.get("thread") or str(uuid.uuid4())
    user_msg = data["message"]
    history = SESSIONS.get(thread, [])
    history, answer = tool_loop(user_msg, history)
    SESSIONS[thread] = history
    return jsonify({"thread": thread, "answer": answer})

@app.get("/")
def root():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(port=7000, debug=False)