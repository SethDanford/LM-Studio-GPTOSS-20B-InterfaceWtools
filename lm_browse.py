import json, re
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# Search backend import (prefer new name `ddgs`)
try:
    from ddgs import DDGS  # pip install ddgs
except Exception:
    from duckduckgo_search import DDGS  # fallback if installed

# ---------- Config ----------
BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY  = "lm-studio"          # any non-empty string
MODEL    = "openai/gpt-oss-20b" # change if needed

SNIPPET_LEN = 120
MAX_TOOL_ROUNDS = 1
MAX_HISTORY_CHARS = 8000

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

TOOLS = [
  {"type":"function","function":{
    "name":"web_search",
    "description":"Search web.",
    "parameters":{
      "type":"object",
      "properties":{
        "q":{"type":"string"},
        "max_results":{"type":"integer","minimum":1,"maximum":5,"default":5}
      },
      "required":["q"],
      "additionalProperties":False
    }
  }},
  {"type":"function","function":{
    "name":"fetch_url",
    "description":"Get text from URL.",
    "parameters":{
      "type":"object",
      "properties":{
        "url":{"type":"string","format":"uri"},
        "max_chars":{"type":"integer","default":6000}
      },
      "required":["url"],
      "additionalProperties":False
    }
  }}
]

SYSTEM = {"role":"system","content":"You can call tools. Always use web_search first for fresh info. Only call fetch_url after you have a specific article URL. Never call fetch_url on a homepage. After using tools, answer in plain text with short bullets and clickable links. Do not emit channel tags."}

# ---------- Helpers ----------
def prune_history(messages, max_chars=MAX_HISTORY_CHARS):
    if not messages: return messages
    system = [m for m in messages if m["role"]=="system"][:1]
    other  = [m for m in messages if m["role"]!="system"]
    total = 0; kept=[]
    for m in reversed(other):
        chunk = m.get("content") or ""
        if isinstance(chunk,(dict,list)): chunk = json.dumps(chunk)
        total += len(chunk); kept.append(m)
        if total >= max_chars: break
    kept.reverse()
    return system + kept

def as_text(content):
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "".join(parts)
    return content or ""

def extract_tool_markups(text):
    """
    Handle LM-style channel tags like:
    <|channel|>commentary to=functions.web_search<|constraint|>json<|message|>{"q":"foo"}
    """
    text = as_text(text)
    calls=[]
    if "to=functions." not in text: return calls
    pat = r"to=functions\.([A-Za-z0-9_]+).*?(?:<\|message\|>|\|message\|>)(\{.*?\})(?=(?:<\||\|start\||$))"
    for m in re.finditer(pat, text, flags=re.S):
        name = m.group(1); arg_str = m.group(2).strip()
        try: args = json.loads(arg_str)
        except Exception:
            try: args = json.loads(arg_str.replace("'", "\""))
            except Exception: args = {}
        calls.append({"name":name,"arguments":args})
    return calls

def do_web_search(q, k):
    hits = list(DDGS().text(q, max_results=k))
    out=[]
    for h in hits[:k]:
        snip = (h.get("body") or "")
        if len(snip) > SNIPPET_LEN: snip = snip[:SNIPPET_LEN] + "..."
        out.append({"title":h.get("title"),"href":h.get("href"),"snippet":snip})
    return out

def do_fetch_url(url, limit):
    r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)[:limit]
    return {"url":url,"text":text,"truncated_to":limit}

# ---------- Chat turn ----------
def run_turn(user_text, messages):
    messages.append({"role":"user","content":user_text})
    tool_rounds = 0
    accum = {"web_results": None, "fetched": []}
    while True:
        messages[:] = prune_history(messages)
        tools_arg = TOOLS if tool_rounds < MAX_TOOL_ROUNDS else None
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools_arg, temperature=0.2, max_tokens=384
        )
        # Guard: LM Studio may return an error object without `choices` if BASE_URL/model is wrong.
        if not getattr(resp, "choices", None) or not resp.choices:
            print("\n[ERROR] chat.completions returned no choices. Verify LM Studio server URL includes /v1, model is loaded, and server is running.")
            print("Raw response:", resp)
            return messages
        msg = resp.choices[0].message

        # 1) Handle LM-style inline tool markup
        markup_calls = extract_tool_markups(getattr(msg, "content", ""))
        if tools_arg and markup_calls:
            messages.append({"role":"assistant","content": as_text(msg.content)})
            for i, mc in enumerate(markup_calls):
                name = mc["name"]; args = mc.get("arguments", {})
                if name == "web_search":
                    q = args.get("q",""); k = min(5, int(args.get("max_results", args.get("topn", 5))))
                    out = do_web_search(q, k)
                    accum["web_results"] = out
                elif name == "fetch_url":
                    out = do_fetch_url(args.get("url"), int(args.get("max_chars", 6000)))
                    accum["fetched"].append(out)
                else:
                    out = {"error":"unknown tool"}
                messages.append({"role":"tool","tool_call_id":f"markup-{i}","name":name,"content":json.dumps(out)})
            tool_rounds += 1
            messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with 5 bullet headlines and clickable links. Do not call tools."})
            continue

        # 2) Handle OpenAI-style tool_calls
        if tools_arg and getattr(msg, "tool_calls", None):
            messages.append({"role":"assistant","tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if name == "web_search":
                    out = do_web_search(args["q"], min(5, args.get("max_results",5)))
                    accum["web_results"] = out
                elif name == "fetch_url":
                    out = do_fetch_url(args["url"], int(args.get("max_chars",6000)))
                    accum["fetched"].append(out)
                else:
                    out = {"error":"unknown tool"}
                messages.append({"role":"tool","tool_call_id": tc.id,"name": name,"content": json.dumps(out)})
            tool_rounds += 1
            messages.append({"role":"system","content":"Using the gathered tool outputs, answer in plain text with 5 bullet headlines and clickable links. Do not call tools."})
            continue

        # 3) Final assistant text
        content = as_text(msg.content).strip()
        if not content or "to=functions." in content:
            # Fallback: format from accumulated tool outputs
            if accum.get("web_results"):
                bullets = []
                for r in accum["web_results"]:
                    t = (r.get("title") or "Untitled").strip()
                    u = r.get("href") or ""
                    bullets.append(f"- {t} — {u}")
                content = "Top results:\n" + "\n".join(bullets)
                print("\nASSISTANT:\n" + content)
                messages.append({"role":"assistant","content":content})
                return messages
            # If no accumulated results, nudge once more for summary
            messages.append({"role":"assistant","content":""})
            messages.append({"role":"system","content":"Answer now in plain text using gathered tool outputs. No tags. If none, say you could not find results."})
            continue
        print("\nASSISTANT:\n" + content)
        messages.append({"role":"assistant","content":content})
        return messages

# ---------- CLI ----------
def main():
    print("LM Studio web-augmented chat. Ctrl+C to exit.")
    messages = [SYSTEM]
    try:
        while True:
            user_text = input("\nYOU:\n").strip()
            if not user_text: continue
            messages = run_turn(user_text, messages)
    except KeyboardInterrupt:
        print("\nBye.")

if __name__ == "__main__":
    main()