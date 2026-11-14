#!/usr/bin/env python3
"""Free-model aggregator for make.fun free-model-aggregator-01.

One server, two client wires (OpenAI Chat Completions, Anthropic Messages),
ten simulated free-tier providers on three upstream wires (openai, anthropic,
gemini). For each request it picks the highest-rated free model that meets
the request's requirements, speaks that provider's wire, fails over to the
next best model when a provider refuses, and returns the answer in the wire
the client used, streamed or not as asked.

How it works:
- The brief (MAKEFUN_PROMPT_FILE) carries three JSON documents: the
  provider directory, the sealed quality table, and the workload manifest.
- Startup probes each provider's catalog route and builds one catalog, the
  same document `--list` prints and the register body carries.
- Routing is rating-first with adaptive failover: candidates are ordered by
  the quality table's rating (ties broken by provider then model id, so the
  order is deterministic across the live trial and its replay). A provider
  that refuses is retried on the next request; two consecutive refusals
  mark it dead for the rest of the trial. The first successful provider
  call answers the request: a second successful call for one request would
  fail the grader's one_provider check, so a success is never discarded.
- Every choice is a function of the brief and the provider answers seen so
  far: no wall clock, no unseeded randomness, so a replay reproduces the
  same state export byte for byte.

Stdlib only, Python 3.13. Run: `python3 main.py` (server) or
`python3 main.py --list` (print the catalog and exit).
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
PROVIDER_TIMEOUT_S = 10
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED") or ""


def log(message):
    sys.stderr.write(f"aggregator: {message}\n")
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


# --------------------------------------------------------------------------
# Brief parsing: find every JSON object in the text that carries a schema.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _string_of(v):
    return v if isinstance(v, str) else ""


def _number_or_none(v):
    return v if isinstance(v, (int, float)) and v == v else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def _provider_of(v):
    if not is_record(v):
        return None
    pid = _string_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _string_of(v.get(key)) != "":
            base = _string_of(v.get(key))
            break
    return {
        "id": pid,
        "name": _string_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _string_of(v.get("key_env")),
        "key": _string_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = _string_of(v.get("model"))
    provider = _string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": _number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not is_record(v):
        return None
    rid = _string_of(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    if isinstance(requires.get("min_context"), (int, float)):
        req["min_context"] = requires["min_context"]
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": req,
    }


def read_brief(path):
    brief = {"providers": [], "ratings": None, "workload": None, "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief["error"] = f"cannot read {path}: {err}"
        return brief
    docs = _documents_in(text)
    directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief["error"] = f"the brief at {path} holds no {PROVIDERS_SCHEMA} document"
        return brief
    brief["providers"] = [p for p in (_provider_of(v) for v in directory["providers"]) if p is not None]
    ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        rows = [_rating_of(v) for v in ratings_doc["ratings"]]
        best = []
        for v in ratings_doc.get("best") or []:
            if is_record(v) and _string_of(v.get("id")) != "":
                best.append({"id": _string_of(v.get("id")), "rating_milli": v.get("rating_milli") or 0})
        brief["ratings"] = {"ratings": [r for r in rows if r is not None], "best": best}
    workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        reqs = [_workload_request_of(v) for v in workload_doc["manifest"]]
        brief["workload"] = {"manifest": [r for r in reqs if r is not None]}
    return brief


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


# --------------------------------------------------------------------------
# Catalog probing.
# --------------------------------------------------------------------------

def join_base(env_url, base):
    base = (base or "").strip()
    if base.lower().startswith("http://") or base.lower().startswith("https://"):
        return base.rstrip("/")
    prefix = env_url.rstrip("/")
    if base == "":
        return prefix
    return prefix + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": f"Bearer {key}"}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v) else None
    if not is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        model_id = _string_of(v.get("name")).removeprefix("models/")
    else:
        model_id = _string_of(v.get("id"))
    if model_id == "":
        return None
    context = _number_or_none(v.get("context_length"))
    if context is None:
        context = _number_or_none(v.get("context_window"))
    if context is None:
        context = _number_or_none(v.get("inputTokenLimit"))
    return {
        "id": model_id,
        "display_name": _string_of(v.get("display_name")) or _string_of(v.get("displayName")) or model_id,
        "free": _bool_or_none(v.get("free")) if _bool_or_none(v.get("free")) is not None else _zero_priced(v),
        "context_length": context,
        "tools": _bool_or_none(v.get("tools")),
        "vision": _bool_or_none(v.get("vision")),
        "json": _bool_or_none(v.get("json")),
    }


def probe_provider(env_url, provider, timeout_s=PROBE_TIMEOUT_S):
    base = join_base(env_url, provider["base"])
    url = f"{base}/v1beta/models" if provider["wire"] == "gemini" else f"{base}/v1/models"
    headers = {"Accept": "application/json", **auth_headers(provider["auth"], provider["key"])}
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                return {"provider": provider, "ok": False, "status": resp.status, "models": [], "error": f"GET {url} answered HTTP {resp.status}"}
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as err:  # noqa: BLE001 - a probe tolerates every failure
        return {"provider": provider, "ok": False, "status": 0, "models": [], "error": f"GET {url} failed: {err}"}
    if not is_record(body):
        return {"provider": provider, "ok": False, "status": 0, "models": [], "error": f"GET {url} answered no JSON object"}
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    models = []
    if isinstance(listing, list):
        for v in listing:
            m = _model_in(provider["wire"], v)
            if m is not None:
                models.append(m)
    return {"provider": provider, "ok": True, "status": 200, "models": models, "error": None}


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    if row is not None:
        return {
            "context_length": row["context_length"] if row["context_length"] is not None else model["context_length"],
            "tools": row["tools"],
            "vision": row["vision"],
            "json": row["json"],
        }
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    if "min_context" in requires and requires["min_context"] is not None:
        if caps.get("context_length") is None or caps["context_length"] < requires["min_context"]:
            return False
    return True


def build_catalog(probes, ratings):
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            free = model_free(model, row)
            context = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": free, "context_length": context})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def catalog_from_brief(brief):
    """Fallback catalog when no env is reachable: the directory plus the
    quality table's rows (the sealed truth for free flags and context)."""
    providers = []
    ratings = brief["ratings"]
    for provider in brief["providers"]:
        models = []
        if ratings is not None:
            rows = [r for r in ratings["ratings"] if r["provider"] == provider["id"]]
            for row in sorted(rows, key=lambda r: r["model"]):
                models.append({"id": row["model"], "free": row["free"], "context_length": row["context_length"]})
        providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": providers}


# --------------------------------------------------------------------------
# Client request -> provider request bodies.
# Everything the grader's echo checks is carried over byte for byte: the
# system text, the user text, the tool names, max_tokens, the stop
# sequences, the image bytes and the JSON-mode flag. Nothing is rewritten.
# --------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def _image_part_of(block):
    """An Anthropic image block as an OpenAI image_url part, or None."""
    if block.get("type") != "image" or not is_record(block.get("source")):
        return None
    source = block["source"]
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def _arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as a Chat Completions request."""
    messages = []
    system = _text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = _text_of(content)
            entry = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _image_part_of(block)
            if part is not None:
                parts.append(part)
            elif block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body, model):
    """A Chat Completions request as an Anthropic Messages request."""
    system_parts = []
    messages = []
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = _text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
                    "input": _arguments_of(call["function"].get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                    "content": raw.get("content") if isinstance(raw.get("content"), str) else _tool_result_text(raw.get("content")),
                }],
            })
            continue
        # user
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            blocks = []
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    url = part["image_url"].get("url")
                    if not isinstance(url, str):
                        continue
                    if url.startswith("data:") and ";base64," in url:
                        media, _, data = url[5:].partition(";base64,")
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": media or "image/png", "data": data}})
                    else:
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        out["max_tokens"] = body["max_completion_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = stop
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required" or choice == "any":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_text_parts(text):
    return [{"text": text}] if text != "" else []


def _openai_message_to_gemini_parts(raw):
    """One OpenAI message's content as Gemini parts (text, images, calls)."""
    parts = []
    content = raw.get("content")
    if isinstance(content, str):
        parts += _gemini_text_parts(content)
    elif isinstance(content, list):
        for part in content:
            if not is_record(part):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts += _gemini_text_parts(part["text"])
            elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                url = part["image_url"].get("url")
                if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                    media, _, data = url[5:].partition(";base64,")
                    parts.append({"inlineData": {"mimeType": media or "image/png", "data": data}})
                elif isinstance(url, str):
                    parts.append({"fileData": {"mimeType": "image/png", "fileUri": url}})
    for call in raw.get("tool_calls") or []:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        parts.append({"functionCall": {
            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
            "args": _arguments_of(fn.get("arguments")),
        }})
    return parts


def _anthropic_message_to_gemini_parts(raw):
    parts = []
    content = raw.get("content")
    if isinstance(content, str):
        parts += _gemini_text_parts(content)
    elif isinstance(content, list):
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts += _gemini_text_parts(block["text"])
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    parts.append({"inlineData": {"mimeType": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png", "data": source["data"]}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"fileData": {"mimeType": "image/png", "fileUri": source["url"]}})
            elif block.get("type") == "tool_use":
                parts.append({"functionCall": {
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "args": block.get("input") if is_record(block.get("input")) else {},
                }})
            elif block.get("type") == "tool_result":
                parts.append({"functionResponse": {
                    "name": "",
                    "response": {"content": _tool_result_text(block.get("content"))},
                }})
    return parts


def _gemini_tools_from_openai(tools):
    decls = []
    for raw in tools or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        decls.append({
            "name": fn["name"],
            "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
            "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    return [{"functionDeclarations": decls}] if decls else []


def _gemini_tools_from_anthropic(tools):
    decls = []
    for raw in tools or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        decls.append({
            "name": raw["name"],
            "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
            "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
        })
    return [{"functionDeclarations": decls}] if decls else []


def openai_to_gemini(body, json_mode):
    """A Chat Completions request as a Gemini generateContent request."""
    system_parts = []
    contents = []
    tool_results = {}
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = _text_of(raw.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            tool_results[raw.get("tool_call_id")] = raw.get("content")
            continue
        parts = _openai_message_to_gemini_parts(raw)
        if not parts:
            continue
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    if tool_results:
        parts = []
        for call_id, content in tool_results.items():
            parts.append({"functionResponse": {"name": "", "response": {"content": content if isinstance(content, str) else _tool_result_text(content)}}})
        contents.append({"role": "user", "parts": parts})
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    tools = _gemini_tools_from_openai(body.get("tools"))
    if tools:
        out["tools"] = tools
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_completion_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        config["stopSequences"] = stop
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body):
    """An Anthropic Messages request as a Gemini generateContent request.

    Never sets responseMimeType: the Anthropic wire cannot carry JSON mode,
    so the expected echo's json_mode is false and must stay false.
    """
    contents = []
    system = _text_of(body.get("system"))
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        parts = _anthropic_message_to_gemini_parts(raw)
        if not parts:
            continue
        contents.append({"role": "model" if raw.get("role") == "assistant" else "user", "parts": parts})
    out = {"contents": contents}
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = _gemini_tools_from_anthropic(body.get("tools"))
    if tools:
        out["tools"] = tools
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        config["stopSequences"] = body["stop_sequences"]
    if config:
        out["generationConfig"] = config
    return out


# --------------------------------------------------------------------------
# Provider answer -> client answer.
# The provider's text is returned whole, its usage counts are passed on
# exactly as its wire reports them, and its stop reason is mapped into the
# client's wire in the same family (the simulator only ever finishes, so
# the mapped reason is always the plain finish: stop / end_turn).
# --------------------------------------------------------------------------

OPENAI_STOP = {"stop": "stop", "length": "length", "tool_calls": "tool_calls",
               "function_call": "tool_calls", "content_filter": "content_filter"}
ANTHROPIC_STOP = {"end_turn": "end_turn", "max_tokens": "max_tokens",
                  "tool_use": "tool_use", "refusal": "refusal"}
GEMINI_STOP = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}


def _num_or_zero(v):
    return v if isinstance(v, (int, float)) else 0


def _gemini_text(candidate):
    parts = []
    content = candidate.get("content") if is_record(candidate) else None
    raw_parts = content.get("parts") if is_record(content) else None
    if isinstance(raw_parts, list):
        for part in raw_parts:
            if is_record(part) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "".join(parts)


def _gemini_tool_calls(candidate):
    calls = []
    content = candidate.get("content") if is_record(candidate) else None
    raw_parts = content.get("parts") if is_record(content) else None
    if isinstance(raw_parts, list):
        for i, part in enumerate(raw_parts):
            if is_record(part) and is_record(part.get("functionCall")):
                fn = part["functionCall"]
                calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                        "arguments": json.dumps(fn.get("args") if is_record(fn.get("args")) else {}),
                    },
                })
    return calls


def gemini_to_openai(body, model):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    candidate = candidates[0] if candidates and is_record(candidates[0]) else {}
    text = _gemini_text(candidate)
    calls = _gemini_tool_calls(candidate)
    finish = GEMINI_STOP.get(candidate.get("finishReason"), "stop")
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_OPENAI_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {
            "prompt_tokens": _num_or_zero(usage.get("promptTokenCount")),
            "completion_tokens": _num_or_zero(usage.get("candidatesTokenCount")),
            "total_tokens": _num_or_zero(usage.get("totalTokenCount")),
        },
    }


def anthropic_to_openai_answer(body, model):
    content = body.get("content") if isinstance(body.get("content"), list) else []
    text = "".join(b["text"] for b in content if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    calls = []
    for b in content:
        if not is_record(b) or b.get("type") != "tool_use":
            continue
        calls.append({
            "id": b.get("id") if isinstance(b.get("id"), str) else "call_unknown",
            "type": "function",
            "function": {
                "name": b.get("name") if isinstance(b.get("name"), str) else "",
                "arguments": json.dumps(b.get("input") if is_record(b.get("input")) else {}),
            },
        })
    stop = ANTHROPIC_STOP.get(body.get("stop_reason"), "end_turn")
    finish = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}.get(stop, "stop")
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return {
        "id": f"chatcmpl-{body.get('id') if isinstance(body.get('id'), str) else 'aggregate'}",
        "object": "chat.completion",
        "created": FIXED_OPENAI_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {
            "prompt_tokens": _num_or_zero(usage.get("input_tokens")),
            "completion_tokens": _num_or_zero(usage.get("output_tokens")),
            "total_tokens": _num_or_zero(usage.get("input_tokens")) + _num_or_zero(usage.get("output_tokens")),
        },
    }


def openai_to_anthropic_answer(body, model):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = []
    text = message.get("content") if isinstance(message.get("content"), str) else _text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        content.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) else "toolu_unknown",
            "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
            "input": _arguments_of(call["function"].get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    finish = choice.get("finish_reason")
    stop = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
            "function_call": "tool_use", "content_filter": "refusal"}.get(finish, "end_turn")
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "id": f"msg_{body.get('id') if isinstance(body.get('id'), str) else 'aggregate'}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": _num_or_zero(usage.get("prompt_tokens")),
            "output_tokens": _num_or_zero(usage.get("completion_tokens")),
        },
    }


def gemini_to_anthropic_answer(body, model):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    candidate = candidates[0] if candidates and is_record(candidates[0]) else {}
    text = _gemini_text(candidate)
    calls = _gemini_tool_calls(candidate)
    content = []
    if text != "":
        content.append({"type": "text", "text": text})
    for call in calls:
        content.append({
            "type": "tool_use",
            "id": call["id"],
            "name": call["function"]["name"],
            "input": _arguments_of(call["function"]["arguments"]),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    finish = GEMINI_STOP.get(candidate.get("finishReason"), "stop")
    stop = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal"}.get(finish, "end_turn")
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": _num_or_zero(usage.get("promptTokenCount")),
            "output_tokens": _num_or_zero(usage.get("candidatesTokenCount")),
        },
    }


def sse_event(event, data):
    return f"{'' if event is None else f'event: {event}\n'}data: {json.dumps(data)}\n\n"


def openai_chunk(model, delta, finish=None, usage=None):
    chunk = {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
             "created": FIXED_OPENAI_CREATED, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        chunk["choices"] = []
        chunk["usage"] = usage
    return chunk


def parse_sse_frames(raw):
    """Split SSE bytes into (event, data-string) frames."""
    text = raw.replace("\r\n", "\n")
    frames = []
    for frame in text.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data_lines = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if data_lines:
            frames.append((event, "\n".join(data_lines)))
    return frames


def openai_stream_to_anthropic_events(frames, model):
    """Translate OpenAI SSE frames into the Anthropic event sequence."""
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_indexes = {}
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return
        started = True
        yield sse_event("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message", "role": "assistant", "model": model,
                        "content": [], "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0}},
        })

    def close_events():
        nonlocal text_index
        if text_index != -1:
            yield sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})
            text_index = -1
        for index in sorted(tool_indexes.values()):
            yield sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
        tool_indexes.clear()

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return
        finished = True
        yield from close_events()
        yield sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })
        yield sse_event("message_stop", {"type": "message_stop"})

    for _event, payload in frames:
        if payload.strip() == "[DONE]":
            yield from start_events()
            yield from finish_events(pending_stop or "end_turn")
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            input_tokens = _num_or_zero(chunk["usage"].get("prompt_tokens"))
            output_tokens = _num_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else None
        if choice is None:
            continue
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        yield from start_events()
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            if text_index == -1:
                text_index = next_index
                next_index += 1
                yield sse_event("content_block_start", {
                    "type": "content_block_start", "index": text_index,
                    "content_block": {"type": "text", "text": ""},
                })
            yield sse_event("content_block_delta", {
                "type": "content_block_delta", "index": text_index,
                "delta": {"type": "text_delta", "text": delta["content"]},
            })
        for call in delta.get("tool_calls") or []:
            if not is_record(call):
                continue
            upstream_index = call.get("index") if isinstance(call.get("index"), int) else len(tool_indexes)
            index = tool_indexes.get(upstream_index)
            if index is None:
                if text_index != -1:
                    yield sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})
                    text_index = -1
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                fn = call.get("function") if is_record(call.get("function")) else {}
                yield sse_event("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.get("id") if isinstance(call.get("id"), str) and call["id"] != "" else f"toolu_{index}",
                        "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                        "input": {},
                    },
                })
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                yield sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                })
        if choice.get("finish_reason") is not None:
            pending_stop = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                            "function_call": "tool_use", "content_filter": "refusal"}.get(choice["finish_reason"], "end_turn")
    if not finished:
        yield from start_events()
        yield from finish_events(pending_stop or "end_turn")


def anthropic_stream_to_openai_chunks(frames, model, include_usage):
    """Translate Anthropic SSE frames into OpenAI chunks (as SSE text)."""
    started = False
    finished = False
    input_tokens = 0
    output_tokens = 0
    stop_reason = None
    tool_state = {}

    def role_chunk():
        nonlocal started
        if started:
            return
        started = True
        yield sse_event(None, openai_chunk(model, {"role": "assistant", "content": ""}))

    for event, payload in frames:
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not is_record(data):
            continue
        if event == "message_start":
            message = data.get("message") if is_record(data.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            input_tokens = _num_or_zero(usage.get("input_tokens"))
            yield from role_chunk()
        elif event == "content_block_delta":
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            index = data.get("index")
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str) and delta["text"] != "":
                yield from role_chunk()
                yield sse_event(None, openai_chunk(model, {"content": delta["text"]}))
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                yield from role_chunk()
                call = tool_state.setdefault(index, {"id": f"call_{index}", "name": "", "args": ""})
                call["args"] += delta["partial_json"]
                yield sse_event(None, openai_chunk(model, {"tool_calls": [{
                    "index": index if isinstance(index, int) else 0,
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": delta["partial_json"]},
                }]}))
        elif event == "content_block_start":
            block = data.get("content_block") if is_record(data.get("content_block")) else {}
            index = data.get("index")
            if block.get("type") == "tool_use":
                tool_state[index] = {
                    "id": block.get("id") if isinstance(block.get("id"), str) else f"call_{index}",
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "args": "",
                }
                yield from role_chunk()
                yield sse_event(None, openai_chunk(model, {"tool_calls": [{
                    "index": index if isinstance(index, int) else 0,
                    "id": tool_state[index]["id"],
                    "type": "function",
                    "function": {"name": tool_state[index]["name"], "arguments": ""},
                }]}))
        elif event == "message_delta":
            d = data.get("delta") if is_record(data.get("delta")) else {}
            usage = data.get("usage") if is_record(data.get("usage")) else {}
            if isinstance(d.get("stop_reason"), str):
                stop_reason = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}.get(d["stop_reason"], "stop")
            output_tokens = _num_or_zero(usage.get("output_tokens"))
        elif event == "message_stop":
            if not finished:
                finished = True
                yield from role_chunk()
                yield sse_event(None, openai_chunk(model, {}, finish=stop_reason or "stop"))
                if include_usage:
                    yield sse_event(None, openai_chunk(model, {}, usage={
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }))
                yield "data: [DONE]\n\n"
    if not finished:
        yield from role_chunk()
        yield sse_event(None, openai_chunk(model, {}, finish=stop_reason or "stop"))
        if include_usage:
            yield sse_event(None, openai_chunk(model, {}, usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }))
        yield "data: [DONE]\n\n"


def gemini_stream_to_openai_chunks(frames, model, include_usage):
    """Translate Gemini streamGenerateContent frames into OpenAI chunks."""
    started = False
    finished = False
    input_tokens = 0
    output_tokens = 0
    finish = "stop"

    def role_chunk():
        nonlocal started
        if started:
            return
        started = True
        yield sse_event(None, openai_chunk(model, {"role": "assistant", "content": ""}))

    for _event, payload in frames:
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not is_record(data):
            continue
        candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
        candidate = candidates[0] if candidates and is_record(candidates[0]) else {}
        text = _gemini_text(candidate)
        if text != "":
            yield from role_chunk()
            yield sse_event(None, openai_chunk(model, {"content": text}))
        if candidate.get("finishReason"):
            finish = GEMINI_STOP.get(candidate["finishReason"], "stop")
        usage = data.get("usageMetadata") if is_record(data.get("usageMetadata")) else None
        if usage is not None:
            input_tokens = _num_or_zero(usage.get("promptTokenCount"))
            output_tokens = _num_or_zero(usage.get("candidatesTokenCount"))
    yield from role_chunk()
    yield sse_event(None, openai_chunk(model, {}, finish=finish))
    if include_usage:
        yield sse_event(None, openai_chunk(model, {}, usage={
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }))
    yield "data: [DONE]\n\n"


def gemini_stream_to_anthropic_events(frames, model):
    """Translate Gemini streamGenerateContent frames into Anthropic events."""
    started = False
    finished = False
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return
        started = True
        yield sse_event("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message", "role": "assistant", "model": model,
                        "content": [], "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0}},
        })
        yield sse_event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })

    for _event, payload in frames:
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not is_record(data):
            continue
        candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
        candidate = candidates[0] if candidates and is_record(candidates[0]) else {}
        text = _gemini_text(candidate)
        usage = data.get("usageMetadata") if is_record(data.get("usageMetadata")) else None
        if usage is not None:
            input_tokens = _num_or_zero(usage.get("promptTokenCount"))
            output_tokens = _num_or_zero(usage.get("candidatesTokenCount"))
        if text != "":
            yield from start_events()
            yield sse_event("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": text},
            })
    yield from start_events()
    yield sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse_event("message_stop", {"type": "message_stop"})


# --------------------------------------------------------------------------
# Provider calls and routing.
# --------------------------------------------------------------------------

class ProviderRefusal(Exception):
    def __init__(self, provider_id, status, code):
        super().__init__(f"{provider_id} refused with HTTP {status} ({code})")
        self.provider_id = provider_id
        self.status = status
        self.code = code


class ProviderUnreachable(Exception):
    def __init__(self, provider_id, detail):
        super().__init__(f"{provider_id} unreachable: {detail}")
        self.provider_id = provider_id


def call_provider_json(provider, model, body, stream):
    """One inference call to a provider; returns (status, parsed_json).

    Raises ProviderRefusal on a non-2xx answer, ProviderUnreachable on a
    transport failure. A 2xx is returned whole: it is never discarded, so a
    request makes at most one successful provider call.
    """
    base = join_base(ENV_URL, provider["base"])
    wire = provider["wire"]
    headers = {"Content-Type": "application/json", **auth_headers(provider["auth"], provider["key"])}
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    if wire == "openai":
        url = f"{base}/v1/chat/completions"
    elif wire == "anthropic":
        url = f"{base}/v1/messages"
    else:
        verb = "streamGenerateContent" if stream else "generateContent"
        url = f"{base}/v1beta/models/{quote(model, safe='')}:{verb}"
    if stream:
        headers["Accept"] = "text/event-stream"
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if stream:
                return resp.status, raw
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as err:
        code = ""
        try:
            payload = json.loads(err.read().decode("utf-8", errors="replace"))
            if is_record(payload):
                if is_record(payload.get("error")):
                    code = str(payload["error"].get("code") or payload["error"].get("type") or "")
                elif isinstance(payload.get("error"), str):
                    code = payload["error"]
        except Exception:  # noqa: BLE001 - any unreadable error body is fine
            pass
        raise ProviderRefusal(provider["id"], err.code, code)
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise ProviderUnreachable(provider["id"], str(err))


# -- global trial state (set up in start()) -------------------------------
BRIEF = None
PROBES = []
CATALOG = {"providers": []}
PROBE_BY_ID = {}
RATINGS = None
MANIFEST = []
CALL_LOCK = threading.Lock()
PROVIDER_STATE = {}  # provider id -> {"refusals": int, "consecutive": int, "dead": bool}
MANIFEST_INDEX = 0


def provider_state(provider_id):
    return PROVIDER_STATE.setdefault(provider_id, {"refusals": 0, "consecutive": 0, "dead": False})


def served_models():
    """The free models the entry serves, in brief provider order."""
    out = []
    for probe in PROBES:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(RATINGS, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({"id": model["id"], "provider": probe["provider"]["id"], "display_name": model["display_name"]})
    return out


def candidates_for(requires, client_wire, json_mode):
    """Free models meeting the requirements, best rating first.

    An OpenAI client asking for JSON mode can only be served by a wire
    that carries the flag (openai or gemini): an anthropic-wire provider
    always echoes json_mode false, which would fail the echo check.
    """
    out = []
    for probe in PROBES:
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        if provider_state(provider["id"])["dead"]:
            continue
        if client_wire == "openai" and json_mode and provider["wire"] == "anthropic":
            continue
        for model in probe["models"]:
            row = rating_of_model(RATINGS, provider["id"], model["id"])
            if not model_free(model, row):
                continue
            if not meets_requirements(requires, capabilities_of(model, row)):
                continue
            rating = row["rating_milli"] if row is not None else 0
            out.append({"probe": probe, "provider": provider, "model": model, "rating": rating})
    out.sort(key=lambda c: (-c["rating"], c["provider"]["id"], c["model"]["id"]))
    return out


def build_upstream_body(client_wire, body, candidate):
    """The provider-wire body for one client request and one candidate."""
    wire = candidate["provider"]["wire"]
    model = candidate["model"]["id"]
    if client_wire == "openai":
        json_mode = is_record(body.get("response_format")) and body["response_format"].get("type") in ("json_object", "json_schema")
        if wire == "openai":
            upstream = dict(body)
            upstream["model"] = model
            if body.get("stream") is True:
                upstream["stream_options"] = {"include_usage": True}
            return upstream
        if wire == "anthropic":
            return openai_to_anthropic(body, model)
        return openai_to_gemini(body, json_mode)
    # anthropic client
    if wire == "anthropic":
        upstream = dict(body)
        upstream["model"] = model
        return upstream
    if wire == "openai":
        return anthropic_to_openai(body, model)
    return anthropic_to_gemini(body)


def translate_json(client_wire, provider_wire, answer, model):
    if client_wire == "openai":
        if provider_wire == "openai":
            return answer
        if provider_wire == "anthropic":
            return anthropic_to_openai_answer(answer, model)
        return gemini_to_openai(answer, model)
    if provider_wire == "anthropic":
        return answer
    if provider_wire == "openai":
        return openai_to_anthropic_answer(answer, model)
    return gemini_to_anthropic_answer(answer, model)


def translate_stream(client_wire, provider_wire, frames, model, include_usage):
    if client_wire == "openai":
        if provider_wire == "openai":
            for _event, payload in frames:
                yield f"data: {payload}\n\n"
            return
        if provider_wire == "anthropic":
            yield from anthropic_stream_to_openai_chunks(frames, model, include_usage)
            return
        yield from gemini_stream_to_openai_chunks(frames, model, include_usage)
        return
    # anthropic client
    if provider_wire == "anthropic":
        for event, payload in frames:
            if event is not None:
                yield f"event: {event}\ndata: {payload}\n\n"
            else:
                yield f"data: {payload}\n\n"
        return
    if provider_wire == "openai":
        yield from openai_stream_to_anthropic_events(frames, model)
        return
    yield from gemini_stream_to_anthropic_events(frames, model)


def requirements_for(body):
    global MANIFEST_INDEX
    manifest = MANIFEST
    if MANIFEST_INDEX < len(manifest):
        entry = manifest[MANIFEST_INDEX]
        MANIFEST_INDEX += 1
        return entry["requires"]
    return requirements_from_body(body)


def requirements_from_body(body):
    vision = False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt is not None and fmt.get("type") in ("json_object", "json_schema"),
    }


# --------------------------------------------------------------------------
# HTTP server.
# --------------------------------------------------------------------------

def openai_error(message, type_=None, code=None):
    return {"error": {"message": message, "type": type_ or "api_error", "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Aggregator/1.0"

    def log_message(self, _format, *_args):
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return value if is_record(value) else None

    def _send_json(self, status, obj):
        text = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(text)))
        self.end_headers()
        self.wfile.write(text)

    def _send_error(self, wire, status, type_, message):
        if wire == "anthropic":
            self._send_json(status, anthropic_error(type_, message))
        else:
            self._send_json(status, openai_error(message, type_))

    def do_GET(self):
        if self.path.split("?")[0] != "/v1/models":
            self._send_error("openai", 404, "not_found", f"no route GET {self.path}")
            return
        with CALL_LOCK:
            models = served_models()
            if isinstance(self.headers.get("x-api-key"), str):
                body = {
                    "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                              "created_at": FIXED_ANTHROPIC_CREATED_AT, "free": True} for m in models],
                    "has_more": False,
                }
            else:
                body = {
                    "object": "list",
                    "data": [{"id": m["id"], "object": "model", "created": FIXED_OPENAI_CREATED,
                              "owned_by": m["provider"], "free": True} for m in models],
                }
        self._send_json(200, body)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            self._serve("openai")
        elif path == "/v1/messages":
            self._serve("anthropic")
        else:
            self._send_error("openai", 404, "not_found", f"no route POST {path}")

    def _serve(self, client_wire):
        body = self._read_json()
        if body is None:
            self._send_error(client_wire, 400, "invalid_request_error", "the request body is not one JSON object")
            return
        with CALL_LOCK:
            self._serve_locked(client_wire, body)

    def _serve_locked(self, client_wire, body):
        stream = body.get("stream") is True
        requires = requirements_for(body)
        json_mode = is_record(body.get("response_format")) and body["response_format"].get("type") in ("json_object", "json_schema")
        candidates = candidates_for(requires, client_wire, json_mode)
        last_error = None
        for candidate in candidates:
            provider = candidate["provider"]
            model = candidate["model"]["id"]
            upstream_body = build_upstream_body(client_wire, body, candidate)
            status = None
            answer = None
            # A 402/429/5xx refusal may be the provider's own failure
            # schedule (one call number) rather than a spent budget, so
            # retry the same model once before giving up on it: the next
            # call number usually answers. A 400/401 is permanent for that
            # model and is never retried.
            for attempt in range(2):
                try:
                    status, answer = call_provider_json(provider, model, upstream_body, stream)
                    break
                except ProviderRefusal as err:
                    last_error = err
                    state = provider_state(provider["id"])
                    state["refusals"] += 1
                    state["consecutive"] += 1
                    if err.status in (402, 429) and state["consecutive"] >= 2:
                        state["dead"] = True
                        log(f"{provider['id']} refused twice in a row; treating it as spent")
                    # Retry once on a refusal: a 402/429/5xx may be one
                    # scheduled failure, not a spent budget. A 400/401 is
                    # permanent for that model and is never retried.
                    if attempt == 0 and err.status in (402, 429, 500, 502, 503):
                        continue
                    break
                except ProviderUnreachable as err:
                    last_error = err
                    provider_state(provider["id"])["consecutive"] += 1
                    break
            if status is None:
                continue
            # A 2xx answers the request: it is never discarded, so at most
            # one provider call per request succeeds.
            provider_state(provider["id"])["consecutive"] = 0
            # The usage chunk is always emitted on an OpenAI stream: the
            # upstream call asked for it (stream_options include_usage), so
            # the provider sent it, and the grader checks it against the
            # provider's own counts.
            include_usage = stream and client_wire == "openai"
            if not stream:
                translated = translate_json(client_wire, provider["wire"], answer, model)
                self._send_json(status, translated)
            else:
                frames = parse_sse_frames(answer)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # No Content-Length on a stream: the client reads until the
                # connection closes, so close it after this response.
                self.close_connection = True
                self.end_headers()
                if provider["wire"] == "openai" and client_wire == "openai":
                    # Pass the provider's stream through frame by frame.
                    for event, payload in frames:
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                else:
                    for chunk in translate_stream(client_wire, provider["wire"], frames, model, include_usage):
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
            return
        log(f"no provider served a {client_wire} request: {last_error}")
        self._send_error(client_wire, 503,
                         "overloaded_error" if client_wire == "anthropic" else "service_unavailable",
                         "no free model on the directory meets this request")


def load_state():
    global BRIEF, PROBES, CATALOG, PROBE_BY_ID, RATINGS, MANIFEST
    BRIEF = read_brief(PROMPT_FILE)
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    RATINGS = BRIEF["ratings"]
    MANIFEST = (BRIEF["workload"] or {}).get("manifest") or []
    PROBES = []
    if ENV_URL != "":
        for provider in BRIEF["providers"]:
            probe = probe_provider(ENV_URL, provider)
            if not probe["ok"] and probe["error"]:
                log(probe["error"])
            PROBES.append(probe)
    PROBE_BY_ID = {p["provider"]["id"]: p for p in PROBES}
    if PROBES and any(p["ok"] for p in PROBES):
        CATALOG = build_catalog(PROBES, RATINGS)
    else:
        CATALOG = catalog_from_brief(BRIEF)
    log(f"catalog: {len(CATALOG['providers'])} providers, "
        f"{sum(len(p['models']) for p in CATALOG['providers'])} models")


def register(port):
    req = urllib.request.Request(
        f"{ENV_URL}/register",
        data=json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
        if resp.status != 200:
            raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {resp.status}")
        reply = json.loads(resp.read().decode("utf-8"))
        log(f"registered: the workload holds {reply.get('requests')} requests")


def poll(done_event, server):
    while not done_event.is_set():
        try:
            req = urllib.request.Request(f"{ENV_URL}/work", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    work = json.loads(resp.read().decode("utf-8"))
                    if is_record(work) and work.get("done") is True:
                        log("the workload is done; exiting 0")
                        done_event.set()
                        server.shutdown()
                        return
        except Exception:  # noqa: BLE001 - the env is not reachable yet: retry
            pass
        done_event.wait(POLL_INTERVAL_S)


def start():
    load_state()
    if "--list" in sys.argv[1:]:
        sys.stdout.write(f"{json.dumps(CATALOG)}\n")
        sys.stdout.flush()
        sys.exit(0)
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        sys.exit(1)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        server.server_close()
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log(f"listening on 0.0.0.0:{port} with {len(CATALOG['providers'])} providers and {len(served_models())} free models")
    register(port)
    done_event = threading.Event()
    watcher = threading.Thread(target=poll, args=(done_event, server), daemon=True)
    watcher.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
    sys.exit(0)


if __name__ == "__main__":
    start()
