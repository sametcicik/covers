"""LLM gateway client. Everything goes through LLM_BASE_URL / LLM_API_KEY (.env).

LLM_API_STYLE=openai (default): POST {base}/chat/completions  (OpenAI-compatible gateways, LiteLLM, OpenRouter...)
LLM_API_STYLE=anthropic:        POST {base}/v1/messages
When no gateway is configured, callers fall back to deterministic heuristics.
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import httpx

from .config import settings
from .util import log


def enabled() -> bool:
    return settings.llm_enabled


def _style() -> str:
    return os.environ.get("LLM_API_STYLE", "openai").lower()


def chat(system: str, user: str | list, model: str | None = None, max_tokens: int = 4000,
         temperature: float = 0.4, images: list[Path] | None = None) -> str:
    if not enabled():
        raise RuntimeError("LLM not configured")
    model = model or settings.llm_model
    base = settings.llm_base_url
    imgs = []
    for p in images or []:
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        imgs.append(b64)
    if _style() == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in imgs]
        content.append({"type": "text", "text": user if isinstance(user, str) else json.dumps(user, ensure_ascii=False)})
        url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "system": system,
                "messages": [{"role": "user", "content": content}]}
        headers = {"x-api-key": settings.llm_api_key, "anthropic-version": "2023-06-01",
                   "Authorization": f"Bearer {settings.llm_api_key}"}
    else:
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in imgs]
        text = user if isinstance(user, str) else json.dumps(user, ensure_ascii=False)
        content = content + [{"type": "text", "text": text}] if imgs else text
        url = base + "/chat/completions"
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    last = None
    for attempt in range(3):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout)
            r.raise_for_status()
            j = r.json()
            if "choices" in j:
                return j["choices"][0]["message"]["content"] or ""
            if "content" in j:
                return "".join(c.get("text", "") for c in j["content"] if c.get("type") == "text")
            raise RuntimeError(f"unexpected LLM response: {str(j)[:200]}")
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("LLM call failed (%d): %s", attempt + 1, e)
    raise RuntimeError(f"LLM failed: {last}")


def parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def chat_json(system: str, user, **kw):
    out = chat(system, user, **kw)
    try:
        return parse_json(out)
    except Exception:
        fix = chat("Return ONLY valid JSON. No prose.", f"Fix this into valid JSON:\n{out[:12000]}", max_tokens=kw.get("max_tokens", 4000))
        return parse_json(fix)
