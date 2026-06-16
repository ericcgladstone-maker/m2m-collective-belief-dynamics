"""LLM provider abstraction. Routes to OpenAI (chat completions via urllib) or
Anthropic (Messages API via the SDK) based on config.PROVIDER, so the substrate
swaps by changing only config.AGENT_MODEL. Belief elicitation is verbalized
point-probability (provider-agnostic), so no logprobs are required."""
from __future__ import annotations
import json
import math
import socket
import time
import urllib.request
import urllib.error
from typing import Optional

import config

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_anthropic_client = None


class ProviderError(RuntimeError):
    pass


class DailyLimitError(ProviderError):
    """Daily request cap (RPD) hit — retrying won't help until it resets (OpenAI)."""
    pass


def chat(messages, model: Optional[str] = None, max_tokens: int = 256,
         temperature: float = 0.0, logprobs: bool = False, top_logprobs: int = 0,
         max_retries: int = 6, timeout: float = 60.0) -> dict:
    """Return {'text': str, 'first_token_logprobs': ... | None, 'usage': {...}}."""
    model = model or config.AGENT_MODEL
    prov = config.provider_for(model)
    if prov == "anthropic":
        return _anthropic_chat(messages, model, max_tokens, temperature, max_retries)
    if prov == "gemini":
        return _gemini_chat(messages, model, max_tokens, temperature, max_retries, timeout)
    return _openai_chat(messages, model, max_tokens, temperature, logprobs,
                        top_logprobs, max_retries, timeout)


# ---------------- OpenAI ----------------
def _openai_chat(messages, model, max_tokens, temperature, logprobs, top_logprobs,
                 max_retries, timeout) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    if logprobs:
        body["logprobs"] = True
        if top_logprobs:
            body["top_logprobs"] = top_logprobs
    data = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            _OPENAI_URL, data=data,
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            ch = d["choices"][0]
            ftlp = None
            lp = ch.get("logprobs") or {}
            content_lp = lp.get("content")
            if content_lp:
                ftlp = [(t["token"], t["logprob"]) for t in content_lp[0].get("top_logprobs", [])]
            return {"text": ch["message"].get("content") or "",
                    "first_token_logprobs": ftlp, "usage": d.get("usage", {})}
        except urllib.error.HTTPError as e:
            code = e.code
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", str(e))
            except Exception:
                msg = str(e)
            last_err = f"HTTP {code}: {msg}"
            ml = msg.lower()
            if code == 429 and ("per day" in ml or "rpd" in ml or "quota" in ml or "billing" in ml):
                raise DailyLimitError(last_err)   # daily cap OR out-of-credit: stop cleanly, don't burn retries
            # Transient 4xx (408 timeout, 409 conflict, 425 too-early, 429 rate, 431
            # edge/CDN blip) + all 5xx are retryable. Observed: a transient edge 431
            # storm permanently dropped ~29% of trials because 431 fell through to raise.
            if code in (408, 409, 425, 429, 431) or 500 <= code <= 599:
                time.sleep(min(2 ** attempt, 20)); continue
            raise ProviderError(last_err)
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            last_err = f"network: {e}"; time.sleep(min(2 ** attempt, 20)); continue
    raise ProviderError(f"exhausted retries: {last_err}")


# ---------------- Anthropic ----------------
def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=8)
    return _anthropic_client


def _anthropic_chat(messages, model, max_tokens, temperature, max_retries) -> dict:
    import anthropic
    client = _get_anthropic()
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    if getattr(config, "CLAUDE_THINKING", False):
        # Controlled deliberation manipulation: enable extended thinking with a fixed
        # token budget (thinking tokens are private; only final text is shared/parsed).
        # Thinking requires temperature=1 and max_tokens > budget.
        budget = getattr(config, "CLAUDE_THINKING_BUDGET", 2048)
        kwargs = {"model": model, "max_tokens": max(max_tokens, budget + 512),
                  "messages": msgs, "temperature": 1.0,
                  "thinking": {"type": "enabled", "budget_tokens": budget}}
    else:
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": msgs,
                  "temperature": temperature}
    if system:
        kwargs["system"] = system
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**kwargs)
            text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text")
            return {"text": text, "first_token_logprobs": None,
                    "usage": {"prompt_tokens": resp.usage.input_tokens,
                              "completion_tokens": resp.usage.output_tokens}}
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as e:
            last_err = str(e); time.sleep(min(2 ** attempt, 20)); continue
    raise ProviderError(f"anthropic exhausted retries: {last_err}")


# ---------------- Gemini ----------------
def _gemini_chat(messages, model, max_tokens, temperature, max_retries, timeout) -> dict:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
           f":generateContent?key={config.GEMINI_API_KEY}")
    gen = {"maxOutputTokens": max_tokens, "temperature": temperature}
    if "2.5" in model:
        if config.GEMINI_THINKING:
            gen["maxOutputTokens"] = max(max_tokens, 8192)   # room for thinking + answer (Pro starved at 2048)
        else:
            gen["thinkingConfig"] = {"thinkingBudget": 0}    # follow scaffold, no silent thinking
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    data = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            cand = (d.get("candidates") or [{}])[0]
            parts = cand.get("content", {}).get("parts", []) or []
            text = "".join(p.get("text", "") for p in parts)
            um = d.get("usageMetadata", {})
            return {"text": text, "first_token_logprobs": None,
                    "usage": {"prompt_tokens": um.get("promptTokenCount", 0),
                              "completion_tokens": um.get("candidatesTokenCount", 0)}}
        except urllib.error.HTTPError as e:
            code = e.code
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", str(e))
            except Exception:
                msg = str(e)
            last_err = f"HTTP {code}: {msg}"
            if code == 429 or 500 <= code <= 599:
                time.sleep(min(2 ** attempt, 20)); continue
            raise ProviderError(last_err)
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            last_err = f"network: {e}"; time.sleep(min(2 ** attempt, 20)); continue
    raise ProviderError(f"gemini exhausted retries: {last_err}")


def belief_from_logprobs(first_token_logprobs, labels=("A", "B")) -> dict:
    """(Retained for the OpenAI logprob path; unused by the verbalized readout.)"""
    if not first_token_logprobs:
        return {lab: 1.0 / len(labels) for lab in labels}
    floor = min(lp for _, lp in first_token_logprobs)
    raw = {}
    for lab in labels:
        best = None
        for tok, lp in first_token_logprobs:
            if tok.strip().upper() == lab.upper():
                best = lp if best is None else max(best, lp)
        raw[lab] = best if best is not None else (floor - 5.0)
    exps = {lab: math.exp(v) for lab, v in raw.items()}
    z = sum(exps.values())
    return {lab: exps[lab] / z for lab in labels}
