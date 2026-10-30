"""
Nebius Token Factory client.

Token Factory exposes an OpenAI-compatible API, so we use the official
`openai` Python package and just point it at Nebius.

Set these environment variables (never commit your key):
    NEBIUS_API_KEY      your Token Factory API key (required)
    CHAT_MODEL          NVIDIA Nemotron model id (default below)
    EMBED_MODEL         embedding model id (default below)

Model ids come from third-party catalog listings. Confirm them against your
own account with:  python scripts/check_models.py
"""

import os
import re

from openai import OpenAI

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Nemotron 3 Nano is the cheapest NVIDIA option, good for development.
DEFAULT_CHAT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"


def get_client() -> OpenAI:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Run: export NEBIUS_API_KEY='your-key'"
        )
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=45.0)


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into embedding vectors (one list of floats each)."""
    client = get_client()
    model = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered sources provided. "
    "Be concise: 2 to 5 sentences, or a short plain bullet list. "
    "Do not use tables, headings, or bold text. "
    "Cite sources like [1] or [2] right after each claim. "
    "If the sources do not contain the answer, say so plainly instead of guessing. "
    "If the sources only cover a related but different topic, say what they do cover and "
    "that they do not answer the question; do not blend the two topics together."
)


def clean_answer(text: str) -> str:
    """Normalize model output: drop any reasoning block, make citations look like [1]."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    # Some models emit 【1】, ［1］ or (1) style markers; the app expects [1].
    text = re.sub(r"[【\[［]\s*(\d+(?:\s*[,，]\s*\d+)*)\s*[】\]］]", lambda m: "[" + re.sub(r"\s*[,，]\s*", "][", m.group(1)) + "]", text)
    return text.strip()


def answer(question: str, sources: list[str]) -> str:
    """Ask Nemotron to answer using only the retrieved sources."""
    client = get_client()
    model = os.environ.get("CHAT_MODEL", DEFAULT_CHAT_MODEL)
    context = "\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
    kwargs = {}
    if os.environ.get("NEMOTRON_NO_THINK") == "1" and not _no_think_rejected:
        # Ask the server to skip the slow "thinking" phase. If it is not supported, we fall back below.
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        response = _chat(client, model, question, context, kwargs)
    except Exception as e:
        if kwargs and getattr(e, "status_code", None) in (400, 422):
            globals()["_no_think_rejected"] = True
            print("[FastFind] server rejected the no-thinking option; continuing without it", flush=True)
            response = _chat(client, model, question, context, {})
        else:
            raise
    return clean_answer(response.choices[0].message.content)


_no_think_rejected = False


def _chat(client, model, question, context, kwargs):
    return client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}",
            },
        ],
        **kwargs,
    )
