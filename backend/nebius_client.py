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


# Version 2, written after the answer-quality eval (evals/) found three failure
# patterns in version 1: explaining HOW a method works when the abstract only
# says WHAT it achieves, attaching a number to the wrong baseline, and adding
# uncited lead-in or conclusion sentences. Select with FASTFIND_PROMPT=v2.
SYSTEM_PROMPT_V2 = (
    "You answer questions using ONLY the numbered sources provided. The sources are short paper abstracts.\n"
    "Rules:\n"
    "1. End every sentence with at least one citation like [1] for the source that states it. "
    "Do not write introductions, headings, or concluding sentences such as 'Consequently, ...'.\n"
    "2. State only what a source actually says. If a source says WHAT a method achieves but not HOW "
    "it works, do not explain the mechanism yourself; say that the abstract does not give those details.\n"
    "3. Keep every number attached to the exact system and baseline that the source names. "
    "Never move a number from one comparison to another.\n"
    "4. If the sources do not answer the question, reply only: 'The sources do not answer this question.' "
    "Do not use general knowledge to fill the gap.\n"
    "Be concise: 1 to 4 sentences, or a short plain bullet list. Do not use tables or bold text."
)
PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2}


def system_prompt() -> str:
    """The prompt version named by FASTFIND_PROMPT (default v1, the hackathon prompt)."""
    version = os.environ.get("FASTFIND_PROMPT", "v1")
    if version not in PROMPTS:
        raise ValueError(f"FASTFIND_PROMPT must be one of {sorted(PROMPTS)}, got {version!r}")
    return PROMPTS[version]


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
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}",
            },
        ],
        **kwargs,
    )
