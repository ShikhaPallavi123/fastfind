"""
Run this FIRST once your key is set. It lists every model your Token Factory
account can use, so you can confirm the exact Nemotron and embedding ids.

    export NEBIUS_API_KEY='your-key'
    python scripts/check_models.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.nebius_client import get_client

client = get_client()
ids = sorted(m.id for m in client.models.list().data)

print(f"{len(ids)} models available:\n")
for model_id in ids:
    tag = ""
    lowered = model_id.lower()
    if "nemotron" in lowered:
        tag = "   <-- NVIDIA, use for CHAT_MODEL"
    elif "embed" in lowered or "bge" in lowered or "e5" in lowered:
        tag = "   <-- embedding, use for EMBED_MODEL"
    print(f"  {model_id}{tag}")
