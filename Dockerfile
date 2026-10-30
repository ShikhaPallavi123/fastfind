# Runs on Hugging Face Spaces (Docker SDK, port 7860) or any Docker host.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PORT=7860
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
# The index (data/index.npz, data/chunks.json) is built beforehand with scripts/ingest.py and shipped with the image.
# NEBIUS_API_KEY must be provided as a secret environment variable, never baked into the image.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
