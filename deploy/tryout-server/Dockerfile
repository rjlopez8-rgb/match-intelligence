# The Club mode try-out server: the match engine only (no AI models, no keys). Plays a fixture with a candidate in a
# position, streaming one JSON line per finished match. Runs on Azure Container Apps, scale to zero, 1 replica.
# Build from the repo root:  docker build -f deploy/tryout-server/Dockerfile -t tryouts .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PORT=8765 \
    MATCH_TRYOUT_DAILY_MATCHES=150 \
    MATCH_ALLOWED_ORIGIN=*

RUN useradd --create-home --uid 10001 engine
WORKDIR /app
COPY engine/ /app/engine/
RUN pip install --no-cache-dir "numpy==2.5.3" /app/engine && rm -rf /app/engine
# the pre-simulated try-outs (the starters' runs are shared by every new try-out in a position)
COPY --chown=engine:engine data/demo/riv-kin-1119/tryouts/ /app/tryouts/

USER engine
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"
CMD ["sh", "-c", "exec matchengine serve --host 0.0.0.0 --port \"$PORT\" --cache /app/tryouts --origin \"$MATCH_ALLOWED_ORIGIN\""]
