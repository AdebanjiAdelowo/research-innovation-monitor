# API image. By default it installs no neural dependencies and serves the built-in
# synthetic corpus with a model-free encoder, so it starts without downloads or secrets.
# Build with --build-arg EXTRAS="app,ml,data" to include PyTorch and the Hub client.
FROM python:3.12-slim

ARG EXTRAS="app"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/rim
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[${EXTRAS}]"

RUN useradd --create-home rim && mkdir -p /srv/rim/data && chown -R rim /srv/rim
USER rim

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "rim.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
