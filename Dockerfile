# Optional packaging recipe; see docs/operations.md for verification status.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib HF_HUB_OFFLINE=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip==26.3 \
    && python -m pip install --no-cache-dir '.[models,app,api]' \
    && useradd --create-home --uid 10001 inspector
COPY app ./app
COPY .streamlit ./.streamlit
USER inspector
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
