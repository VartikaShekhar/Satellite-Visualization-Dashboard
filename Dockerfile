FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

# Upgrade Streamlit and install deps
RUN pip3 install -r requirements.txt && pip3 install streamlit>=1.32.0

# Set up .streamlit config to avoid permission error
RUN mkdir -p /app/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
enableXsrfProtection = false\n\
enableCORS = false\n\
port = 8501\n\
\n\
[browser]\n\
gatherUsageStats = false\n" > /app/.streamlit/config.toml

ENV STREAMLIT_CONFIG_DIR=/app/.streamlit
ENV HOME=/app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]