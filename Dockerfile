FROM ubuntu:24.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip vim \
    build-essential curl git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv "${VIRTUAL_ENV}"

# Upgrade pip toolchain in venv
RUN "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel

# Install Poetry into the SAME venv
RUN "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir "poetry==1.8.3"

# Sanity check: Poetry should be in PATH via /opt/venv/bin
RUN poetry --version

# App root
WORKDIR /app

# Copy dependency manifests first (better layer caching)
COPY pyproject.toml poetry.lock* /app/

# Install deps into the SAME venv (no nested venvs)
# Optional build-arg to include a group, e.g. --build-arg POETRY_WITH=cpu or gpu
ARG POETRY_WITH=
RUN poetry config virtualenvs.create false \
 && (poetry lock --no-interaction --no-update || poetry lock --no-interaction) \
 && poetry install --no-interaction --no-root --only main ${POETRY_WITH:+--with ${POETRY_WITH}}

# Copy the rest of the project
COPY . /app

# Entrypoint & default command
# Ensure your repo has docker/entrypoint.sh and it uses /app as the working dir
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "mimo_ofdm_over_cdl/training.py"]
