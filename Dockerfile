FROM python:3.13-slim

# System deps for lightgbm/shap/osqp (lean set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libomp-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src
COPY services /app/services
COPY scripts /app/scripts
COPY constraints.yaml costs.yaml /app/
# optional artifacts/expected if you want to bake them
# COPY artifacts /app/artifacts
# COPY expected /app/expected

# Install
RUN python -m pip install --upgrade pip wheel setuptools && \
    pip install -e . && \
    pip cache purge

EXPOSE 8000
ENV PORT=8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
