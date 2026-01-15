# Dockerfile for LORE Application
# Main container for running optimization experiments

FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ==========================================================================
# Production image
# ==========================================================================
FROM python:3.11-slim-bookworm AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r lore && useradd -r -g lore lore

# Set up application
WORKDIR /app
COPY --chown=lore:lore . .

# Create directories
RUN mkdir -p /app/results /app/logs && \
    chown -R lore:lore /app/results /app/logs

# Switch to non-root user
USER lore

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose port for API mode
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "src.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
