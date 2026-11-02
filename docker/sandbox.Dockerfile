# Image the ops agent runs untrusted test commands in. No network at run time (--network none).
ARG BASE=python:3.11-slim
FROM ${BASE}
# Optional offline wheelhouse (docker/wheels) for builders without PyPI access.
COPY wheels/ /tmp/wheels/
RUN if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
      pip install --no-cache-dir --no-index --find-links /tmp/wheels pytest==8.3.5; \
    else pip install --no-cache-dir pytest==8.3.5; fi \
 && rm -rf /tmp/wheels && useradd -m -u 1000 runner
USER runner
WORKDIR /work
