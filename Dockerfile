FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY arp ./arp
RUN pip install --no-cache-dir .
COPY prompts ./prompts
COPY configs ./configs
COPY evals ./evals
COPY scripts ./scripts
ENV ARP_HTTP_MODE=replay ARP_HOME=/app/.arp
ENTRYPOINT ["arp"]
CMD ["--help"]
