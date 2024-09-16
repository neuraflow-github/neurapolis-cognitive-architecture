FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Add a step to verify the lock file
RUN poetry lock --check

EXPOSE 8080

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
