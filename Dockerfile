FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
COPY src ./src

RUN apt-get update && apt-get install -y git

# Remove the git config line with the hardcoded token
# Instead, use ARG to allow passing the token at build time
ARG GITHUB_TOKEN
RUN git config --global url."https://$(echo ${GITHUB_TOKEN}@github.com/)".insteadOf "https://github.com/"

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Add a step to verify the lock file
RUN poetry lock --check

EXPOSE 8080

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
