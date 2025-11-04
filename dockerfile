FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# system deps (if you later add geo libs, you can apt-get them here)
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./pyproject.toml
COPY src ./src

RUN pip install --no-cache-dir fastapi uvicorn pydantic pydantic-settings pandas numpy sqlmodel sqlalchemy httpx pytest
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "haven.api.http:app", "--host", "0.0.0.0", "--port", "8000"]
