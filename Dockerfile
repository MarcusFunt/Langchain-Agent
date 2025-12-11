FROM vllm/vllm-openai:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install compilation tools and Python headers before installing Python dependencies.
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
