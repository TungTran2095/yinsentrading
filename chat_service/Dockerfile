FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8005

RUN mkdir -p /app/data
COPY data /app/data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
