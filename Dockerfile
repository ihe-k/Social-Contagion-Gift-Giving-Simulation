FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY . .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
