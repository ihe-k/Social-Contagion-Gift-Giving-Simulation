FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip early
RUN pip install --upgrade pip setuptools wheel

# Install requirements with no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
