# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip/setuptools/wheel and install dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app on all network interfaces
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
