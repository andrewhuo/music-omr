FROM python:3.11-slim

# ✅ Set working directory
WORKDIR /app

# ✅ Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy your code
COPY . .

# ✅ Run the server on port 8080 (Cloud Run standard)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
