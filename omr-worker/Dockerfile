# 1. Supported Java 11 runtime (Eclipse Temurin)
FROM eclipse-temurin:11-jre

# 2. Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip unzip wget && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Install Audiveris using the official Ubuntu .deb package
WORKDIR /opt
RUN wget https://github.com/Audiveris/audiveris/releases/download/5.9.0/Audiveris-5.9.0-ubuntu22.04-x86_64.deb && \
    dpkg -i Audiveris-5.9.0-ubuntu22.04-x86_64.deb || true && \
    apt-get update && apt-get -f install -y && \
    rm Audiveris-5.9.0-ubuntu22.04-x86_64.deb

# 4. Set Audiveris home directory
ENV AUDIVERIS_HOME=/usr/share/audiveris

# 5. Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy worker code
COPY worker.py .

# 7. Start the worker
CMD ["python", "worker.py"]
