FROM python:3.12.4-slim

WORKDIR /nicbot

# Install tree utility (optional, for debugging)
RUN apt-get update && apt-get install -y tree && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all other project files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "prompt.py", "--server.port=8501", "--server.address=0.0.0.0"]
