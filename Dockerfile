# Use a polyglot base image with Node and Python
FROM nikolaik/python-nodejs:python3.11-nodejs20

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json and install Node dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application
COPY . .

# Create downloads folder for YT service
RUN mkdir -p downloads

# Expose ports (FastAPI on 8000, Node on 10000)
EXPOSE 8000
EXPOSE 10000

# Start both services using concurrently
CMD ["npx", "concurrently", "\"python3 -m uvicorn ai_service:app --host 0.0.0.0 --port 8000\"", "\"node index.js\""]
