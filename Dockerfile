FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (leverages Docker cache for speed)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the entire app
COPY . /app

# Expose port 8000 for the web server
EXPOSE 8000

# Start the uvicorn server
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
