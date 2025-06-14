# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# It's good practice to set a default port, can be overridden at runtime
ENV PORT=8000

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# For example, if you had a package that needed gcc or other build tools:
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

# Install Python dependencies
# Using --no-cache-dir can reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes app.py, models.py, graph_builder.py, state.py, and the agents/ & memory/ directories.
COPY . .

# Expose the port the app runs on
EXPOSE 8001

# Command to run the Uvicorn server
# Using shell form of CMD to allow environment variable expansion
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
