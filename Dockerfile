# Use official Python 3.11 slim image
FROM python:3.11-slim-buster

# Install system packages
RUN apt update -y && apt install -y awscli

# Set working directory inside container
WORKDIR /app

# Copy project files to the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install -r requirements.txt

# Set default command to run the app
CMD ["python", "app.py"]
