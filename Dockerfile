FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application files
COPY . .

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "Test.py"]
