# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# We assume you still need to install paho-mqtt if it wasn't installed in the environment used for requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install paho-mqtt

# Make port 8501 (Streamlit's default) available to the world outside this container
EXPOSE 8501

# Run Streamlit when the container launches
# NOTE: The docker-compose.yml specifies the command, but this is a good default
# CMD ["streamlit", "run", "app.py"]