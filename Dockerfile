        # Use a lightweight Python base image
        FROM python:3.9-slim-buster

        # Set the working directory inside the container
        WORKDIR /app

        # Copy the requirements file and install Python dependencies
        # Using --no-cache-dir to reduce image size
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt

        # Copy your application source code
        # This includes src/ (your API, data_processing etc.)
        # and mlruns/ (where MLflow stores artifacts and models locally)
        COPY src/ /app/src/
        COPY mlruns/ /app/mlruns/

        # Expose the port that FastAPI will listen on
        EXPOSE 8000

        # Command to run the FastAPI application using Uvicorn
        # --host 0.0.0.0 makes the server accessible from outside the container
        # --port 8000 specifies the port to listen on
        # src.api.main:app refers to the 'app' object in 'main.py' inside 'src/api'
        CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
        