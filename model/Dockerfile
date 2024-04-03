# Use an appropriate base image for PyTorch
FROM python:3.12.2-bookworm

# Create a working directory for the application
WORKDIR /app

# Copy the application code to the working directory
COPY ./conf ./conf
COPY ./crawler ./crawler
COPY ./data ./data
COPY ./model ./model
COPY ./tests ./tests
COPY ./utils ./utils
COPY ./serve.py ./serve.py
COPY ./requirements-serve.txt ./requirements.txt
COPY ./artifacts/sp2000.model ./artifacts/sp2000.model
COPY ./artifacts/sp2000.vocab ./artifacts/sp2000.vocab
COPY ./artifacts/stockfish-ubuntu-x86-64-avx2 ./artifacts/stockfish-ubuntu-x86-64-avx2
COPY ./artifacts/model.pt ./artifacts/model.pt

# Install required dependencies (if not already in the base image)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the application will run on (if applicable)
EXPOSE 8080

# Define the command to run the application
ENTRYPOINT ["gunicorn", "serve:app", "-b 0.0.0.0:8080"]