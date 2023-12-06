# We start from the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:22.05-py3

# Set the working directory
WORKDIR /app

# Build the python environment
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .
