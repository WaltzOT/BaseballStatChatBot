FROM nvcr.io/nvidia/pytorch:23.04-py3

# Install NeMo Toolkit dependencies for CPU only
RUN pip install nemo_toolkit[nlp] --no-cache-dir
RUN pip install pydantic==1.10.12 --force-reinstall

# Set working directory
WORKDIR /workspace

# Copy project files
COPY ./data /workspace/data
COPY ./scripts /workspace/scripts
COPY ./models /workspace/models

# Set output model folder
ENV MODEL_DIR=/workspace/models

# Default command to run
CMD ["python", "scripts/initialTraining.py"]


