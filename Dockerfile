FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
COPY requirements.txt /tmp/requirements.txt

# Install other packages
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    mkdir -p /workspace/ro_codebase && \
    mkdir /workspace/data
WORKDIR /workspace/ro_codebase

# Activate the Conda environment
ENV PYTHONPATH=/workspace/ro_codebase:$PYTHONPATH
