FROM pytorch/pytorch:latest
COPY requirements.txt /tmp/requirements.txt

# Install other packages
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    mkdir -p /workspace/ro_codebase && \
    mkdir /workspace/data
WORKDIR /workspace/ro_codebase

# Activate the Conda environment
ENV PYTHONPATH=/workspace/ro_codebase:$PYTHONPATH
