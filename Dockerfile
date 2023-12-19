FROM nvcr.io/nvidia/pytorch:23.10-py3
COPY requirements.txt requirements.txt
ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && \
    apt-get install screen tmux nvtop htop -y && \
    conda run -n base python -m pip install --upgrade pip && \
    conda run -n base python -m pip uninstall opencv-python && \
    conda run -n base python -m pip install opencv-python-headless && \
