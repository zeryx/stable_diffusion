FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime


WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential git curl ssh software-properties-common
RUN  add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.10 python3.10-venv
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
# Install the AWS cli separately to prevent issues with boto being written over
RUN pip3.10 install awscli

# Similarly, if you're using GCP be sure to update this command to install gsutil
# RUN apt-get install -y curl
# RUN curl -sSL https://sdk.cloud.google.com | bash
# ENV PATH="$PATH:/root/google-cloud-sdk/bin"

ENV VENV /opt/venv
# Virtual environment
RUN python3.10 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
COPY ./requirements.txt /root
RUN pip install -r /root/requirements.txt

# Copy the actual code
COPY . /root

# This tag is supplied by the build script and will be used to determine the version
# when registering tasks, workflows, and launch plans
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
