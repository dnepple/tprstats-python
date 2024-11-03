# syntax=docker/dockerfile:1

# installs Conda, Poetry & Cookiecutter using miniforge 
FROM condaforge/miniforge3:latest

RUN apt update -y && \
  apt upgrade -y && \
  apt install -y --no-install-recommends \
  git \
  make \
  curl

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$HOME/root/.local/bin:${PATH}"