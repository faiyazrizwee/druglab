# Multi-platform base image with Miniconda for scientific cheminformatics stacks
FROM continuumio/miniconda3:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install OS libraries required for RDKit, OpenBabel, and 3D graphics rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the conda environment file first to leverage Docker layer caching
COPY environment.yml .

# Fast environment creation using libmamba solver
RUN conda install -n base conda-libmamba-solver -y && \
    conda config --set solver libmamba && \
    conda env create -f environment.yml && \
    conda clean -afy

# Copy application source code
COPY . .

# Install DrugLab package into the conda environment
RUN conda run -n druglab pip install --no-deps -e .

# Expose port for the interactive 3D Pocket Tuner web GUI
EXPOSE 8080

# Set entrypoint to run druglab within the activated conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "druglab", "druglab"]

# Default argument displays help
CMD ["--help"]
