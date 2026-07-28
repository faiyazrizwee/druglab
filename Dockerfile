# Use miniconda3 as the base image to handle complex scientific dependencies (like RDKit and AutoDock Vina)
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the conda environment file first to leverage Docker cache for dependencies
COPY environment.yml .

# Create the conda environment and clean up cache to reduce image size
RUN conda env create -f environment.yml && \
    conda clean -afy

# Copy the rest of the application code
COPY . .

# Set the entrypoint to use the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "druglab", "python", "druglab.py"]

# Provide a default argument so running the container without arguments shows help
CMD ["--help"]
