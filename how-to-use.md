
# How to Use the SUPIR Docker Container

This document provides instructions on how to build and run the Docker container for the SUPIR application. The container is designed to be a dual-mode worker, allowing you to run it either as an interactive Gradio web application or as a serverless API endpoint on platforms like RunPod.

## Prerequisites

- Docker installed on your system.
- An NVIDIA GPU with CUDA support is highly recommended for decent performance.

## Building the Docker Image

First, you need to build the Docker image from the `Dockerfile` located in the root of the project. This process will install all the necessary dependencies and download the required models.

**Note:** The model download step will significantly increase the size of your Docker image.

Open a terminal in the project's root directory and run the following command:

```bash
docker build -t supir-app .
```

This will create a Docker image named `supir-app`.

## Running the Container

The container's behavior is controlled by the `MODE_TO_RUN` environment variable.

### Mode 1: Interactive Gradio Web UI (`pod` mode)

This mode launches the Gradio web interface, allowing you to use SUPIR through your browser.

To run the container in this mode, use the following command. This will map the container's port 7860 to port 7860 on your local machine.

```bash
docker run -it -p 7860:7860 --gpus all -e MODE_TO_RUN=pod supir-app
```

Once the container is running and the Gradio app has started, you can access the web UI by opening your browser and navigating to:

[http://localhost:7860](http://localhost:7860)

### Mode 2: Serverless API (`serverless` mode)

This mode is intended for programmatic use, such as deploying on a serverless platform like RunPod. It exposes an API endpoint that accepts a JSON payload to process an image.

To run the container in this mode, use the following command:

```bash
docker run -it --gpus all -e MODE_TO_RUN=serverless supir-app
```

The container will start and listen for requests.

#### API Request

You can send a POST request to the endpoint with a JSON payload. Here is an example using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "input": {
        "img_path": "https://path/to/your/image.jpg",
        "upscale": 2,
        "sampler_mode": "TiledRestoreEDMSampler",
        "seed": 12345
    }
}' http://localhost:8000/run
```

*(Note: The exact URL and port will depend on how you deploy the serverless worker.)*

#### Input Parameters

The `input` object in the JSON payload can contain any of the parameters that are available in the `run_supir_cli.py` script. The most important one is `img_path`, which must be a URL to the image you want to process.

#### API Response

The API will return a JSON object containing the URL of the processed image, which will be stored in a RunPod bucket.

```json
{
  "image_url": "https://runpod.io/bucket/path/to/output.png"
}
```
