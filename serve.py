import gc
import argparse
import asyncio
import os
from io import BytesIO
from time import time
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import ray
import torch
import uvicorn
from loguru import logger
from fastapi import FastAPI,  UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

from trellis_generator.trellis_gs_processor import GaussianProcessor
from config import config


# Setting up default attention backend for trellis generator: can be 'flash-attn' or 'xformers'
os.environ['ATTN_BACKEND'] = config.model.attn_backend


def get_args() -> argparse.Namespace:
    """ Function for getting arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=config.server.host)
    parser.add_argument("--port", type=int, default=config.server.port)
    return parser.parse_args()


executor = ThreadPoolExecutor(max_workers=config.server.max_workers)


class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    """ Function that loading all models and warming up the generation."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=config.ray.ignore_reinit_error)
        logger.info("Ray initialized successfully.")
    
    major, minor = torch.cuda.get_device_capability(0)

    if major == 9:
        vllm_flash_attn_backend = "FLASH_ATTN"
    else:
        vllm_flash_attn_backend = "FLASHINFER"

    try:
        logger.info("Loading Trellis generator model...")
        app.state.trellis_generator = GaussianProcessor(config.image.default_image_shape, vllm_flash_attn_backend)
        app.state.trellis_generator.load_models()
        clean_vram()
        logger.info("Model loading is complete.")
    except Exception as e:
        logger.exception(f"Exception during model loading: {e}")
        raise SystemExit("Model failed to load → exiting server")

    try:
        logger.info("Warming up Trellis generator...")
        app.state.trellis_generator.warmup_generator()
        clean_vram()
        logger.info("Warm-up is complete. Server is ready.")

    except Exception as e:
        logger.exception(f"Exception during warming up the generator: {e}")
        raise SystemExit("Warm-up failed → exiting server")

    yield
    
    # Cleanup Ray on shutdown
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown successfully.")


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def clean_vram() -> None:
    """ Function for cleaning VRAM. """
    gc.collect()
    torch.cuda.empty_cache()


def generation_block(prompt_image: Image.Image, seed: int = -1) -> BytesIO:
    """ Function for 3D data generation using provided image"""

    t_start = time()
    buffer, _ = app.state.trellis_generator.get_model_from_image_as_ply_obj(image=prompt_image, seed=seed)

    t_get_model = time()
    logger.debug(f"Model Generation took: {(t_get_model - t_start)} secs.")

    clean_vram()

    t_gc = time()
    logger.debug(f"Garbage Collection took: {(t_gc - t_get_model)} secs")

    return buffer


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> Response:
    """ Generates a 3D model as a PLY buffer """

    logger.info("Task received. Prompt-Image")

    try:
        # Validate file type
        if not prompt_image_file.content_type or not prompt_image_file.content_type.startswith('image/'):
            logger.warning(f"Invalid content type: {prompt_image_file.content_type}")
        
        contents = await prompt_image_file.read()
        
        # Validate file size
        max_size = config.image.max_file_size_mb * 1024 * 1024
        if len(contents) > max_size:
            logger.error(f"File size {len(contents)} exceeds maximum {max_size}")
            return Response(content=f"File size exceeds {config.image.max_file_size_mb}MB limit", status_code=400)
        
        if len(contents) == 0:
            logger.error("Empty file received")
            return Response(content="Empty file received", status_code=400)
        
        prompt_image = Image.open(BytesIO(contents))
        
        # Validate image dimensions
        if prompt_image.size[0] < config.image.min_width or prompt_image.size[1] < config.image.min_height:
            logger.error(f"Image too small: {prompt_image.size}")
            return Response(
                content=f"Image must be at least {config.image.min_width}x{config.image.min_height} pixels",
                status_code=400
            )

        loop = asyncio.get_running_loop()
        buffer = await loop.run_in_executor(executor, generation_block, prompt_image, seed)
        logger.info("Task completed.")

        return StreamingResponse(buffer, media_type="application/octet-stream")
    
    except Exception as e:
        logger.exception(f"Error during model generation: {e}")
        return Response(content=f"Error processing image: {str(e)}", status_code=500)


@app.get("/version", response_model=str)
async def version() -> str:
    """ Returns current endpoint version."""
    return app.version


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    """ Return if the server is alive and model status """
    model_loaded = hasattr(app.state, 'trellis_generator') and app.state.trellis_generator is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "ray_initialized": ray.is_initialized()
    }


if __name__ == "__main__":
    args: argparse.Namespace  = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
