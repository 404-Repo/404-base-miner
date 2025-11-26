import gc
import argparse
import asyncio
import os
from io import BytesIO
from time import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from loguru import logger
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import Response, StreamingResponse
from trellis_generator.trellis_gs_processor import GaussianProcessor


os.environ['ATTN_BACKEND'] = 'flash-attn' # 'flash-attn', 'xformers'

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=1)

major, minor = torch.cuda.get_device_capability(0)

if major == 9:
    vllm_flash_attn_backend = "FLASH_ATTN"
else:
    vllm_flash_attn_backend = "FLASHINFER"

gaussian_processor = GaussianProcessor((int(1024/2), int(1024/2), 3), vllm_flash_attn_backend)
gaussian_processor.load_models()


def get_args():
    """ Function for getting arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    return parser.parse_args()


def clean_vram() -> None:
    """ Function for cleanong VRAM   """
    gc.collect()
    torch.cuda.empty_cache()


def generation_block(prompt_image: Image.Image) -> BytesIO:
    """ Function for 3D data generation using provided image"""

    t_start = time()
    buffer, _ = gaussian_processor.get_model_from_image_as_ply_obj(image=prompt_image, seed=-1)

    t_get_model = time()
    logger.debug(f"Model Generation took: {(t_get_model - t_start)} secs.")

    clean_vram()

    t_gc= time()
    logger.debug(f"Garbage Collection took: {(t_gc - t_get_model)} secs")

    return buffer


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...)) -> Response:
    """ Generates a 3D model as a PLY buffer """

    logger.info("Task received. Prompt-Image")

    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(executor, generation_block, prompt_image)
    logger.info(f"Task completed.")

    return StreamingResponse(buffer, media_type="application/octet-stream")


if __name__ == "__main__":
    args = get_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, timeout_keep_alive=300)
