from fastapi import File, UploadFile
from PIL import Image
import io
import numpy as np


async def process_img_for_prediction(image: Image):
    image = image.resize((256, 256)).convert("RGB")
    image = np.array(image, dtype=float)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    print(image.shape)
    return image


async def read_img_file(img_file: UploadFile = File(...)) -> Image:
    image_data = await img_file.read()
    image = Image.open(io.BytesIO(image_data))
    return image
