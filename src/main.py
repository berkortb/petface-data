from fastapi import FastAPI, File, UploadFile, HTTPException
from src.services.vision import get_emotion_from_image, predict_img_classification
from src.services.img_process import read_img_file
from src.params import PET_LABELS
import asyncio

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/")
async def predict_uploaded_img(img_file: UploadFile = File(...)):
    try:
        image = await read_img_file(img_file)
        emotion_res, pet_type_idx = await asyncio.gather(get_emotion_from_image(image),
                                                         predict_img_classification(image))
        return {"prediction": emotion_res, "classification": PET_LABELS[pet_type_idx]}
    except Exception as e:
        print("Error in prediction", e)
        raise HTTPException(
            status_code=500,
            detail="An error with the prediction ocurred",
        ) from e
