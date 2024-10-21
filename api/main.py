import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from torchvision.datasets.phototour import read_image_file
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
import tensorflow as tf


app=FastAPI()




Model = keras.models.load_model("../saved_models/potatoes.keras")
Class_names=["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async  def ping():
     return "Hello God, please help me !!!"

def read_file_as_image(data)-> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))

    return image

@app.post("/predict")
async  def predict(
        file:UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch=np.expand_dims(image, 0)
    predictions = Model.predict(img_batch)
    confidence = np.max(predictions[0])*100
    return{
        "Class": Class_names[np.argmax(predictions[0])],
         "Confidence":confidence
    }

if __name__ =="__main__":
    uvicorn.run(app,host='localhost', port=9000)
