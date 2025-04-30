from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io 

app = FastAPI()

@app.get("/")
def greet():
    return{"Message":"Bonjour !"}

def load():
    model_path = "trained_plant_disease_model.keras"
    model = load_model(model_path, compile=False)
    return model
#Chargement du model
model = load()

def preproccess(img):
    img = img.resize((224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile):
    image_data = await file.read()

    #Ouvrir l'image 
    img =Image.open(io.BytesIO(image_data))

    # preproccessing
    img_processed = preproccess(img)

    #prediction
    prediction = model.predict(img_processed)
    rec = prediction[0][0].tolist

    return {"Prediction": rec}
