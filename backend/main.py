import uvicorn
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from backend.ml_logic.data import load_nii_from_gcp
from backend.ml_logic.model1 import glioma_label_unet_model
import tensorflow as tf

# Define the FastAPI app
app = FastAPI()

# Load the model
MODEL_FILE_NAME = 'model_glioma_t1_nii_3dUnet.h5'
MODEL_PATH = f'../saved_models/{MODEL_FILE_NAME}' # set the path where you want to save the model
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Model '{MODEL_FILE_NAME}' loaded successfully.")

# Define the input and output schema for the API
class InputSchema(BaseModel):
    nii_file_name: str
    channel: str

class OutputSchema(BaseModel):
    prediction: float

# Define the API endpoint
@app.post("/predict", response_model=OutputSchema)
async def predict(input_data: InputSchema):
    nii_file_name = input_data.nii_file_name
    channel = input_data.channel
    img = load_nii_from_gcp(nii_file_name + '_' + channel + '.nii', 'cache_folder')
    img = img.get_fdata()
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.expand_dims(img, axis=-1)  # Add channel dimension
    prediction = model.predict(img)
    return {"prediction": prediction[0][0]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
