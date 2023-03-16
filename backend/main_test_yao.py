import uvicorn
from fastapi import FastAPI, File, UploadFile
import shutil
import tensorflow as tf
from ml_logic.preprocessor import preprocess_nii_for_test
from pathlib import Path

# Define the FastAPI app
app = FastAPI()

# Load the model
MODEL_FILE_NAME = 'model_glioma_t1_nii_3dUnet.h5'
MODEL_PATH = Path('..', 'saved_models', MODEL_FILE_NAME) # set the path where you want to save the model

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # read the contents of the uploaded file
    # do something with the contents of the file here
    return {"filename": file.filename}

@app.get("/predict")
def predict(file: UploadFile = File(...)):


    # load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Preprocess the uploaded file
    processed_img_data = preprocess_nii_for_test(file)

    # Make predictions using the saved model
    predictions = model.predict(processed_img_data)

    # Return the predictions
    return {"predictions": predictions}
