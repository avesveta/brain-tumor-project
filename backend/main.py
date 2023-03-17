import uvicorn
from fastapi import FastAPI, File, UploadFile
import shutil
import tensorflow as tf
from ml_logic.preprocessor import preprocess_nii_for_test
from pathlib import Path
from tempfile import NamedTemporaryFile

# Define the FastAPI app
app = FastAPI()

# Load the model
MODEL_FILE_NAME = 'model_glioma_t1_nii_3dUnet.h5'
MODEL_PATH = Path('saved_models', MODEL_FILE_NAME) # set the path where you want to save the model

dir = Path.cwd()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"file_size": len(file)}

    def make_tmp_file(upload_file: UploadFile) -> Path:
        try:
            suffix = Path(upload_file.filename).suffix
            with NamedTemporaryFile(delete=False, suffix=suffix, dir=dir) as tmp:
                shutil.copyfileobj(upload_file.file, tmp)
                tmp_path = Path(tmp.name)
        finally:
            upload_file.file.close()

        return tmp_path

    file_path = make_tmp_file(file)

    # load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Preprocess the uploaded file
    processed_img_data = preprocess_nii_for_test(file_path)

    file_path.unlink()

    # Make predictions using the saved model
    predictions = (model.predict(processed_img_data) > 0.5).astype("int32")
    prediction = int(predictions[0][0])
    if prediction == 0:
        prediction_str = 'the tumor is a low-grade glioma'
    else:
        prediction_str = 'the tumor is a high-grade glioma'


    # Return the predictions
    return {"predictions": prediction_str}

@app.get("/hello")
async def hello():
    return {"predictions": 'hello'}
