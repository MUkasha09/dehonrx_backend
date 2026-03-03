from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = FastAPI()

CLASS_NAMES = [
    'CORN_Bacterial Leaf Streak',
    'CORN_Common_rust',
    'CORN_Gray_leaf_spot',
    'CORN_Healthy',
    'CORN_Maize Chlorotic Mottle Virus',
    'PEANUT_ALTERNARIA LEAF SPOT',
    'PEANUT_HEALTHY',
    'PEANUT_LEAF SPOT (EARLY AND LATE)',
    'PEANUT_ROSETTE',
    'PEANUT_RUST'
]

# Load model once
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((160, 160))

    input_data = np.array(image, dtype=np.float32)

    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_index = int(np.argmax(output))
    confidence = float(output[pred_index])

    return {
        "disease": CLASS_NAMES[pred_index],
        "confidence": round(confidence, 4)
    }