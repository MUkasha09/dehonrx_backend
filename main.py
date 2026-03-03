from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = FastAPI()

# Load model once at startup
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # adjust if needed

    input_data = np.expand_dims(image, axis=0)
    input_data = np.array(input_data, dtype=np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output)
    confidence = float(np.max(output))

    return {
        "prediction": int(prediction),
        "confidence": confidence
    }