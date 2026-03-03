from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = FastAPI()

# Load model once at startup
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize to correct model size
    image = image.resize((160, 160))

    # Convert to numpy
    input_data = np.array(image, dtype=np.float32)

    # Normalize
    input_data = input_data / 255.0

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    prediction = int(np.argmax(output))
    confidence = float(np.max(output))

    return {
        "prediction": prediction,
        "confidence": confidence
    }