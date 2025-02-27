from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import numpy as np  # Explicitly import NumPy
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor
MODEL_NAME = "nguyenkhoa/dinov2_Liveness_detection_v2.1.4"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file contents into memory
        contents = await file.read()
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Process image manually if necessary - avoid return_tensors in processor
        inputs = processor(images=image)
        
        # Convert to tensor manually
        pixel_values = torch.tensor(np.array(inputs["pixel_values"])).unsqueeze(0)
        
        # Run inference with the manually created tensor
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        
        # Get prediction
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
        
        # Calculate confidence
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence = float(probabilities[predicted_class_idx].item())
        
        # Return results
        return {
            "class_id": predicted_class_idx,
            "class_label": predicted_label,
            "confidence": confidence
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }

@app.get("/api_working")
async def api_working():
    return {"message": "API is working"}

@app.get("/")
async def root():
    return {"message": "Liveness Detection API is running. Use /predict/ endpoint to detect liveness."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
