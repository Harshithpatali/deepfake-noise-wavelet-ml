from pathlib import Path
import tempfile,os
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from production_inference import predict_image

app=FastAPI(title="DeepFake Noise Wavelet ML")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400,"Upload an image file.")
    data=await file.read()
    if len(data)>10*1024*1024: raise HTTPException(413,"Maximum upload size is 10 MB.")
    suffix=Path(file.filename or ".jpg").suffix or ".jpg"
    fd,path=tempfile.mkstemp(suffix=suffix); os.close(fd)
    try:
        Path(path).write_bytes(data)
        return predict_image(path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Inference failed: {type(e).__name__}: {e}")
    finally: Path(path).unlink(missing_ok=True)
