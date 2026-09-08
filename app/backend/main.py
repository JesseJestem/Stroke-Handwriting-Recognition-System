import base64
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from handwriting.core.logging import configure_logging
from handwriting.legacy.predictor import predict_from_files


configure_logging()
logger = logging.getLogger("handwriting.api")

app = FastAPI()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Middleware - allow to send request to backend
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], #allow all websites
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#~~~~~~~~~~~~~~~~~~~~~~
#Path creator
#~~~~~~~~~~~~~~~~~~~~~~

#Sample saving path
BASE_DIR = Path(__file__).resolve().parents[2] #take absolute file path from 2 lvl above (project folder)
IMAGE_DIR = BASE_DIR / "data" / "raw" / "images" #save images in data/raw/images/A/1image.png
STROKE_DIR = BASE_DIR / "data" / "raw" / "strokes" #save strokes in data/raw/strokes/A/1stroke.png
#create folder if not exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
STROKE_DIR.mkdir(parents=True, exist_ok=True)

#temp files for prediction (outside)
TEMP_DIR = Path(tempfile.gettempdir()) / "hybrid_handwriting_temp"
#create temp
TEMP_DIR.mkdir(parents=True, exist_ok=True)

#~~~~~~~~~~~~~~~~~~~~~~
#Classes
#~~~~~~~~~~~~~~~~~~~~~~

#stroke point model
class Point(BaseModel):
    x: float
    y: float
    t: float
    pressure: float
    pen_down: bool

#API reqest model
class SampleRequest(BaseModel):
    label: str #letter,number,symbhol
    image: str #base64 image
    strokes: list[Point] #list with moving [x, y, time]
    canvas_width: int
    canvas_height: int

#prediction model
class PredictRequest(BaseModel):
    image: str
    strokes: list[Point]
    canvas_width: int
    canvas_height: int

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#API ENDPOINTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Test GET endpoint: if open http://localhost:8000/ "/" - def root will start and ansver with message
@app.get("/")
def root():
    return{"message": "Hybrid Handwriting Data Collection API"}

#Sample save POST endpoint
@app.post("/save-sample")
#take JSON from request and check model SampleRequest after transform to python-object sample.
def save_sample(sample: SampleRequest):
    label = sample.label #create folder for every letter or symbhol

    if label.isalpha() and len(label) == 1:
        if label.isupper():
            folder_label = f"upper_{label}" #saving upper root
        else:
            folder_label = f"lower_{label}" #saving lower root
    else:
        folder_label = label

    #create folders for every symhol
    image_label_dir = IMAGE_DIR / folder_label #data/raw/images/A
    stroke_label_dir = STROKE_DIR / folder_label #data/raw/strokes/A

    image_label_dir.mkdir(parents=True, exist_ok=True)
    stroke_label_dir.mkdir(parents=True, exist_ok=True)
    
    #create uniqe timestamp mark
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    #create name of files
    image_filename = f"{folder_label}_{timestamp}.png"
    stroke_filename = f"{folder_label}_{timestamp}.json"

    #create file path
    image_path = image_label_dir / image_filename
    stroke_path = stroke_label_dir / stroke_filename

    #from data:image/png;base64,dhHBhvh2hvH... take second base part and decode it
    image_data = sample.image.split(",")[1] 
    image_bytes = base64.b64decode(image_data)

    with open (image_path, "wb") as f: #wb - png is binary file so save as write binary
        f.write(image_bytes)

    #prepare stroke data creating dict with data
    stroke_data = {
        "label": label,
        "image_path": str(image_path),
        "canvas_width": sample.canvas_width,
        "canvas_height": sample.canvas_height,
        "strokes": [point.model_dump() for point in sample.strokes], #save as dict
    }

    with open(stroke_path, "w", encoding="utf-8") as f:
        #save dict as JSON, ensure_ascii=False - no englis aslo ok, indent=2 - create readeble structure
        json.dump(stroke_data, f, ensure_ascii=False, indent=2)

    #count total of examples
    samples_count = len(list(image_label_dir.glob("*.png")))

    return{
        "status": "saved",
        "label": label,
        "image_path": str(image_path),
        "stroke_path": str(stroke_path),
        "points_count": len(sample.strokes),
        "samples_count": samples_count,
    }

#Prediction POST endpoint
@app.post("/predict")
def prediction_sample(sample: PredictRequest):

    #temp_files ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    #temp path
    image_path = TEMP_DIR / f"predict_{timestamp}.png"
    stroke_path = TEMP_DIR / f"predict_{timestamp}.json"

    #save temp img
    try:
        image_data = sample.image.split(",")[1] #take only base64 part of request
        image_bytes = base64.b64decode(image_data) #encode base 64
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        #save temp stroke
        stroke_data = {
            "label": None, #dont know label before predict
            "image_path": str(image_path),
            "canvas_width": sample.canvas_width,
            "canvas_height": sample.canvas_height,
            "strokes": [point.model_dump() for point in sample.strokes], #model_dump() - pydantic object to python
        }
        with open(stroke_path, "w", encoding="utf-8") as f:
            json.dump(stroke_data, f, ensure_ascii=False, indent=2) #python to json ensure_ascii=False-not only Eng, indent=2-readeble formate
        
        #result saving
        result = predict_from_files(
            image_path=image_path,
            stroke_path=stroke_path,
            top_k=3,
        )

        return{
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "top_3": result["top_k"],
        }
    
    #display error on console logs
    except Exception as e:
        logger.exception("Prediction failed")
        #500-server error to frontend + details in JSON- "detail": "Model not found: saved_models/hybrid_letters.keras"
        raise HTTPException(status_code=500, detail=str(e))
    
    #delete temp files after prediction
    finally:
        image_path.unlink(missing_ok=True) #unlink- delete, missing ok- ok if deleted
        stroke_path.unlink(missing_ok=True)