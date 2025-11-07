"""
main.py
@author: Conner Santa Monica
Project: Motion Predictor

Description:
Entry point for the FastAPI web application. 
Handles routing for static assets, video upload endpoints, 
and video source selection (webcam or uploaded file). 
Integrates with video.py for frame processing and motion detection.

Modules:
- FastAPI (server & routing)
- uvicorn (development server)
- video.py (video stream logic)
"""

from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from video import read_vid, motion_log, set_video_source
import os
import shutil


app = FastAPI()
templates = Jinja2Templates(directory="templates")
UPLOAD_DIR = "vid"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/set_source")
def set_source(source: str = Form(...)):
    try:
        if source == "webcam":
            set_video_source(0)
            return {"message": "Using webcam"}
        elif source == "uploaded":
            latest_files = sorted(
                os.listdir(UPLOAD_DIR),
                key=lambda x: os.path.getmtime(os.path.join(UPLOAD_DIR, x))
            )
            if not latest_files:
                return {"error": "No uploaded videos found"}
            path = os.path.join(UPLOAD_DIR, latest_files[-1])
            set_video_source(path)
            return {"message": f"Using uploaded video: {latest_files[-1]}"}
        else:
            return {"error": f"Invalid source: {source}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload_video")
async def upload_video(file: UploadFile):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        set_video_source(file_path)
        return {"message": f"File '{file.filename}' uploaded and selected as current source"}
    except Exception as e:
        return {"error": f"Upload failed: {e}"}

@app.get("/video")
def video_feed():
    return StreamingResponse(
        read_vid(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/motion/latest")
def latest_motion_data():
    if not motion_log:
        return JSONResponse({"message": "No data yet"})
    return JSONResponse(motion_log[-1])

@app.get("/motion/all")
def all_motion_data():
    return JSONResponse(motion_log)

