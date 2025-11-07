import video
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from video import read_vid, motion_log


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "Name": "User"})

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

