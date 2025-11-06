import video
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from video import read_vid


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

