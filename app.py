from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import main as photo_mosaic

app = FastAPI()
templates = Jinja2Templates(directory="templates")

upload_folder = Path("./uploads")
output_folder = Path("./outputs")

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return templates.TemplateResponse("upload.html", {"request": {}})

@app.post("/upload/")
async def create_mosaic(
    file: UploadFile = File(...),
    text_prompt: str = Form(...),
    cell_size: int = Form(...),
    model_choice: str = Form(...),
):
    target_image_path = upload_folder / file.filename
    with target_image_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_filename = photo_mosaic.create_mosaic_from_uploaded_image(
        str(target_image_path), text_prompt, cell_size, model_choice, output_folder=output_folder
    )
    return FileResponse(output_folder / output_filename)