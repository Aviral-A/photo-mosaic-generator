import os
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from embeddings_utils import EmbeddingsModel
import main as photo_mosaic
from database import Database

templates = Jinja2Templates(directory="../templates")

# Initialize the EmbeddingsModel
embeddings_model = EmbeddingsModel()


def get_application(database):
    app = FastAPI()

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
        target_image_path = Path(f"{uuid.uuid4()}-{file.filename}")
        with target_image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_filename = photo_mosaic.create_mosaic_from_uploaded_image(
            str(target_image_path),
            text_prompt,
            cell_size,
            model_choice,
            output_folder=Path("./outputs"),
        )

        # Compute the embeddings for the given image
        embeddings = embeddings_model.compute_embeddings(str(target_image_path))

        database.add_image(
            filename=str(target_image_path),
            output_filename=output_filename,
            embeddings=embeddings,
        )

        os.remove(str(target_image_path))
        os.remove(f"./outputs/{output_filename}")

        return FileResponse(f"./outputs/{output_filename}")

    return app


db = Database()
app = get_application(db)