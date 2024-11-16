from fastapi import (
    FastAPI,
    UploadFile,
    BackgroundTasks,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
import fitz
from dotenv import load_dotenv
import os
from openai import OpenAI
from ai import create_repository

app = FastAPI()

UPLOAD_DIR = Path("pdf_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to index.html file
INDEX_HTML_PATH = Path("index.html")


@app.get("/")
async def root():
    """
    Serve the index.html file at the root URL (/).
    """
    if INDEX_HTML_PATH.exists():
        return FileResponse(INDEX_HTML_PATH)
    raise HTTPException(status_code=404, detail="index.html not found")


# Store connected WebSocket clients
connected_clients = []


@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


def progress_callback(cbt, progress):
    message = f"{cbt} Progress: {progress * 100:.2f}%"
    print(message)
    for client in connected_clients:
        try:
            client.send_text(message)
        except:
            connected_clients.remove(client)


def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        return text


def really_long_progress_fn_with_callbacks(file_path: Path, progress_callback):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
            progress_callback(f"Load pdf page {page.number}", page.number / len(doc))
        print("Finished processing PDF")
        print(text[:100])
        create_repository(text, progress_callback)
        print("Finished creating repository")
        return text


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    print(f"Received file: {file.filename}")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    save_path = UPLOAD_DIR / file.filename
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        really_long_progress_fn_with_callbacks, save_path, progress_callback
    )

    return JSONResponse(
        content={"message": "File uploaded successfully", "file": str(save_path)}
    )


@app.get("/status/")
def status():
    return {"status": "Server is running"}
