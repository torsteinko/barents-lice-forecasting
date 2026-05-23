from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chat_service import SiteChatService
from .config import ROOT_DIR

load_dotenv(ROOT_DIR / ".env")

VIEWER_DIR = ROOT_DIR / "viewer"
chat_service = SiteChatService()

app = FastAPI(title="Barents Lice Forecasting API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/viewer", StaticFiles(directory=str(VIEWER_DIR)), name="viewer")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    selected_site_id: str | None = None
    visible_site_ids: list[str] | None = None


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(VIEWER_DIR / "index.html")


@app.get("/api/sites")
def get_sites() -> dict[str, object]:
    return chat_service.get_geojson()


@app.get("/api/health")
def get_health() -> dict[str, object]:
    return {
        "status": "ok",
        "site_count": chat_service.get_site_count(),
        "viewer_dir": str(Path("viewer")),
        "llm": chat_service.get_llm_status(),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    return chat_service.answer_question(
        request.message,
        selected_site_id=request.selected_site_id,
        visible_site_ids=request.visible_site_ids,
    )
