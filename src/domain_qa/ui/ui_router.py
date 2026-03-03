from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    """
    Serves the chat page.
    """
    index_path = TEMPLATES_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")