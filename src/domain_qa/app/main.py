from __future__ import annotations

from fastapi import FastAPI

from domain_qa.app.api import router as api_router
from domain_qa.ui.ui_router import router as ui_router
from domain_qa.app.logging_config import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Domain QA Chatbot")

    app.include_router(api_router)
    app.include_router(ui_router)

    return app


app = create_app()