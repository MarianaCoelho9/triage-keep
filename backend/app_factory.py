import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from voice_agent.config import get_services
from voice_agent.core.logging_utils import log_event

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_event(component="app", event="startup")
    get_services()  # Trigger loading
    yield
    log_event(component="app", event="shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TriageKeep API",
        description="AI-Assisted Triage Support System",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify the frontend origin
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the data directory to serve static files (e.g., intro audio)
    # Ensure the directory exists
    os.makedirs("data", exist_ok=True)
    app.mount("/static", StaticFiles(directory="data"), name="static")

    return app
