"""FastAPI application entry point for SatQuery AI Backend."""

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent.agent import SatQueryAgent
from backend.agent.registry import TOOL_REGISTRY
from backend.agent.schemas import AgentResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("backend.main")

app = FastAPI(
    title="SatQuery AI API",
    description="Agentic Remote Sensing Multi-Modal Analysis and Query System",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = SatQueryAgent()

# Temp directory for saving uploaded multipart images
UPLOAD_DIR = Path("backend/temp_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=str(UPLOAD_DIR)), name="static")

# Mount Smart Satellite Monitoring and Change Alert System
from backend.monitoring.db import init_db
from backend.monitoring.routes import router as monitoring_router

init_db()
app.include_router(monitoring_router)


class JSONAnalyzeRequest(BaseModel):
    query: str
    images: List[str] = []
    sensor_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.get("/health")
def health_check():
    """Health check endpoint providing available specialist models."""
    return {
        "status": "healthy",
        "service": "SatQuery AI Agent Orchestrator",
        "available_specialists": list(TOOL_REGISTRY.keys()),
    }


@app.get("/api/agent/tools")
def list_tools():
    """Return list and descriptions of registered specialist models."""
    return {
        name: {
            "model_id": tool.model_id,
            "description": tool.description,
            "accepted_inputs": tool.accepted_inputs,
        }
        for name, tool in TOOL_REGISTRY.items()
    }


@app.post("/api/agent/analyze", response_model=AgentResponse)
async def analyze_query_multipart(
    query: str = Form(...),
    images: List[UploadFile] = File(default=[]),
    sensor_type: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
):
    """Analyze query and uploaded images via the SatQuery AI Agent."""
    saved_paths = []
    parsed_metadata = {}

    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except Exception:
            parsed_metadata = {"raw_metadata": metadata}

    try:
        # Save uploaded files temporarily
        for upload in images:
            if upload.filename:
                temp_file = UPLOAD_DIR / upload.filename
                with open(temp_file, "wb") as buffer:
                    content = await upload.read()
                    buffer.write(content)
                saved_paths.append(temp_file)

        # Run analysis through agent orchestrator
        response = agent.analyze(
            query=query,
            images=saved_paths,
            sensor_type=sensor_type,
            metadata=parsed_metadata,
        )
        return response

    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent/analyze/json", response_model=AgentResponse)
def analyze_query_json(request: JSONAnalyzeRequest):
    """JSON-based endpoint when images are provided as file paths on disk."""
    try:
        response = agent.analyze(
            query=request.query,
            images=request.images,
            sensor_type=request.sensor_type,
            metadata=request.metadata,
        )
        return response
    except Exception as e:
        logger.error("JSON analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

