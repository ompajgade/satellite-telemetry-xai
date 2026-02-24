from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Explainable Satellite Telemetry Fault Diagnosis API",
    description="FCNN + Knowledge Graph + RAG-based explanation",
    version="1.0"
)

app.include_router(router)