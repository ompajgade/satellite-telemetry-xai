from fastapi import APIRouter
from api.schemas import (
    DetectRequest, DetectResponse,
    DiagnoseRequest, DiagnoseResponse,
    ExplainRequest, ExplainResponse
)
from api.detectors import detect_anomaly
from kg.reasoning import diagnose_root_cause
from rag.retriever import explain_with_rag

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    score, is_anomaly = detect_anomaly(req.features)
    return DetectResponse(anomaly_score=score, is_anomaly=is_anomaly)


@router.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    diagnosis = diagnose_root_cause(req.anomalous_channels)
    return diagnosis


@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    explanation = explain_with_rag(req.diagnosis)
    return ExplainResponse(explanation=explanation)


@router.post("/analyze")
def full_pipeline(req: DiagnoseRequest):
    """
    Demo endpoint: diagnose + explain together
    """
    diagnosis = diagnose_root_cause(req.anomalous_channels)
    explanation = explain_with_rag(diagnosis)

    return {
        "diagnosis": diagnosis,
        "explanation": explanation
    }