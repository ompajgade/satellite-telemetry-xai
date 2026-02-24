from pydantic import BaseModel
from typing import List, Dict, Any


class DetectRequest(BaseModel):
    features: List[float]  # 18 features (dataset.csv style)


class DetectResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool


class DiagnoseRequest(BaseModel):
    anomalous_channels: List[str]


class DiagnoseResponse(BaseModel):
    affected_components: List[str]
    affected_subsystems: List[str]
    ranked_root_causes: List[List[Any]]


class ExplainRequest(BaseModel):
    diagnosis: Dict[str, Any]


class ExplainResponse(BaseModel):
    explanation: str