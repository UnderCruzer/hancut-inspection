"""
판정 API — 판독 앱이 보낸 X-ray 사진에서 위해물품 유무를 판정한다.

모델이 로드되지 않았으면 503. 최종 판정은 판독관이 하며, '재검' 구간은
반드시 사람에게 넘어간다.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app import judgment
from app.predictor import Predictor

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

router = APIRouter()


class JudgmentResponse(BaseModel):
    item: str
    score: float
    zone: str
    needs_review: bool
    reasons: list[str]
    thresholds: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    thresholds_loaded: bool


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=getattr(request.app.state, "predictor", None) is not None,
        thresholds_loaded=bool(getattr(request.app.state, "thresholds", None)),
    )


@router.post("/v1/inspections", response_model=JudgmentResponse)
async def judge(request: Request, image: UploadFile = File(...)) -> JudgmentResponse:
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    thresholds = getattr(request.app.state, "thresholds", None)
    if predictor is None or not thresholds:
        raise HTTPException(status_code=503, detail="판정 모델이 로드되지 않았다")

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"지원하지 않는 형식: {image.content_type}")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="빈 파일이다")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"파일이 너무 크다 (최대 {MAX_UPLOAD_BYTES} bytes)")

    prediction = predictor.predict(payload)
    table = judgment.thresholds_for(thresholds, prediction.item)
    zone = judgment.zone_of(prediction.score, table)

    return JudgmentResponse(
        item=prediction.item,
        score=round(prediction.score, 4),
        zone=zone,
        needs_review=judgment.needs_review(zone),
        reasons=prediction.reasons,
        thresholds={"low": table.low, "high": table.high},
        model_version=prediction.model_version,
    )


def create_app(predictor: Predictor | None = None, thresholds_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="한컷점검 판정 API", version="0.1.0")
    app.state.predictor = predictor

    path = thresholds_path or Path(os.getenv("THRESHOLDS_PATH", "models/thresholds.json"))
    app.state.thresholds = judgment.load_thresholds(path) if path.exists() else None

    app.include_router(router)
    return app


app = create_app()
