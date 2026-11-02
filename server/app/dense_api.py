"""12개 품목의 실험용 판정 API. dense 점수는 보정된 존재 확률이 아니다."""
import io
import logging
import warnings

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app import judgment
from app.dense import SCORE_METHOD

router = APIRouter(prefix="/v2")
logger = logging.getLogger(__name__)
MAX_IMAGE_PIXELS = 20_000_000


class ItemResult(BaseModel):
    item: str
    score: float
    zone: str
    needs_review: bool
    thresholds: dict[str, float]


class DenseResponse(BaseModel):
    items: list[ItemResult]
    score_method: str = SCORE_METHOD
    model_version: str
    bundle_sha256: str
    experimental: bool = True


@router.get("/health")
def health(request: Request):
    return {"model_loaded": request.app.state.dense_predictor is not None, "experimental": True}


@router.post("/inspections", response_model=DenseResponse)
def inspect(request: Request, image: UploadFile = File(...)):
    # sync 경로는 FastAPI의 thread pool에서 실행된다. GPU 호출로 이벤트 루프를 막지 않는다.
    predictor = request.app.state.dense_predictor
    if predictor is None:
        raise HTTPException(503, "dense 판정 모델이 로드되지 않았다")
    formats = {"image/jpeg": "JPEG", "image/png": "PNG"}
    if image.content_type not in formats:
        raise HTTPException(415, "JPEG 또는 PNG가 필요하다")
    limit = request.app.state.max_upload_bytes
    payload = image.file.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(413, "파일이 너무 크다")
    if not payload:
        raise HTTPException(400, "빈 파일이다")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as decoded:
                if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "이미지 픽셀 수가 너무 많다")
                if decoded.format != formats[image.content_type]:
                    raise HTTPException(415, "이미지 내용과 MIME 형식이 다르다")
                decoded.verify()
            # verify()가 헤더만 확인하는 JPEG도 전체 디코딩이 가능해야 한다.
            with Image.open(io.BytesIO(payload)) as decoded:
                decoded.load()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, "이미지 픽셀 수가 너무 많다") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(400, "손상되었거나 읽을 수 없는 이미지다") from None

    try:
        scores = predictor.predict(payload)
        bundle = predictor.bundle
        if set(scores) != set(bundle.names):
            raise ValueError("dense 출력 품목 누락 또는 추가")
        items = []
        for name in bundle.names:
            table = bundle.thresholds[name]
            score = float(scores[name])
            zone = judgment.zone_of(score, table)
            items.append(ItemResult(item=name, score=score, zone=zone,
                                    needs_review=judgment.needs_review(zone),
                                    thresholds={"low": table.low, "high": table.high}))
        return DenseResponse(items=items, model_version=f"sha256:{bundle.weights_sha256}",
                             bundle_sha256=bundle.bundle_sha256)
    except Exception:
        logger.exception("dense 추론 실패")
        raise HTTPException(503, "dense 판정을 완료하지 못했다") from None
