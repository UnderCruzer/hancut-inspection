"""실험용 dense 추론. 오프라인 점수와 같은 전처리/가중치/임계값만 사용한다."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.judgment import Thresholds

SCORE_METHOD = "yolo_class_max_pre_nms"
PREPROCESS = "letterbox_square_bgr_to_rgb_float32_v1"
ULTRALYTICS_VERSION = "8.4.164"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class DenseBundle:
    names: tuple[str, ...]
    weights_sha256: str
    thresholds: dict[str, Thresholds]
    bundle_sha256: str
    imgsz: int = 640

    @classmethod
    def load(cls, path: Path) -> "DenseBundle":
        content = path.read_bytes()
        raw = json.loads(content)
        expected = {"schema_version": 1, "score_method": SCORE_METHOD,
                    "preprocess": PREPROCESS, "ultralytics_version": ULTRALYTICS_VERSION,
                    "imgsz": 640, "purpose": "exploratory", "batch_size": 1,
                    "cudnn_allow_tf32": False, "cuda_matmul_allow_tf32": False}
        for key, value in expected.items():
            if raw.get(key) != value:
                raise ValueError(f"dense bundle {key} 불일치")
        names = raw["class_names"]
        if (not isinstance(names, list) or len(names) != 12
                or any(not isinstance(n, str) or not n or n == "default" for n in names)
                or len(set(names)) != 12):
            raise ValueError("서로 다른 12개 품목이 필요하다")
        digest = raw["weights_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("가중치 SHA256이 필요하다")
        table = raw["thresholds"]
        if set(table) != {*names, "default"}:
            raise ValueError("모든 품목의 임계값이 필요하다")
        thresholds = {name: Thresholds(float(t["low"]), float(t["high"]), t.get("miss_rate_cap"))
                      for name, t in table.items()}
        return cls(tuple(names), digest, thresholds, hashlib.sha256(content).hexdigest())


class DensePredictor:
    """신뢰하는 로컬 체크포인트만 전달한다. GPU 라이브러리는 설정 시에만 로드한다."""

    def __init__(self, weights: Path, bundle: DenseBundle, device: str = "cpu"):
        if sha256(weights) != bundle.weights_sha256:
            raise ValueError("가중치와 dense 임계값의 SHA256이 다르다")
        import ultralytics
        if ultralytics.__version__ != ULTRALYTICS_VERSION:
            raise ValueError("오프라인 실험과 ultralytics 버전이 다르다")
        import torch
        # 배치 크기/TF32에 따른 오차가 보정 경계를 바꾸지 않도록 오프라인과 통일한다.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        from ultralytics import YOLO
        from ultralytics.data.augment import LetterBox

        model = YOLO(str(weights))
        if tuple(model.names[i] for i in sorted(model.names)) != bundle.names:
            raise ValueError("가중치 클래스 순서가 임계값과 다르다")
        self.bundle = bundle
        self.device = torch.device(device)
        self.net = model.model.to(self.device).float().eval()
        self.letterbox = LetterBox(new_shape=(bundle.imgsz, bundle.imgsz), auto=False, stride=32)
        self.lock = Lock()

    def predict(self, image_bytes: bytes) -> dict[str, float]:
        import cv2
        import numpy as np
        import torch

        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("이미지를 디코딩하지 못했다")
        img = self.letterbox(image=img)
        chw = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))
        with self.lock, torch.no_grad():
            x = torch.from_numpy(chw).unsqueeze(0).to(self.device).float() / 255
            raw = self.net(x)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            if raw.ndim != 3 or raw.shape[:2] != (1, 4 + len(self.bundle.names)) or raw.shape[2] != 8400:
                raise ValueError("지원하지 않는 YOLO 출력 형태")
            values = raw[:, 4:, :]
            if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError("잘못된 dense 점수")
            maxima = values.amax(dim=2).cpu().tolist()[0]
        # 보정 CSV의 .9g 표현과 같은 경계값을 사용한다. 1e-8 같은 값도 보존된다.
        return {name: float(format(score, ".9g")) for name, score in zip(self.bundle.names, maxima)}
