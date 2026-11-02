# 판정 API

기존 `/v1/inspections`는 단일 품목 predictor 주입 방식이다. 실험용
`/v2/inspections`는 E1 모델의 NMS 이전 앵커별 클래스 점수 최댓값으로
12개 품목을 모두 반환한다. 작은 점수를 0으로 반올림하지 않는다.
점수는 보정된 존재 확률이 아니며, 박스 위치를 제공하지 않는다.

## Dense 실험 경로

2026-10-27 A10G에서 18,220장 재추출과 보정, 108장·1,296개 품목 API 비교를 완료했다.
HTTP 실패·점수 허용 오차 초과·판정 구간 차이 모두 0건이고, 별도 Uvicorn 서버에서도
easy/hard/hidden 각 1장(총 36개 품목)의 점수·판정이 정확히 일치했다.
서버 테스트 47개, GPU 환경 ML 테스트 224개 통과. **공개 서비스 배포와 앱 화면 연결은 하지 않았다.**
실측 비교 결과는 `../docs/results.md`의 2026-10-27 항목, 불일치 원인과 수정은
[`../docs/dense-api-validation.md`](../docs/dense-api-validation.md)에 있다.
해당 결과는 반복 사용한 PIDray 시험 분포를 재분할한 탐색 결과이며 독립 신규 시험이나
서비스 놓침률 보장이 아니다. API도 `experimental: true`로 이를 표시한다.

검증한 임계값은 `ml/configs/dense_e1_f5_seed0_fp32_20261027.json`에 보관한다.
이 bundle은 지정된 E1 가중치와 batch 1·TF32 비활성화 조건 전용이다.

GPU 환경에서 저장소 루트 기준으로 실행한다. 기존 환경의 CUDA용 PyTorch를 유지하고,
추론 의존성은 `server/requirements-dense.txt`를 사용한다(Ultralytics 8.4.164 고정).
아래 명령은 추가 학습 없이 기존 가중치로 예측·임계값을 재생성한다.
출력 폴더는 아직 존재하지 않아야 한다.

```bash
python ml/scripts/predict_dense_scores.py ml/runs/detect/e1_smoke/weights/best.pt \
  --out-dir ml/runs/dense_api_candidate --splits test_easy test_hard test_hidden --device 0 --batch 1 --ieee-fp32
python ml/scripts/export_dense_bundle.py ml/runs/dense_api_candidate/dense \
  --seed 0 --out ml/runs/dense_api_bundle
```

이전 CSV에는 출처 manifest가 없으므로 그대로 서버 임계값으로 내보내지 않는다.
새 추출은 가중치 SHA256, 클래스 순서, 전처리, 해상도, 라이브러리 버전, 배치 크기, TF32 설정 및 CSV 해시를 기록한다.
API용 임계값은 batch=1, cuDNN/행렬곱 TF32 비활성화 조건에서만 내보내고 로드한다.
내보내기는 이미지 단위로 50:50 보정/평가를 분리하고 설정 F(δ=0.05/36)의 5% 임계값을 선택한다.
동시에 `e2.json`에 5/2/1% 상한별 놓침·재검·오경보·자동 처리율과
전체/품목/품목×난이도별 분모를 저장한다. `n`은 이미지×품목 수,
`n_threat`는 양성 수, `n_non_threat`는 음성 수다. 기존 실험과 동일한 seed를 사용해
재현성을 확인해야 하며, 이 명령을 실행했다는 것만으로 실측 상한 충족을 뜻하지 않는다.

실제 이미지로 오프라인 점수 및 API 판정의 일치를 확인한다(새 출력 JSON 경로 사용).

```bash
python server/scripts/check_dense_parity.py \
  --weights ml/runs/detect/e1_smoke/weights/best.pt \
  --bundle ml/runs/dense_api_bundle/dense-bundle.json \
  --pred-dir ml/runs/dense_api_candidate/dense \
  --images data/yolo/images --device cuda:0 --seed 0 --per-cell 3 \
  --out ml/runs/dense_api_bundle/api-parity.json
```

점수 오차 및 판정 불일치가 있으면 출력 보고서로 확인한다. 보정과 API는 batch=1 및 TF32 비활성화를
공유한다. 배치 크기/수치 설정이 다른 기존 실험의 임계값을 그대로 로드하지 않는다.
PIDray hidden은 `.png` 파일명이어도 실제 JPEG이므로 비교 도구는 내용을 확인해 MIME을 정한다.
이 검사는 품목×난이도별 양성 사진을 샘플링한 연결 검증이며 새로운 성능 평가가 아니다.

확인 후 로컬 실험 서버를 띄우려면 저장소 루트에서 실행한다. `.env.example`은 자동 로드되지 않는다.

```bash
export DENSE_WEIGHTS_PATH="$PWD/ml/runs/detect/e1_smoke/weights/best.pt"
export DENSE_BUNDLE_PATH="$PWD/ml/runs/dense_api_bundle/dense-bundle.json"
export DENSE_DEVICE=cuda:0
python -m uvicorn app.main:app --app-dir server --host 127.0.0.1 --port 8000
```

`GET /v2/health`에서 `model_loaded`를 확인한 뒤 JPEG/PNG를
`POST /v2/inspections`의 multipart `image` 필드로 보낸다.
응답 `items`에는 품목별 `score`, `zone`, `needs_review`, `thresholds`가 들어간다.
응답의 `model_version`과 `bundle_sha256`으로 사용한 가중치·임계값을 식별한다.
가중치/임계값/클래스 순서/버전이 맞지 않거나 추론이 실패하면 503을 반환한다.
기존 `/health`는 v1 상태이고 v2 상태는 별도 경로에서 확인한다.

## 로컬 테스트

```bash
cd server
../.venv/bin/python -m pytest -q
```

GPU 패키지 없이도 API·업로드 검증·실패 처리·artifact 계약 테스트를 실행할 수 있다.
가중치와 PIDray 이미지, 예측 데이터는 커밋하지 않는다.
