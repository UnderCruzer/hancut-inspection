# 한컷점검 앱

보안검색 X-ray 사진을 고르면 판정 서버(`POST /v2/inspections`)로 보내고,
**사진 판정**(품목 하나라도 적발이면 적발, 재검이 있으면 재검, 모두 기준 아래면 통과)과
위해물품 12종의 품목별 점수 · 구간을 보여준다. 재검은 반드시 판독관이 본다.

판정 기준은 E1 검출기 dense 점수 + 설정 F · 놓침 5% 상한 (`ml/configs/dense_e1_f5_seed0_fp32_20261027.json`).
PIDray 시험 분포에서 정한 실험용 기준이며 실제 공항 장비에서 검증하지 않았다.

## 맥에서 시연 (서버 CPU + 앱 Chrome)

1. 추론 환경 (한 번만, 저장소 루트):

   ```bash
   python3 -m venv .venv-demo
   .venv-demo/bin/pip install torch torchvision
   .venv-demo/bin/pip install -r server/requirements-dense.txt
   ```

2. E1 가중치를 `ml/runs/detect/e1_smoke/weights/best.pt` 에 둔다
   (SHA-256 `2d41d3d3…5622754`, 다르면 서버가 503 을 낸다). 커밋하지 않는다.

3. 판정 서버 (저장소 루트):

   ```bash
   DENSE_WEIGHTS_PATH=ml/runs/detect/e1_smoke/weights/best.pt \
   DENSE_BUNDLE_PATH=ml/configs/dense_e1_f5_seed0_fp32_20261027.json \
   DENSE_DEVICE=cpu CORS_ORIGINS=http://localhost:5555 \
   .venv-demo/bin/python -m uvicorn app.main:app --app-dir server --host 127.0.0.1 --port 8000
   ```

   `curl localhost:8000/v2/health` 가 `"model_loaded":true` 면 준비된 것이다.

4. 앱 (`app/`):

   ```bash
   flutter run -d chrome --web-port 5555
   ```

   `CORS_ORIGINS` 와 앱 주소(포트)가 같아야 브라우저가 응답을 받는다.
   서버 주소는 화면에서 바꾸거나 `--dart-define=HANCUT_API=http://…` 로 정한다.

시연 사진은 `demo/xray/` 에 PIDray 시험 사진을 로컬로만 둔다 (학술 한정 · 재배포 금지라 커밋하지 않는다).
맥 CPU 점수는 검증한 GPU · fp32 조건과 아주 조금 다를 수 있다.

## 개발

```bash
flutter analyze
flutter test
```
