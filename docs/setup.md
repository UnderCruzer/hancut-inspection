# 개발 환경 준비

## 필요한 것

| 도구 | 버전 | 확인 |
|------|------|------|
| Python | 3.11 | `python3.11 --version` |
| Flutter | 3.41 이상 | `flutter --version` |
| Docker | 선택 (서버 배포용) | `docker --version` |
| gh CLI | 이슈·PR 작업용 | `gh auth status` |

## 한 번만 하는 준비

저장소 루트에 가상환경 하나를 만들어 `ml`과 `server`가 함께 쓴다.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r ml/requirements.txt -r server/requirements.txt
cd app && flutter pub get && cd ..
```

`.venv/`는 `.gitignore`에 있다.

## 테스트

```bash
cd ml     && ../.venv/bin/python -m pytest -q     # 44개
cd server && ../.venv/bin/python -m pytest -q     # 14개
cd app    && flutter analyze && flutter test      # 6개
```

## 서버 실행

```bash
cd server
../.venv/bin/uvicorn app.main:app --reload --port 8000
```

- `GET /health` — 모델·임계값 로드 여부 확인
- `POST /v1/inspections` — 이미지 업로드 → 판정

모델을 아직 붙이지 않았으므로 판정 요청은 **503**을 반환한다. 정상 동작이다.

임계값 파일은 학습 쪽 E2 결과로 만든다. 형식은 `server/thresholds.example.json` 참고.

```bash
cp server/thresholds.example.json server/models/thresholds.json   # models/ 는 gitignore 대상
```

## 앱 실행

```bash
cd app && flutter run
```

촬영 화면은 W9에 붙인다. 지금은 판정 구간이 화면에 어떻게 보이는지만 확인하는 골격이다.

## 데이터

`docs/data.md`를 먼저 읽는다. 원본은 `data/` 아래에 두고 저장소에 올리지 않는다.

```
data/
  pidray.zip       받은 파일 11GB. 풀고 나면 지워도 된다
  pidray/          압축 해제본 (#14)
  index.csv        이미지 x 품목 인덱스 (#6) — 커밋하지 않는다. 26MB, 재생성 가능
  thresholds.json  확정 임계값 (E2) — 커밋한다
```

`/data/` 는 통째로 무시하되 작은 `.csv`·`.json` 은 예외로 커밋한다. 이미지는 재배포 금지다.
`index.csv` 는 예외에서 다시 뺐다 — 26MB 파생 파일이라 커밋하면 저장소가 불어난다.

인덱스는 인자 없이 만든다. 어노테이션 폴더를 알아서 찾는다.

```bash
python3 ml/scripts/build_index.py
```

SIXray 는 받을 수 없어 쓰지 않는다(#7). `docs/data.md` 참조.

## 자주 겪는 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| `No module named pytest` | 가상환경이 아닌 시스템 파이썬 사용 | `../.venv/bin/python -m pytest` 로 실행 |
| `ModuleNotFoundError: hancut` | 잘못된 디렉터리에서 실행 | `ml/`에서 실행 (pyproject의 `pythonpath` 설정) |
| 서버가 계속 503 | 임계값 파일 없음 또는 모델 미탑재 | `/health`로 어느 쪽인지 확인 |
| CI 워크플로 push 거부 | 토큰에 `workflow` 권한 없음 | `gh auth refresh -s workflow` |

## E1 — 검출기 학습 (#23, AWS 인스턴스)

저장소 루트에서, `source /opt/pytorch/bin/activate` 한 상태로 실행한다. 학습은 tmux 안에서 돌린다.

```bash
pip install -q ultralytics
python3 ml/scripts/make_yolo_dataset.py          # data/yolo 생성. 이미지는 링크라 금방 끝난다
```

**먼저 1 에폭만 돌려 시간을 잰다.** 에폭당 시간 × 에폭 수로 본 학습 시간과 비용을 정한 뒤 시작한다.

```bash
yolo detect train data=data/yolo/pidray.yaml model=yolo11s.pt imgsz=640 epochs=1 batch=32 workers=4 device=0 project=$PWD/ml/runs/detect name=e1_smoke exist_ok=True
```

**`project` 는 반드시 절대 경로로 준다.** ultralytics 8.4 는 상대 경로 앞에 `runs/detect` 를 붙여,
`project=ml/runs/detect` 가 `runs/detect/ml/runs/detect` 가 된다(2026-09-28 실측).

1 에폭 실측(A10G, yolo11s, batch 32): **약 2분 45초**, GPU 메모리 7.6GB.

본 학습은 에폭 수와 이름만 바꾼다. 가중치는 `ml/runs/detect/<name>/weights/best.pt` 에 생기고 커밋하지 않는다.

```bash
yolo detect train data=data/yolo/pidray.yaml model=yolo11s.pt imgsz=640 epochs=50 patience=15 batch=32 workers=4 device=0 project=$PWD/ml/runs/detect name=e1 exist_ok=True
```

학습이 끝나면 보정셋과 시험셋 점수를 뽑고 평가 CLI 를 돌린다.

```bash
python3 ml/scripts/predict_scores.py ml/runs/detect/e1/weights/best.pt calib
python3 ml/scripts/predict_scores.py ml/runs/detect/e1/weights/best.pt test_hidden
python3 ml/scripts/evaluate.py ml/runs/predictions/calib.csv --output-dir ml/runs/e2 --test-csv ml/runs/predictions/test_hidden.csv
```

| 폴더 | 무엇 | 쓰임 |
|---|---|---|
| `images/train` | 학습 85% | 가중치 학습 |
| `images/val` | 학습 5% | 학습 중 에폭 선택 |
| `images/calib` | 학습 10% | 임계값 보정. **val 과 섞지 않는다** |
| `images/test_*` | 시험 easy · hard · hidden | 최종 평가. 학습에도 보정에도 쓰지 않는다 |
