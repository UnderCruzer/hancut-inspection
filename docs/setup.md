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
  pidray/     PIDray 원본 47,677장 (#14)
  sixray/     SIXray 층화 서브셋 (#7)
  index.csv   라벨에서 만든 인덱스 (#6)
  subset.csv  층화 추출 결과 (#7) — 이 파일만 커밋한다
```

## 자주 겪는 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| `No module named pytest` | 가상환경이 아닌 시스템 파이썬 사용 | `../.venv/bin/python -m pytest` 로 실행 |
| `ModuleNotFoundError: hancut` | 잘못된 디렉터리에서 실행 | `ml/`에서 실행 (pyproject의 `pythonpath` 설정) |
| 서버가 계속 503 | 임계값 파일 없음 또는 모델 미탑재 | `/health`로 어느 쪽인지 확인 |
| CI 워크플로 push 거부 | 토큰에 `workflow` 권한 없음 | `gh auth refresh -s workflow` |
