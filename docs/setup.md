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

## 반복 학습 — 칼·가위 혼동 (#33, AWS 인스턴스)

E1 과 달라지는 것은 학습 사진 목록 하나다. 총기·칼·가위가 든 사진을 3번씩 보여준다.
결과는 E1 을 덮어쓰지 않도록 이름을 `os` 로 나눠 둔다.

```bash
python3 ml/scripts/make_oversampled_train.py      # data/yolo/images/train_os, pidray_os.yaml
```

출력 첫 줄의 배수만큼 에폭이 길어진다 (E1 2분 45초 × 배수).

```bash
yolo detect train data=data/yolo/pidray_os.yaml model=yolo11s.pt imgsz=640 epochs=50 patience=15 batch=32 workers=4 device=0 project=$PWD/ml/runs/detect name=os exist_ok=True
```

점수 추출은 `predictions_os` 로 따로 쓴다.

```bash
for s in calib test_easy test_hard test_hidden; do python3 ml/scripts/predict_scores.py ml/runs/detect/os/weights/best.pt $s --out ml/runs/predictions_os/$s.csv; done
```

비교는 E1 과 같은 스크립트에 폴더와 가중치만 바꿔 준다.

```bash
for c in 0.05 0.02 0.01; do python3 ml/scripts/e2_compare.py --pred-dir ml/runs/predictions_os --cap $c; done
python3 ml/scripts/analyze_misses.py --weights ml/runs/detect/os/weights/best.pt --pred-dir ml/runs/predictions_os
```

## 발표용 예시 사진과 검출기 세부 성능 (#35, AWS 인스턴스)

```bash
python3 ml/scripts/make_samples.py        # ml/runs/samples — 품목 12종 잘라낸 사진, 난이도별 예시. 커밋하지 않는다
python3 ml/scripts/eval_detector.py       # val · 시험 세 난이도의 전체 · 품목별 정밀도 · 재현율 · mAP
```

예시 사진은 PIDray 재배포 금지 조건 때문에 저장소에 올리지 않는다. 발표 자료에만 원 논문을 인용해 쓴다.

## 라이터·탄약 타일 추론 비교

재학습 없이 원본 전체 사진과 네 모서리의 겹치는 확대 영역을 함께 본다.
타일의 라이터·탄약 박스만 원본 좌표로 복원하고 클래스별 NMS로 병합한다.
다른 10종의 박스는 두 비교군에서 동일한지 코드로 확인한다.
설정은 `ml/configs/target_tiles_20261017.json`에 기록했다.

```bash
source /opt/pytorch/bin/activate
cd ~/hancut-inspection
python -m pytest -q ml/tests/test_tiling.py ml/tests/test_eval_tiled.py
# 새 출력 경로만 사용한다. 기존 경로면 중단한다.
bash ml/scripts/run_tiled_experiment.sh "$PWD/ml/runs/target_tiles_new"
```

장시간 실행은 위 명령을 tmux 안에서 시작한다. 출력은 `pipeline.log`에 기록되므로
학습·실험 세션 화면 대신 별도 터미널에서 다음처럼 조회한다.

```bash
tail -f ~/hancut-inspection/ml/runs/target_tiles_20261017/pipeline.log
cat ~/hancut-inspection/ml/runs/target_tiles_20261017/stage.txt
cat ~/hancut-inspection/ml/runs/target_tiles_20261017/pipeline.exit
```

`tail -f`에서 Ctrl+C는 로그 조회만 끝낸다. 완료 여부는 `stage=complete`와
`pipeline.exit=0`으로 확인한다. 결과는 `inference/<split>/metrics.json`,
운영 지표는 `e2_<baseline|tiled>_<상한>/summary.md`다.
검출 비교는 동일한 정사각 패딩(`rect=False`)과 고정 conf=0.25를 사용하므로,
기존 `val`의 직사각 패딩·자동 선택 운영점과 숫자를 직접 비교하지 않는다.

단순 타일의 오탐을 줄이는 후처리 후보는 **calib만** 사용해서 선택한다.
기존 전체 사진 박스를 복구하고, 첫 타일 병합에서 살아남은 추가 박스의 점수를 조정한다.
원시 타일 박스 전체를 다시 병합한 실험은 아니다. 선택 시점 이후에는 설정을 바꾸지 않는다.

```bash
python ml/scripts/refine_tiled.py select \
  --source ml/runs/target_tiles_20261017/inference \
  --out ml/runs/target_tiles_20261017/frozen_selection.json
python ml/scripts/refine_tiled.py evaluate \
  --source ml/runs/target_tiles_20261017/inference \
  --selection ml/runs/target_tiles_20261017/frozen_selection.json \
  --out ml/runs/target_tiles_20261017/refined
```

`select`는 calib 완료 뒤 실행할 수 있고, `evaluate`는 네 분할 완료 뒤 실행한다.
`refined`에도 기존 E2 CLI를 동일하게 적용할 수 있다. 기존 실험 폴더에 재실행하지 않는다.

## 겹침 MixUp 미세조정 비교 (#38)

E1 가중치에서 같은 설정으로 대조군(mixup 0)과 mixup 0.3을 각각 20에폭 이어 학습한다.
두 쪽 모두 원래 E1 가중치에서 새로 시작하고, 출력 폴더가 이미 있으면 중단한다.
설정은 `ml/configs/overlap_mixup_20261017.json`이다.

```bash
source /opt/pytorch/bin/activate
cd ~/hancut-inspection
python ml/scripts/run_mixup_experiment.py prepare          # 가중치 해시·분할 수·경로 중복 확인
python ml/scripts/run_mixup_experiment.py train --arm smoke  # 1에폭 점검, smoke_result.json 생성
tmux new -s mixup-run
python ml/scripts/run_mixup_experiment.py run > ml/runs/overlap_mixup_20261017/pipeline.log 2>&1
```

`run`은 smoke 결과가 없으면 시작하지 않는다. 학습 두 개 → E1·대조군·mixup 검출 평가 →
점수 추출 → E2 상한 1·2·5%를 순서대로 실행한다. 진행은 `status.json`, 완료는 `pipeline.exit=0`으로 확인한다.

## 세 모델 점수 결합 (#41, AWS 인스턴스)

학습 없이 저장된 예측 CSV 만 쓴다. 세 갈래를 만들고 같은 방식으로 평가한다.

```bash
E1=ml/runs/predictions; M960=ml/runs/lighter_train960_20261016/predictions; MIX=ml/runs/overlap_mixup_20261017/predictions_mixup; OUT=ml/runs/fusion_20261019
python3 ml/scripts/fuse_scores.py --model E1=$E1 --model 960=$M960 --out $OUT/e1_960                                     # 지난 결합 재현 확인
python3 ml/scripts/fuse_scores.py --model E1=$E1 --model 960=$M960 --model mixup=$MIX --out $OUT/all3                      # 세 모델 전부
python3 ml/scripts/fuse_scores.py --model E1=$E1 --model 960=$M960 --model mixup=$MIX --select calib-ap --out $OUT/calib_ap  # 품목별 선택(calib)
for a in e1_960 all3 calib_ap; do for c in 0.01 0.02 0.05; do python3 ml/scripts/e2_compare.py --pred-dir $OUT/$a --cap $c --seed 0 --out-dir $OUT/e2_${a}_$c; done; done
```

`calib-ap` 의 품목별 선택은 학습 분할 calib(2,944장)에서만 정해진다. 시험 결과를 보고 바꾸지 않는다.
