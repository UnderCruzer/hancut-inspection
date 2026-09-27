# 한컷점검 — Windows 새 채팅 인수인계

작성: 2026-09-22. 맥의 대화와 작업 맥락을 Windows에서 이어가기 위한 문서다.
채팅 원문이나 앱 세션을 복제한 것은 아니다. 아래는 사용자와 합의한 작업 및 확인된 상태다.

## 새 채팅에서 할 일

이 문서와 AGENTS.md, CONTRIBUTING.md를 읽고 기존 프로젝트를 이어간다.
주제를 다시 제안하지 않는다. 사용자는 1인 AI 캡스톤을 준비하며, Windows는 AWS 접속·개발에 사용하고 실제 학습은 AWS에서 한다.
먼저 현재 저장소/PR 및 AWS 리전을 확인한다. 데이터 확보 상태를 추정하지 말고 파일로 확인한다.
다음 개발 우선순위는 #5 라벨 구조 문서화와 #6 라벨 파서 구현이다.

## 저장소 가져오기

비공개 저장소이므로 UnderCruzer 저장소에 접근 가능한 GitHub 계정으로 인증해야 한다.
Git이 설치된 Windows 터미널에서 다음을 실행한다. GitHub CLI가 있다면 먼저 gh auth login을 사용할 수 있다.

```powershell
git clone --branch feat/issue-9-evaluation-cli https://github.com/UnderCruzer/hancut-inspection.git
cd hancut-inspection
```

이미 복제했다면 작업 중인 변경을 보존하고 해당 브랜치로 이동한다. 맥의 .venv는 복사하지 않고 Windows 또는 AWS에서 새로 구성한다.
브랜치가 향후 삭제되었다면 PR #11 병합 여부를 확인하고 main을 사용한다.

- 저장소: https://github.com/UnderCruzer/hancut-inspection
- 작업 브랜치: feat/issue-9-evaluation-cli
- PR: https://github.com/UnderCruzer/hancut-inspection/pull/11
- 인수인계 작성 전 확인: PR OPEN, ML/Server/App CI 모두 성공. 아직 병합하지 않았다.

## 주제 및 서비스

한컷점검: 소방시설 사진에서 시각적 상태를 분류하고, 사람이 확인한 결과를 점검 기록으로 연결하는 AX 서비스.
사진만으로 실제 작동 여부나 법적 적합성을 보장하지 않는다.
핵심은 정확도 하나가 아니라 놓침률·확인 필요 비율·오경보율을 함께 평가하는 3구간 판정이다.
규약: y_true=1은 미준수, score=p(미준수).

## 구현 완료 / 미완료

- 완료: ML 서브셋 추출·이미지 축소·E2 지표, FastAPI/Flutter 골격.
- #9 완료 구현: ml/scripts/evaluate.py. CSV → report.md, metrics.json, thresholds.json.
- 검증 CSV로 임계값 선택, 별도 시험 CSV에는 고정 임계값 적용. 입력 오류·중복 ID·분할 누수·상한 미충족 등을 검사한다.
- 서버 호환 default/시설별 임계값 저장, 반올림하지 않는다.
- 로컬 테스트 ML 66, 서버 14, Flutter 6 = 86개 및 flutter analyze 통과.
- 미완료: 실제 라벨 파서, 실제 모델 학습/추론 연결, 학습용 데이터 변환.
- 실제 모델 성능 측정은 전혀 하지 않았다. 합성 예시를 성능으로 제시하지 않는다.
- 사용법: docs/experiment-protocol.md, 데이터 조건: docs/data-access-check.md.

## 데이터: 다운로드 파일까지 확인한 최신 결과

AI Hub 「다중밀집시설 및 주거시설 화재 안전 데이터」 dataSetSn=518.
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=518

맥에서 아래 파일을 실제 읽었다. 이 파일들은 Git에 넣지 않았으므로 Windows에 자동으로 오지 않는다.

1. Downloads/New_Sample: 이미지 390장 + JSON 390개, 약 543MB. 전부 가스누설경보기, 이미지 category.compliance는 모두 준수. 사진/JSON 짝 누락 및 JSON 읽기 오류 없음. 종류 박스 중 시각경보장치로 기록된 사례 1건 존재.
2. Downloads/115.다중밀집시설 및 주거시설 화재 안전 데이터/01.데이터/2.Validation/라벨링데이터: VL1~VL25.zip 모두 확보. ZIP 안의 JSON을 실제 읽어 아래 매핑 확인. 전체 JSON 읽기 오류 없음.
3. 본 Training 이미지/라벨 및 Validation 이미지의 다운로드 여부는 미확인이다. 샘플/검증 라벨을 받았다고 전체 승인을 추정하지 않는다.

| 번호 | category_s 시설 | 검증 JSON 개수 |
|---|---|---:|
| 5 | 방화문 | 2299 |
| 6 | 비상조명등 | 2169 |
| 8 | 소형소화기 | 2812 |
| 11 | 스프링클러헤드 | 2090 |
| 14 | 연기감지기 | 2492 |
| 15 | 열감지기 | 2492 |
| 20 | 통로유도등 | 2292 |
| 22 | 피난구유도등 | 2292 |

각 목표 시설 검증 라벨에 준수·미준수 모두 있다. VL8은 준수1420/미준수1392.
VL3은 category_m/category_s가 비어 있는 JSON 15000개(준수12000/미준수3000). 추가 해석 전 임의로 시설을 지정하지 않는다.

### 먼저 소형소화기 8번부터

| 파일 | 용도 | 사이트 표시 크기 | filekey |
|---|---|---|---|
| TS8.zip | 학습 이미지 | 47.53GB | 52956 |
| TL8.zip | 학습 라벨 | 31.18MB | 52931 |
| VS8.zip | 검증 이미지 | 5.96GB | 53006 |
| VL8.zip | 검증 라벨 | 3.90MB | 52981 |

합계 약53.53GB. VL8은 맥에 이미 있다. 실제 Training 내부 시설은 TL8을 받아 대조해야 한다.
8종 전체 TS+VS 사진 묶음 합계는 화면 기준 약344.4GB(라벨 별도).
이전의 126GB는 비례 추정이라 다운로드 산정에 사용하지 않는다. 8만 장/640px/약6GB 역시 목표·추정이며 실측 아님.
압축해제를 고려해 선택 다운로드의 2~3배 여유 공간을 확보하라는 AI Hub 안내를 적용한다.

### 라벨 구조에서 이미 확인한 것

- image: id, file_name, width/height 등. 숫자가 문자열인 필드가 있다.
- category: category_m, category_s, compliance, compliance_reason.
- labeled_data: BOX 좌표(TL_X/TL_Y/TR_X 등), left/top/width/height, class_id/class_no, class_nm_ko, class_dept, yolo_txt.
- 샘플에는 시설 종류(BBOX_B_9)와 상태(BBOX_A_22) 박스가 별도로 있다. 둘을 독립된 실제 객체 둘로 세지 않는다.
- 이미지 단위 compliance와 박스 상태를 모두 검토한다. 다중 객체, 혼합 상태, 클래스 충돌 규칙을 정한 뒤 파서를 작성한다.
- 실제 현장/연출 여부는 이미지 인상만으로 확정하지 않는다. 구축 문서 확인이 필요하다.

## AWS 확인 상태

사용자가 브라우저 터미널에서 직접 실행한 화면으로 확인했다. 맥 로컬 학습 환경이 아니다.

- Ubuntu Deep Learning AMI, 사전 환경 /opt/pytorch.
- NVIDIA A10G, GPU 메모리 23028MiB.
- 루트 디스크 193G, 사용14G, 여유180G (해당 시점).
- PyTorch 2.13.0+cu130, torch.cuda.is_available() = True.
- 최초 import 도중 Ctrl+C 중단 후 다시 실행하여 성공했다. 환경 재설치 불필요.
- 접속 사용자 ubuntu. 새 접속마다 source /opt/pytorch/bin/activate.
- 현재 인스턴스 ID 및 리전은 아직 확인 못했다. 과거 IAM 오류에는 us-east-1이 있었다. 현재 리전을 추정하지 않는다.
- 과거 보안그룹 변경은 ControlOnlyOwnResources 정책 explicit deny. 이름만으로 정확한 조건을 단정하거나 인스턴스를 삭제하지 않는다.
- Windows에서 같은 AWS 계정/리전의 기존 인스턴스에 접속하면 서버 파일은 이어서 사용한다. 현재 실행/중지 상태는 미확인이다.

## 데이터 이용 조건 — 다운로드 전에 확인

공식 정책: https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105
국외 반출에는 수행기관/NIA 별도 합의 필요. 미국 리전으로 무조건 직접 다운로드하라는 과거 안내는 정정했다.
국내 AWS 비공개 EC2/EBS 이용의 구체 조건, 발표 원본 이미지 및 ID 목록 공개 범위는 아직 미확인이다.
#2와 docs/data-access-check.md에 남겼다. 외부 문의는 발송하지 않았다.
자격증명/API key/.pem, 원본·라벨·가중치는 Git에 올리지 않는다.
저장소 AGENTS.md의 ID 목록 커밋 지침도 공개 허용 범위 확인 후 적용한다.

## 이슈 상태 및 다음 작업

- #1: 공식 경로 확인, 1인 작업 방식 수정. 이후 샘플/검증 라벨 확보가 이 대화에서 확인됨. 개인 승인 화면은 미확인.
- #2: 공식 약관 조사 기록, 위 미확인 조건 때문에 OPEN.
- #5: 위 실제 라벨 조사 결과를 문서/이슈에 반영하고 평가 단위·코드 매핑을 확정.
- #6: 확인된 스키마로 파서 구현. 깨진/충돌 라벨 개수 보고, 합성 테스트 사용.
- #7/#8: Training 확보 후 서브셋/축소 실행 및 용량 실측.
- #9: PR #11 구현 및 테스트 완료, OPEN PR 상태.
- #10: 후반에 OCR 또는 점검표 초안 중 하나만 선택. 1인 범위를 유지.

PPT 3개는 맥 작업 폴더에 있다(캡스톤 발표, 1주차 진행보고, 데이터셋). Git에 포함됐다고 가정하지 않는다.
meeting-automation의 예전 수정은 임시 폴더 소실/미푸시로 전해졌으므로 구현 완료로 간주하지 않는다.

## Windows 새 채팅 시작 문장

이 저장소의 docs/windows-handoff.md와 AGENTS.md를 읽고 맥에서 진행하던 한컷점검 1인 AI 캡스톤을 이어서 진행해줘. 주제를 다시 정하지 말고, PR #11과 현재 파일 상태부터 확인해줘. 다음은 검증 라벨 조사 결과를 #5에 반영하고 #6 파서를 구현하는 일이야. 학습은 AWS에서 하며 다운로드 전 현재 리전과 데이터 이용 조건을 확인해야 해. 원본 데이터와 비밀키는 Git에 올리지 마.
