# 소방시설 상태 데이터 검토 — 2026-09-24

## 결론과 확인 수준

소방시설 주제를 유지한다. 우선 후보는 Weqaa의 소화기 압력계 상태·부식 주석이다.
VisA/Real-IAD 제품 검사 전환은 사용자의 의도와 달랐으므로 취소한다. 다운로드한 파일을 삭제하지는 않는다.
여러 종류 소방시설의 모든 법적 준수 항목을 판정할 데이터는 확보하지 못했다.

- 공개 페이지의 라이선스, 버전 정보, 실제 이미지 한 장의 Raw Data 주석까지 확인했다.
- 전체 ZIP 내보내기는 로그인 화면에서 멈췄다. 전체 클래스별 건수·용량·품질·학습 가능성은 아직 검증하지 않았다.
- 논문/옛 모델 클래스/프로젝트 전체 클래스/버전별 export 클래스는 서로 구분한다.

## Weqaa

- 프로젝트: https://universe.roboflow.com/cv-project-g2qeh/weqaa
- 현재 공개 최신 버전: https://universe.roboflow.com/cv-project-g2qeh/weqaa/dataset/18
- 프로젝트 이미지 3,592장. v18은 증강 포함 7,664장, 640×640.
- v18 화면 분할: train 6,144 / valid 768 / test 752.
- 전처리: 방향 보정, 검은 여백을 넣어 640×640, 클래스 3개 remap 및 1개 drop, 주석 없는 이미지 필터링.
- 학습 이미지당 출력 3개, 회전·채도·밝기 증강. 7,664장을 독립 촬영 7,664건으로 간주하지 않는다.
- 표시 라이선스 CC BY 4.0. 해당 라이선스는 전 세계 이용을 허용하므로 표시 조건 기준으로 미국 AWS 이용 후보에 적합하다. 출처·라이선스 링크·변경 사항을 기록한다. 실제 export에 별도 조건이 있다면 함께 확인한다.
- 법문: https://creativecommons.org/licenses/by/4.0/legalcode.en (Section 2(a)(1), Section 3)

### 실제로 읽은 주석

https://universe.roboflow.com/cv-project-g2qeh/weqaa/dataset/18/images/acff4a7d8bcccf4f03c1ad556375bd67?split=train

`IMG_0202_MOV-2.jpg`의 Raw Data → Annotation Data에 `gauge_bad` polygon 1개와 좌표가 있다.
이미지 크기는 640×640, source에 원본 연결 ID, preprocessing에 증강 내역이 있다.
라벨 목록에 rust·pin_exist도 노출되지만 전체 export 클래스 확정 증거로 사용하지 않는다.
프로젝트 개요의 8개 옛 모델 클래스와 논문의 6개 클래스를 혼용하면 안 된다.

### 논문이 정의한 라벨과 사용 한계

출처: Alayed et al., Real-Time Inspection of Fire Safety Equipment using Computer Vision and Deep Learning, ETASR 2024.
https://www.etasr.com/index.php/ETASR/article/view/6753
https://doi.org/10.48084/etasr.6753

| 논문 라벨 | 의미 | 우리 서비스에서 허용할 주장 |
|---|---|---|
| gauge_good | 압력계 녹색 범위 | 사진상 압력 표시 정상 후보 |
| gauge_bad | 압력계 적색 범위 | 사진상 압력 이상 의심 |
| rust | 부식 영역 | 부식 의심 위치 |
| pin_exist | 안전핀 존재 | 핀이 보임; 미검출이면 확인 불가 |
| hose_exist | 호스 존재 | 호스가 보임; 미검출이면 확인 불가 |
| expire_date | 날짜 라벨 위치 | 날짜 라벨 위치 표시만 가능 |

압력 실제 수치·작동 여부·호스 내부 막힘·안전핀 누락·유효기간 초과의 정답이 모두 있는 것은 아니다.
expire_date는 OCR 문자열/유효기간 초과 정답이 아니다. 미검출을 결함 없음이나 부품 누락으로 바꾸지 않는다.
논문의 7,663장과 현재 v18의 7,664장은 버전 차이이며 현재 수치를 논문 수치로 덮어쓰지 않는다.

### 데이터 누수 위험 — 화면에서 발견

원본 Browse 목록에서 같은 파일 접두사의 프레임이 train/valid/test에 흩어져 있다.
예: IMG_0202_MOV-2.jpg(train), IMG_0202_MOV-4.jpg(valid), IMG_0202_MOV-9.jpg(test).
이는 같은 영상의 인접 프레임일 가능성이 높다. 논문도 영상 프레임 추출을 설명한다.
공식 split만으로 독립 현장 일반화 성능을 주장하지 않는다. 원본 영상·소화기 개체 단위 그룹을 확인하고 증강 전 분리한다.
개체 ID를 복원할 수 없으면 그 한계를 명시하고 영상 그룹 기반 평가와 중복/근접중복 검사를 한다.

## 다른 후보 비교

| 후보 | 라벨 확인 | 판단 |
|---|---|---|
| FireNet (UCL) | 소화기·감지기·유도 표지 등 8개 장비 종류, 검출/분할; 상태 라벨 확인 안 됨 | 장비 인식 보조용, 주 결함 데이터 대체 불가 |
| Fire-ART | 15개 장비 클래스, 2,626장/6,627개 객체; hidden equipment는 가림이지 고장 정답 아님 | 장비 인식 보조용 |
| AI Hub 518 | 기존 준수/미준수 데이터 | 국외 반출 합의가 없는 현재 미국 AWS에 사용하지 않음 |

FireNet: https://rdr.ucl.ac.uk/articles/dataset/FireNet/9137798 (CC BY-NC 4.0)
Fire-ART: https://github.com/UMRATE/Fire-ART (저장소 CC BY 4.0 표기; FireNet 등 원본 혼합 출처는 각각의 조건 확인)

## 확보 후 완료해야 할 검사

1. v18 export의 클래스 목록/ID/좌표 형식과 라이선스 파일을 확인한다. 공개 UI의 원본 polygon이 선택한 export에서도 보존되는지 확인한다.
2. 클래스별 이미지·주석·원본 영상·독립 소화기 수를 집계한다. gauge_good/gauge_bad/rust 유효 주석을 직접 검수한다.
3. 다운로드/압축 해제 크기를 실측한다. GB 수치를 이미지 개수만으로 확정하지 않는다.
4. 그룹별 split 및 양성·음성 기준을 작성한다. 주석이 없는 영역을 무조건 정상으로 처리하지 않는다.
5. 그룹 분할 후 클래스가 부족하면 실험 범위를 줄인다. 단순 증강으로 독립 표본 부족을 해결했다고 주장하지 않는다.
6. 사전학습 모델 기준선을 학습한 뒤 클래스별 재현율, 놓친 건수/분모, 검토율을 평가한다.
