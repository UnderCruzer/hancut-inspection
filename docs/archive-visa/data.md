# 현재 데이터 — VisA

2026-09-24 전환. AI Hub 518은 현재 학습에 사용하지 않는다.

- Amazon VisA: 12종 10,821장 (정상9,621 / 이상1,200), 이미지 라벨과 픽셀 마스크.
- 1차 대상 pcb1: 정상1,004 / 이상100 (전체 수량이며 학습셋 수량 아님).
- 데이터 CC BY 4.0. 출처·라이선스 링크·변경 사항 표기. 코드의 Apache-2.0과 구분.
- 공식 파일 HEAD 확인: 1,929,840,640 bytes, 약1.93GB / 1.80GiB. 전체 12종 TAR이다.
- 해제 후 용량은 미실측. 공간 확인 후 다운로드한다. AI Hub의 filekey/API key는 불필요.

## AWS Ubuntu 터미널

```bash
mkdir -p ~/hancut-data/visa
cd ~/hancut-data/visa
df -h .
curl -fL --retry 3 -C - -o VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
```

다운로드 완료 후 확인:

```bash
stat -c %s VisA_20220922.tar
tar -tf VisA_20220922.tar | head -30
```

예상 바이트 수와 일치 여부 확인 후 별도 폴더에 해제한다. 크기 일치는 암호학적 무결성 검증은 아니다.
공식 TAR 내부 경로를 확인하기 전 시설별 폴더명을 추정해 이동하지 않는다.
마스크·CSV·분할을 실제 확인한 후 로더/학습 명령을 작성한다. 아직 다운로드/학습 실행한 것은 아니다.

## 출처 및 발표 표기

Amazon Science, VisA; Zou et al., SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation (ECCV 2022).
변경 사항(리사이즈, 자체 분할 등)을 실험별로 함께 기록한다.

- https://github.com/amazon-science/spot-diff
- https://registry.opendata.aws/visa/
- https://creativecommons.org/licenses/by/4.0/

AI Hub 이용 기록과 이전 용량은 archive-aihub/data.md 및 data-access-check.md에 남긴다.
