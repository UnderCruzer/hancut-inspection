"""CSV 기반 E2 보정 및 고정 임계값 시험. 실측 성능은 별도 시험셋으로 보고한다."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from . import zones

FIELDS = ("image_id", "item", "y_true", "score")


def read_predictions(path: Path) -> list[dict]:
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames) or not set(FIELDS) <= set(reader.fieldnames):
            raise ValueError(f"{path}: 필수 열 {FIELDS}, 중복 없는 헤더가 필요합니다")
        for row in reader:
            line = reader.line_num
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{line}: 열 개수 불일치")
            row = {key: value.strip() for key, value in row.items()}
            if not row["image_id"] or not row["item"] or row["item"] == "default":
                raise ValueError(f"{path}:{line}: 빈 ID/품목 또는 예약 품목명 default")
            # 품목 단위 이진 정의라 사진 한 장이 품목 수만큼 행을 가진다. 고유 키는 (사진, 품목)이다.
            key = (row["image_id"], row["item"])
            if key in seen:
                raise ValueError(f"{path}:{line}: 중복 (image_id, item) {key}")
            seen.add(key)
            if row["y_true"] not in ("0", "1"):
                raise ValueError(f"{path}:{line}: y_true는 0 또는 1")
            try:
                score = float(row["score"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: score는 숫자여야 합니다") from exc
            if not 0 <= score <= 1:
                raise ValueError(f"{path}:{line}: score는 유한한 0..1 값이어야 합니다")
            rows.append({**row, "y_true": int(row["y_true"]), "score": score})
    if not rows:
        raise ValueError(f"{path}: 데이터가 비었습니다")
    return rows


def vectors(rows):
    return [r["y_true"] for r in rows], [r["score"] for r in rows]


def groups(rows):
    return {name: [r for r in rows if r["item"] == name]
            for name in sorted({r["item"] for r in rows})}


def rate(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("비율은 유한한 0..1 값이어야 합니다")
    return number


def run(args):
    calibration = read_predictions(args.csv)
    grouped = groups(calibration)
    for name, rows in {"default": calibration, **grouped}.items():
        if {r["y_true"] for r in rows} != {0, 1}:
            raise ValueError(f"{name}: 임계값 보정에는 품목 있음/없음 표본이 모두 필요합니다")
    caps = sorted(set([*args.miss_caps, args.select_cap]))
    ys, scores = vectors(calibration)
    items = [r["item"] for r in calibration]
    overall = zones.sweep(ys, scores, caps, args.false_alarm_cap)
    by_item = zones.sweep_by_item(items, ys, scores, caps, args.false_alarm_cap)
    table = {}
    for name, rows in {"default": calibration, **grouped}.items():
        y, s = vectors(rows)
        threshold = zones.fit_thresholds(y, s, args.select_cap, args.false_alarm_cap)
        point = zones.evaluate(y, s, threshold)
        # high<=1 규약에서 score=1인 음성은 자동 적발로 판정된다.
        if point.miss_rate > args.select_cap or point.false_alarm_rate > args.false_alarm_cap:
            raise ValueError(f"{name}: 선택한 상한을 만족하지 못해 임계값 내보내기를 중단합니다")
        # 표의 반올림된 low/high를 재사용하면 경계 판정이 달라진다.
        table[name] = {"low": threshold.low, "high": threshold.high, "miss_rate_cap": args.select_cap}
    report = ["# E2 임계값 보정 결과", "", "검증셋에서 임계값을 선택한 결과입니다. 독립 시험 성능이 아닙니다.",
              "y_true=1: 품목 있음, score=p(품목 있음). 상한은 이 표본에서의 경험적 비율이며 실서비스 보장이 아닙니다.",
              "", f"## 전체 (n={len(calibration)})", "", zones.format_table(overall)]
    for name, rows in grouped.items():
        report.extend(["", f"## 품목: {name} (n={len(rows)})", "", zones.format_table(by_item[name])])
    metrics = {"calibration": {"overall": overall, "by_item": by_item},
               "selected_miss_rate_cap": args.select_cap, "false_alarm_cap": args.false_alarm_cap,
               "calibration_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest()}
    if args.test_csv:
        test = read_predictions(args.test_csv)
        if {r["image_id"] for r in test} & {r["image_id"] for r in calibration}:
            raise ValueError("검증셋/시험셋 image_id 중복: 분할 누수")
        test_points = {}
        for name, rows in {"전체": test, **groups(test)}.items():
            # 전체도 품목별 임계값을 적용한다. 미등록 품목은 default로 평가한다.
            decisions = [zones.zone_of(r["score"], zones.Thresholds(**{k: table.get(r["item"], table["default"])[k] for k in ("low", "high")})) for r in rows]
            n_pos = sum(r["y_true"] for r in rows)
            n_neg = len(rows) - n_pos
            review = decisions.count(zones.REVIEW) / len(rows)
            point = {"n": len(rows), "n_threat": n_pos,
                     "miss_rate": sum(r["y_true"] == 1 and d == zones.AUTO_CLEAR for r, d in zip(rows, decisions)) / n_pos if n_pos else None,
                     "false_alarm_rate": sum(r["y_true"] == 0 and d == zones.AUTO_ALARM for r, d in zip(rows, decisions)) / n_neg if n_neg else None,
                     "review_rate": review, "auto_rate": 1 - review}
            test_points[name] = point
        metrics["test"] = test_points
        metrics["test_sha256"] = hashlib.sha256(args.test_csv.read_bytes()).hexdigest()
        metrics["test_default_items"] = sorted(set(groups(test)) - set(grouped))
        report.extend(["", "## 독립 시험셋 (품목별 고정 임계값, 미등록 품목은 default)", "",
                       "시험셋에서는 임계값을 다시 맞추지 않습니다. null은 해당 정답 클래스가 없어 계산 불가입니다.",
                       "```json", json.dumps(test_points, ensure_ascii=False, indent=2), "```"])
    else:
        report.extend(["", "독립 시험셋 미제공: 모델의 일반화 성능은 아직 평가하지 않았습니다."])
    output = args.output_dir
    targets = [output / name for name in ("report.md", "metrics.json", "thresholds.json")]
    if any(p.exists() for p in targets):
        raise ValueError("출력 파일이 이미 있습니다. 새 output-dir을 선택하세요")
    if any(p.resolve() in {args.csv.resolve(), args.test_csv.resolve() if args.test_csv else None} for p in targets):
        raise ValueError("입력 파일을 덮어쓸 수 없습니다")
    output.mkdir(parents=True, exist_ok=True)
    targets[0].write_text("\n".join(report) + "\n", encoding="utf-8")
    for path, value in ((targets[1], metrics), (targets[2], table)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="검증셋 예측 CSV (학습/시험셋 사용 금지)")
    parser.add_argument("--output-dir", type=Path, required=True)
    # D12: 측정 가능한 범위에서 정한 격자. ml/configs/items.json 의 miss_rate_caps.primary 와 같아야 한다.
    parser.add_argument("--miss-caps", nargs="+", type=rate, default=[0.01, 0.02, 0.05])
    parser.add_argument("--select-cap", type=rate, default=0.01)
    parser.add_argument("--false-alarm-cap", type=rate, default=0.05)
    parser.add_argument("--test-csv", type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = run(args)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"오류: {exc}\n")
    print("\n".join(str(path) for path in paths))
    return 0
