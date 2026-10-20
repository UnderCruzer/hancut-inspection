#!/usr/bin/env python3
"""여러 검출기의 예측 CSV 를 품목별 최댓값으로 합친다 (#41).

    python3 ml/scripts/fuse_scores.py \\
        --model E1=ml/runs/predictions \\
        --model 960=ml/runs/lighter_train960_20261016/predictions \\
        --model mixup=ml/runs/overlap_mixup_20261017/predictions_mixup \\
        --select calib-ap --out ml/runs/fusion_20261019/calib_ap

각 폴더에는 predict_scores.py 가 만든 calib.csv · test_easy.csv · test_hard.csv · test_hidden.csv 가 있어야 한다.
출력 폴더도 같은 네 파일이라 e2_compare.py --pred-dir 로 바로 평가한다.

--select all       모든 품목에 모든 모델의 최댓값
--select calib-ap  품목마다 쓸 모델 조합을 calib(학습 분할) AP 로 고른다. 시험셋은 보지 않는다
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hancut.eval import cli, fusion, scores  # noqa: E402

SPLITS = ("calib", "test_easy", "test_hard", "test_hidden")


def parse_models(specs: list[str]) -> dict[str, Path]:
    out = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"'{spec}' 를 읽지 못했다. 형식은 이름=폴더")
        if name in out:
            raise SystemExit(f"모델 이름이 겹친다: {name}")
        out[name] = Path(path)
    if len(out) < 2:
        raise SystemExit("모델이 둘 이상 있어야 합친다")
    return out


def load(models: dict[str, Path], split: str) -> dict[str, dict]:
    by_model = {}
    for name, folder in models.items():
        path = folder / f"{split}.csv"
        if not path.is_file():
            raise SystemExit(f"없다: {path}")
        by_model[name] = fusion.index_rows(cli.read_predictions(path))
    fusion.check_aligned(by_model)
    return by_model


def report(models, chosen, table, select: str) -> str:
    names = list(models)
    lines = ["# 점수 결합 (#41)", "", f"방식: 품목별 이미지 점수 최댓값 · 선택 `{select}`", "",
             "| 모델 | 예측 폴더 |", "|---|---|"] + [f"| {n} | `{p}` |" for n, p in models.items()]
    lines += ["", "## 품목별로 쓴 모델", "", "| 품목 | 쓴 모델 | " + " | ".join(f"calib AP {n}" for n in names) + " | calib AP 결합 |",
              "|---|---|" + "---:|" * (len(names) + 1)]
    for item, combo in chosen.items():
        single = [f"{table[item][(n,)]:.3f}" if table else "-" for n in names]
        fused = f"{table[item][combo]:.3f}" if table else "-"
        lines.append(f"| {item} | {' + '.join(combo)} | " + " | ".join(single) + f" | {fused} |")
    used = sorted({m for c in chosen.values() for m in c}, key=names.index)
    lines += ["", f"추론에 필요한 모델: {', '.join(used)} ({len(used)}개). 품목 하나라도 쓰면 그 모델은 모든 사진에 돌려야 한다."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="이름=예측 폴더 (두 번 이상)")
    parser.add_argument("--select", choices=("all", "calib-ap"), default="all")
    parser.add_argument("--min-gain", type=float, default=0.001, help="모델을 더 쓰려면 calib AP 가 이만큼 넘게 올라야 한다")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    models = parse_models(args.model)
    calib = load(models, "calib")
    items = sorted({k[1] for k in next(iter(calib.values()))})
    if args.select == "calib-ap":
        chosen, table = fusion.select_by_ap(calib, items, args.min_gain)
    else:
        chosen, table = {i: tuple(models) for i in items}, {}

    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        by_model = calib if split == "calib" else load(models, split)
        # dense 점수(#43)는 아주 작은 값이 많아 소수 6자리로 쓰면 0 으로 뭉친다
        n = scores.write_predictions(fusion.fuse(by_model, chosen), args.out / f"{split}.csv", fmt=".9g")
        print(f"{split}: {n:,}행")
    (args.out / "selection.json").write_text(json.dumps(
        {"select": args.select, "min_gain": args.min_gain, "models": {k: str(v) for k, v in models.items()},
         "chosen": {k: list(v) for k, v in chosen.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    text = report(models, chosen, table, args.select)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
