"""README / 보고서용 논문식 그림을 만든다 (#64).

숫자는 `docs/figures/data/results.json` (docs/results.md 에서 옮겨 적음) 과
`docs/figures/data/browser_parity.csv` (배포 사이트 vs PyTorch 서버, 36장 × 12종) 에서만 읽는다.
사진은 쓰지 않는다.

    python ml/scripts/make_figures.py            # docs/figures/*.png
    python ml/scripts/make_figures.py --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "figures" / "data"

ZONES = ("auto_clear", "review", "auto_alarm")


def load_results(path: Path = DATA / "results.json") -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]
    if len(items) != 12 or len(set(items)) != 12:
        raise ValueError("items must list the 12 PIDray classes once")
    for row in data["method_progression"]["rows"]:
        if not row["lo"] <= row["mean"] <= row["hi"]:
            raise ValueError(f"mean outside range: {row['label']}")
    if set(data["detector_map50"]["values"]) != set(items):
        raise ValueError("mAP table must cover all 12 items")
    for name, row in data["detector_map50"]["values"].items():
        if len(row) != len(data["detector_map50"]["splits"]) or not all(0 <= v <= 1 for v in row):
            raise ValueError(f"bad mAP row: {name}")
    e2 = data["e2_by_cap"]
    if e2["positives"] + e2["negatives"] != e2["units"]:
        raise ValueError("positives + negatives must equal units")
    for row in e2["rows"]:
        auto = 100 * (1 - row["review"] / e2["units"])
        if abs(auto - row["auto"]) > 0.01:
            raise ValueError(f"auto rate does not match review count: {row}")
    return data


def load_parity(path: Path = DATA / "browser_parity.csv") -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["server_score"] = float(row["server_score"])
        row["browser_score"] = float(row["browser_score"])
        if row["server_zone"] not in ZONES or row["browser_zone"] not in ZONES:
            raise ValueError(f"unknown zone: {row}")
    return rows


def parity_summary(rows: list[dict], floor: float = 1e-3) -> dict:
    """구간 뒤집힘 수와, 점수가 floor 이상인 행의 최대 상대 차."""
    flips = sum(r["server_zone"] != r["browser_zone"] for r in rows)
    rel = [abs(r["browser_score"] - r["server_score"]) / r["server_score"] for r in rows if r["server_score"] >= floor]
    return {
        "rows": len(rows),
        "images": len({r["image"] for r in rows}),
        "flips": flips,
        "max_rel": max(rel) if rel else 0.0,
    }


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })
    return plt


GREY, BLUE, RED, ORANGE = "#8a8f98", "#2f5d9e", "#c0392b", "#d98a1e"


def fig_methods(plt, data: dict, out: Path) -> Path:
    rows = data["method_progression"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        color = {"final": BLUE, "exploratory": ORANGE}.get(r["kind"], GREY)
        ax.barh(yi, r["mean"], color=color, height=0.62, zorder=2)
        ax.errorbar(r["mean"], yi, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=2.5, zorder=3)
        note = f"{r['mean']:.1f}%" + (f"  ({r['violations']}/10 cap violated)" if r["violations"] else "")
        ax.text(r["hi"] + 1.0, yi, note, va="center", fontsize=8)
    ax.set_yticks(y, [f"{r['label']}  ×{r['cost']}" for r in rows])
    ax.set_xlim(0, 80)
    ax.set_xlabel("Automated decisions (% of image × item), mean and range over 10 seeds")
    ax.set_title("Fig. 1  Automation at a 5% miss cap (all 36 item×difficulty cells, δ = 0.05/36)", loc="left")
    ax.grid(axis="y", visible=False)
    path = out / "fig1_methods.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_calibration(plt, data: dict, out: Path) -> Path:
    cal = data["calibration"]
    caps = cal["caps"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    markers = ["o", "s", "^", "D"]
    colors = [RED, ORANGE, GREY, BLUE]
    for (name, v), m, c in zip(cal["settings"].items(), markers, colors):
        a1.plot(caps, v["miss"], marker=m, color=c, label=name, lw=1.2, ms=4)
        a2.plot(caps, v["auto"], marker=m, color=c, lw=1.2, ms=4)
    a1.plot([0, 6], [0, 6], ls="--", color="black", lw=0.8, label="miss = cap")
    a1.set_xlim(0.5, 5.5)
    a1.set_xticks(caps)
    a1.set_xlabel("Miss-rate cap (%)")
    a1.set_ylabel("Observed miss rate (%)")
    a1.set_title("(a) Does the cap hold?", loc="left")
    a2.set_xticks(caps)
    a2.set_xlabel("Miss-rate cap (%)")
    a2.set_ylabel("Automated (%)")
    a2.set_title("(b) What it costs", loc="left")
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Fig. 2  Calibrating on the training distribution breaks the cap (E2, 9,109 test images)",
                 x=0.0, ha="left", fontsize=10)
    fig.tight_layout()
    path = out / "fig2_calibration.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_map(plt, data: dict, out: Path) -> Path:
    det = data["detector_map50"]
    items = data["items"]
    m = [det["values"][i] for i in items]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(m, cmap="RdYlBu", vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(det["splits"])), det["splits"])
    ax.set_yticks(range(len(items)), items)
    ax.grid(False)
    for r, row in enumerate(m):
        for c, v in enumerate(row):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < 0.45 or v > 0.93 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="mAP@0.5")
    ax.set_title("Fig. 3  E1 detector mAP@0.5 by item and split", loc="left")
    path = out / "fig3_detector_map.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_review(plt, data: dict, out: Path) -> Path:
    v = data["review_nms_vs_dense"]["values"]
    names = list(v)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    x = range(len(names))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [v[n][0] for n in names], w, color=GREY, label="NMS score", zorder=2)
    ax.bar([i + w / 2 for i in x], [v[n][1] for n in names], w, color=BLUE, label="dense score (pre-NMS)", zorder=2)
    for i, n in enumerate(names):
        ax.text(i + w / 2, v[n][1] + 2, f"{v[n][1]:.0f}", ha="center", fontsize=7)
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Sent to human review (%)")
    ax.set_ylim(0, 118)
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax.set_title("Fig. 4  Same model, same forward pass: review rate by score rule", loc="left")
    ax.grid(axis="x", visible=False)
    path = out / "fig4_review_rate.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_parity(plt, rows: list[dict], out: Path) -> Path:
    s = parity_summary(rows)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    colors = {"auto_clear": "#2e8657", "review": ORANGE, "auto_alarm": RED}
    for zone in ZONES:
        pts = [r for r in rows if r["server_zone"] == zone]
        ax.scatter([max(r["server_score"], 1e-7) for r in pts], [max(r["browser_score"], 1e-7) for r in pts],
                   s=8, color=colors[zone], label=zone, alpha=0.8, linewidths=0, zorder=2)
    ax.plot([1e-7, 1], [1e-7, 1], ls="--", color="black", lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-7, 1.5)
    ax.set_ylim(1e-7, 1.5)
    ax.set_xlabel("Server score (PyTorch, OpenCV)")
    ax.set_ylabel("Browser score (ONNX, wasm)")
    ax.legend(fontsize=7, loc="upper left", markerscale=1.5)
    ax.set_title(f"Fig. 5  Browser vs server: {s['rows']} scores,\n{s['images']} images, {s['flips']} zone flips, "
                 f"max rel. diff {100 * s['max_rel']:.1f}% (score ≥ 1e-3)", loc="left", fontsize=9)
    path = out / "fig5_browser_parity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(out: Path) -> list[Path]:
    plt = _style()
    out.mkdir(parents=True, exist_ok=True)
    data = load_results()
    rows = load_parity()
    return [
        fig_methods(plt, data, out),
        fig_calibration(plt, data, out),
        fig_map(plt, data, out),
        fig_review(plt, data, out),
        fig_parity(plt, rows, out),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    for path in make_all(p.parse_args().out):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
