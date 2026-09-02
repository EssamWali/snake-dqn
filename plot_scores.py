"""Draw the learning curve from scores.csv.

    python plot_scores.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
WINDOW = 25


def rolling_mean(values, window):
    out, total = [], 0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def main():
    with open(ROOT / "scores.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    games = [int(r["game"]) for r in rows]
    scores = [int(r["score"]) for r in rows]

    figure, axes = plt.subplots(figsize=(9, 4.2), dpi=140)
    figure.patch.set_facecolor("#12150f")
    axes.set_facecolor("#12150f")

    axes.plot(games, scores, linewidth=0.8, alpha=0.35, color="#9AC57E", label="score")
    axes.plot(
        games,
        rolling_mean(scores, WINDOW),
        linewidth=2.2,
        color="#9AC57E",
        label=f"mean of last {WINDOW}",
    )
    # Exploration is on a fixed schedule: epsilon = 80 - games, so from game 80 the
    # agent is purely greedy. The curve should bend around there, and it does.
    axes.axvline(80, color="#DB9166", linewidth=1, linestyle="--", alpha=0.8)
    axes.annotate(
        "exploration ends",
        xy=(80, max(scores) * 0.92),
        xytext=(92, max(scores) * 0.92),
        color="#DB9166",
        fontsize=9,
    )

    axes.set_xlabel("game", color="#959e90")
    axes.set_ylabel("apples eaten", color="#959e90")
    axes.tick_params(colors="#959e90")
    for spine in axes.spines.values():
        spine.set_color("#2a3026")
    axes.grid(color="#2a3026", linewidth=0.6)
    axes.set_axisbelow(True)
    legend = axes.legend(facecolor="#1a1e17", edgecolor="#2a3026", labelcolor="#e7ebe3")
    legend.get_frame().set_alpha(1)

    out = ROOT / "docs" / "learning-curve.png"
    out.parent.mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(out, facecolor=figure.get_facecolor())
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
