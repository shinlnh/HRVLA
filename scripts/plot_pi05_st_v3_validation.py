#!/usr/bin/env python3
"""Render paired PI0.5 ST v3 pilot MSE evidence as a dependency-free SVG."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def render(report: dict) -> str:
    if report.get("metric") != "open_loop_first_action_mse_macro_episode":
        raise ValueError("unexpected validation metric")
    base = report["base"]["episode_first_action_mse"]
    tuned = report["tuned"]["episode_first_action_mse"]
    episodes = sorted(base, key=int)
    if len(episodes) != 10 or set(episodes) != set(tuned):
        raise ValueError("expected ten paired validation episodes")
    maximum = max(max(base.values()), max(tuned.values())) * 1.12
    width, height = 1050, 510
    left, right, top, bottom = 92, 40, 105, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    baseline = top + plot_height
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Paired PI0.5 ST-RT validation first-action MSE by episode">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:20px;font-weight:700}'
        '.subtitle{font-size:13px}.axis{font-size:12px}.note{font-size:11px;fill:#4b5563}</style>',
        '<text class="title" x="92" y="34">PI0.5 + SONIC + ST-RT: validation pilot</text>',
        f'<text class="subtitle" x="92" y="58">{escape(report["source_dataset"])} · '
        f'base {report["base"]["macro_episode_first_action_mse"]:.4f} · '
        f'fine-tuned {report["tuned"]["macro_episode_first_action_mse"]:.4f} '
        'macro episode MSE</text>',
        f'<line x1="{left}" y1="{baseline}" x2="{width-right}" y2="{baseline}" '
        'stroke="#374151" stroke-width="1.5"/>',
    ]
    for tick in range(5):
        value = maximum * tick / 4
        y = baseline - plot_height * tick / 4
        items.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
                     'stroke="#e5e7eb" stroke-width="1"/>')
        items.append(f'<text class="axis" x="{left-12}" y="{y+4:.2f}" '
                     f'text-anchor="end">{value:.3f}</text>')
    group_width = plot_width / len(episodes)
    bar_width = group_width * 0.30
    for index, episode in enumerate(episodes):
        center = left + group_width * (index + 0.5)
        for offset, value, color in ((-bar_width, base[episode], "#334155"),
                                     (0, tuned[episode], "#06b6d4")):
            bar_height = plot_height * value / maximum
            items.append(f'<rect x="{center+offset:.2f}" y="{baseline-bar_height:.2f}" '
                         f'width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>')
        items.append(f'<text class="axis" x="{center:.2f}" y="{baseline+23}" '
                     f'text-anchor="middle">{episode}</text>')
    items.extend([
        '<rect x="760" y="25" width="13" height="13" fill="#334155"/>',
        '<text class="axis" x="779" y="36">base</text>',
        '<rect x="835" y="25" width="13" height="13" fill="#06b6d4"/>',
        '<text class="axis" x="854" y="36">fine-tuned</text>',
        f'<text class="axis" x="{left+plot_width/2:.2f}" y="{height-55}" '
        'text-anchor="middle">Validation episode index</text>',
        '<text class="note" x="92" y="476">Eight fixed frames per episode; paired diffusion seeds. '
        'Open-loop diagnostic only, not Isaac Sim success rate or a paper result.</text>',
        '</svg>',
    ])
    return "\n".join(items) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding="utf-8")


if __name__ == "__main__":
    main()
