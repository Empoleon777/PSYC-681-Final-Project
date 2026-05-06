#!/usr/bin/env python3
"""Build publication-style SVG visuals from experiment artifacts."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Graphs" / "experiment_viz"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def write_svg(path: Path, width: int, height: int, body: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; fill: #1b1f23; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".subtitle { font-size: 12px; fill: #5a6472; }",
        ".axis { font-size: 11px; fill: #3a4451; }",
        ".label { font-size: 12px; font-weight: 600; }",
        ".value { font-size: 11px; fill: #203040; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]
    payload.extend(body)
    payload.append("</svg>")
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def color_for_value(v: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    # blue gradient (light -> dark)
    r = int(232 - 140 * t)
    g = int(242 - 165 * t)
    b = int(252 - 120 * t)
    return f"rgb({r},{g},{b})"


def draw_model_ladder_heatmap() -> Path:
    path = OUT_DIR / "figure01_model_ladder_heatmap.svg"
    rows = read_csv(ROOT / "outputs" / "final_32_36" / "task32_model_ladder_table.csv")
    models = sorted({r["model"] for r in rows})
    regime_order = [
        "in_domain",
        "leave_one_topic_out",
        "leave_one_community_out",
        "external_transfer_mitweet",
    ]
    values: Dict[Tuple[str, str], float] = {(r["model"], r["regime"]): float(r["macro_f1_mean"]) for r in rows}
    v_list = list(values.values())
    vmin = min(v_list) if v_list else 0.0
    vmax = max(v_list) if v_list else 1.0

    left = 190
    top = 110
    cw = 190
    ch = 64
    width = left + cw * len(regime_order) + 40
    height = top + ch * len(models) + 80
    body: List[str] = []
    body.append('<text class="title" x="28" y="42">Model Ladder by Evaluation Regime</text>')
    body.append('<text class="subtitle" x="28" y="64">Macro-F1 values from outputs/final_32_36/task32_model_ladder_table.csv</text>')

    for j, regime in enumerate(regime_order):
        x = left + j * cw + cw / 2
        label = regime.replace("_", " ")
        body.append(f'<text class="label" x="{x:.1f}" y="{top - 16}" text-anchor="middle">{esc(label)}</text>')

    for i, model in enumerate(models):
        y = top + i * ch
        body.append(f'<text class="label" x="{left - 14}" y="{y + ch / 2 + 4:.1f}" text-anchor="end">{esc(model)}</text>')
        for j, regime in enumerate(regime_order):
            x = left + j * cw
            v = values.get((model, regime), 0.0)
            fill = color_for_value(v, vmin, vmax)
            body.append(f'<rect x="{x}" y="{y}" width="{cw - 6}" height="{ch - 6}" rx="10" fill="{fill}" stroke="#c5d2e3"/>')
            body.append(f'<text class="value" x="{x + (cw - 6) / 2:.1f}" y="{y + (ch - 6) / 2 + 4:.1f}" text-anchor="middle">{v:.3f}</text>')

    # color legend
    lx = 30
    ly = height - 36
    lw = 240
    lh = 14
    for k in range(lw):
        t = k / max(1, lw - 1)
        c = color_for_value(vmin + t * (vmax - vmin), vmin, vmax)
        body.append(f'<line x1="{lx + k}" y1="{ly}" x2="{lx + k}" y2="{ly + lh}" stroke="{c}"/>')
    body.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="none" stroke="#9eb2ca"/>')
    body.append(f'<text class="axis" x="{lx}" y="{ly - 4}" text-anchor="start">Low ({vmin:.3f})</text>')
    body.append(f'<text class="axis" x="{lx + lw}" y="{ly - 4}" text-anchor="end">High ({vmax:.3f})</text>')

    write_svg(path, width, height, body)
    return path


def draw_ablation_bars() -> Path:
    path = OUT_DIR / "figure02_ablation_deltas.svg"
    rows = read_csv(ROOT / "outputs" / "final_32_36" / "task33_ablation_table.csv")
    rows = sorted(rows, key=lambda r: float(r["delta"]))
    values = [float(r["delta"]) for r in rows]
    vmax = max(abs(v) for v in values) if values else 1.0

    left = 320
    top = 86
    row_h = 36
    bar_h = 20
    plot_w = 460
    width = 840
    height = top + row_h * len(rows) + 70
    zero_x = left + plot_w / 2
    scale = (plot_w / 2 - 10) / max(1e-9, vmax)

    body: List[str] = []
    body.append('<text class="title" x="28" y="40">Ablation Effect Sizes (Task 33)</text>')
    body.append('<text class="subtitle" x="28" y="62">Positive bars indicate improvement/support; values are deltas from task33_ablation_table.csv</text>')
    body.append(f'<line x1="{zero_x:.1f}" y1="{top - 14}" x2="{zero_x:.1f}" y2="{height - 36}" stroke="#9fb0c6" stroke-dasharray="4 4"/>')
    body.append(f'<text class="axis" x="{zero_x - 4:.1f}" y="{top - 22}" text-anchor="end">0</text>')

    for i, row in enumerate(rows):
        y = top + i * row_h
        delta = float(row["delta"])
        x0 = zero_x
        x1 = zero_x + delta * scale
        bx = min(x0, x1)
        bw = abs(x1 - x0)
        color = "#2e7d32" if delta >= 0 else "#c62828"
        label = f'{row["ablation_id"]} {row["name"]}'
        body.append(f'<text class="axis" x="{left - 14}" y="{y + bar_h - 3}" text-anchor="end">{esc(label)}</text>')
        body.append(f'<rect x="{bx:.1f}" y="{y}" width="{max(1.5, bw):.1f}" height="{bar_h}" rx="4" fill="{color}" opacity="0.85"/>')
        tx = x1 + 6 if delta >= 0 else x1 - 6
        anchor = "start" if delta >= 0 else "end"
        body.append(f'<text class="value" x="{tx:.1f}" y="{y + bar_h - 4}" text-anchor="{anchor}">{delta:.3f}</text>')

    write_svg(path, width, height, body)
    return path


def draw_robustness_panel() -> Path:
    path = OUT_DIR / "figure03_robustness_panel.svg"
    noise = read_csv(ROOT / "outputs" / "final_32_36" / "task34_noise_perturbation.csv")
    cons = read_csv(ROOT / "outputs" / "final_32_36" / "task34_counterfactual_consistency.csv")
    cons = [r for r in cons if r["edit_type"] != "overall"]

    width = 980
    height = 430
    body: List[str] = []
    body.append('<text class="title" x="28" y="40">Robustness and Counterfactual Consistency</text>')
    body.append('<text class="subtitle" x="28" y="62">Task 34 noise stress test and flat-vs-hier consistency by edit type</text>')

    # left panel: noise line
    lx, ly, lw, lh = 60, 105, 390, 250
    body.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#fbfdff" stroke="#d4deea"/>')
    xs = [float(r["noise_rate"]) for r in noise]
    ys = [float(r["macro_f1_mean"]) for r in noise]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    yr = max(1e-9, ymax - ymin)

    def px(x: float) -> float:
        return lx + 35 + (x - xmin) / max(1e-9, xmax - xmin) * (lw - 60)

    def py(y: float) -> float:
        return ly + lh - 30 - (y - ymin) / yr * (lh - 60)

    points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    body.append(f'<polyline points="{points}" fill="none" stroke="#1565c0" stroke-width="3"/>')
    for x, y in zip(xs, ys):
        body.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4.2" fill="#1565c0"/>')
        body.append(f'<text class="value" x="{px(x):.1f}" y="{py(y) - 8:.1f}" text-anchor="middle">{y:.3f}</text>')
    body.append(f'<text class="label" x="{lx + 12}" y="{ly + 22}">Noise stress curve</text>')
    body.append(f'<text class="axis" x="{lx + lw / 2:.1f}" y="{ly + lh - 6}" text-anchor="middle">Noise rate</text>')
    body.append(f'<text class="axis" x="{lx + 10}" y="{ly + lh / 2:.1f}" transform="rotate(-90 {lx + 10} {ly + lh / 2:.1f})" text-anchor="middle">Macro-F1</text>')

    # right panel: consistency grouped bars
    rx, ry, rw, rh = 510, 105, 420, 250
    body.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="#fbfdff" stroke="#d4deea"/>')
    body.append(f'<text class="label" x="{rx + 12}" y="{ry + 22}">Counterfactual consistency</text>')

    groups = [r["edit_type"] for r in cons]
    g_w = (rw - 70) / max(1, len(groups))
    maxv = 1.0
    for i, r in enumerate(cons):
        gx = rx + 45 + i * g_w
        flat = float(r["flat_consistency"])
        hier = float(r["hierarchical_consistency"])
        b_w = (g_w - 18) / 2
        f_h = (rh - 70) * (flat / maxv)
        h_h = (rh - 70) * (hier / maxv)
        fy = ry + rh - 30 - f_h
        hy = ry + rh - 30 - h_h
        body.append(f'<rect x="{gx}" y="{fy:.1f}" width="{b_w:.1f}" height="{f_h:.1f}" fill="#90caf9" stroke="#5e92c7"/>')
        body.append(f'<rect x="{gx + b_w + 8:.1f}" y="{hy:.1f}" width="{b_w:.1f}" height="{h_h:.1f}" fill="#42a5f5" stroke="#1e88e5"/>')
        body.append(f'<text class="axis" x="{gx + g_w / 2:.1f}" y="{ry + rh - 8}" text-anchor="middle">{esc(r["edit_type"].replace("_", " "))}</text>')
    # legend
    body.append(f'<rect x="{rx + rw - 165}" y="{ry + 14}" width="12" height="12" fill="#90caf9" stroke="#5e92c7"/>')
    body.append(f'<text class="axis" x="{rx + rw - 148}" y="{ry + 24}">Flat</text>')
    body.append(f'<rect x="{rx + rw - 103}" y="{ry + 14}" width="12" height="12" fill="#42a5f5" stroke="#1e88e5"/>')
    body.append(f'<text class="axis" x="{rx + rw - 86}" y="{ry + 24}">Hierarchical</text>')

    write_svg(path, width, height, body)
    return path


def draw_error_profile_panel() -> Path:
    path = OUT_DIR / "figure04_error_profile_panel.svg"
    summary = read_json(ROOT / "outputs" / "final_32_36" / "task35_error_analysis_summary.json")
    stage = summary.get("stage_counts", {})
    tax = summary.get("taxonomy_counts", {})

    width = 980
    height = 460
    body: List[str] = []
    body.append('<text class="title" x="28" y="40">Error Composition Profile (Task 35)</text>')
    body.append('<text class="subtitle" x="28" y="62">Stage-level failures and content taxonomy from 100 sampled error examples</text>')

    # stage bars
    lx, ly, lw, lh = 60, 105, 390, 280
    body.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#fbfdff" stroke="#d4deea"/>')
    stage_items = sorted(stage.items(), key=lambda kv: kv[1], reverse=True)
    smax = max(stage.values()) if stage else 1
    bw = (lw - 80) / max(1, len(stage_items))
    body.append(f'<text class="label" x="{lx + 12}" y="{ly + 22}">Failure stage distribution</text>')
    for i, (k, v) in enumerate(stage_items):
        x = lx + 45 + i * bw
        h = (lh - 70) * (float(v) / max(1, smax))
        y = ly + lh - 30 - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 14:.1f}" height="{h:.1f}" fill="#ffcc80" stroke="#f4a259"/>')
        body.append(f'<text class="value" x="{x + (bw - 14) / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{v}</text>')
        body.append(f'<text class="axis" x="{x + (bw - 14) / 2:.1f}" y="{ly + lh - 8}" text-anchor="middle">{esc(k.replace("_", " "))}</text>')

    # taxonomy horizontal bars
    rx, ry, rw, rh = 510, 105, 420, 280
    body.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="#fbfdff" stroke="#d4deea"/>')
    body.append(f'<text class="label" x="{rx + 12}" y="{ry + 22}">Content taxonomy counts</text>')
    tax_items = sorted(tax.items(), key=lambda kv: kv[1], reverse=True)
    tmax = max(tax.values()) if tax else 1
    row_h = (rh - 46) / max(1, len(tax_items))
    for i, (k, v) in enumerate(tax_items):
        y = ry + 34 + i * row_h
        w = (rw - 190) * (float(v) / max(1, tmax))
        body.append(f'<text class="axis" x="{rx + 10}" y="{y + row_h * 0.65:.1f}">{esc(k.replace("_", " "))}</text>')
        body.append(f'<rect x="{rx + 172}" y="{y + 5:.1f}" width="{w:.1f}" height="{row_h - 12:.1f}" fill="#81c784" stroke="#5fa96a"/>')
        body.append(f'<text class="value" x="{rx + 178 + w:.1f}" y="{y + row_h * 0.65:.1f}">{v}</text>')

    write_svg(path, width, height, body)
    return path


def draw_checkpoint_strip() -> Path:
    path = OUT_DIR / "figure05_checkpoint_status.svg"
    summary = read_json(ROOT / "outputs" / "final_32_36" / "task36_final_summary.json")
    status = summary.get("checkpoint_status", {})
    items = list(status.items())

    width = 1100
    height = 220
    body: List[str] = []
    body.append('<text class="title" x="28" y="40">Checkpoint Status Snapshot</text>')
    body.append('<text class="subtitle" x="28" y="62">Task 36 checkpoint map (status booleans from final summary)</text>')

    left = 30
    top = 90
    tile_w = 128
    tile_h = 88
    gap = 8
    for i, (name, ok) in enumerate(items):
        x = left + i * (tile_w + gap)
        fill = "#dff3e5" if bool(ok) else "#fde2e2"
        stroke = "#5ea776" if bool(ok) else "#cc5a5a"
        badge = "PASS" if bool(ok) else "FAIL"
        body.append(f'<rect x="{x}" y="{top}" width="{tile_w}" height="{tile_h}" rx="10" fill="{fill}" stroke="{stroke}"/>')
        body.append(f'<text class="label" x="{x + 10}" y="{top + 24}">{esc(name.split("_")[0])}</text>')
        body.append(f'<text class="axis" x="{x + 10}" y="{top + 46}">{esc(name[2:].replace("_", " ")[:24])}</text>')
        body.append(f'<text class="value" x="{x + 10}" y="{top + 70}">{badge}</text>')

    write_svg(path, width, height, body)
    return path


def write_index(paths: List[Path]) -> Path:
    index = OUT_DIR / "README.md"
    lines = [
        "# Experiment Visualization Bundle",
        "",
        "Generated from current artifacts in `outputs/final_32_36` and related model outputs.",
        "",
        "Figures:",
    ]
    for p in paths:
        lines.append(f"- [{p.name}]({p.name})")
    lines.append("")
    lines.append("Generator: `scripts/make_experiment_visuals.py`")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = [
        draw_model_ladder_heatmap(),
        draw_ablation_bars(),
        draw_robustness_panel(),
        draw_error_profile_panel(),
        draw_checkpoint_strip(),
    ]
    generated.append(write_index(generated))
    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
