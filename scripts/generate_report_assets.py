#!/usr/bin/env python3
"""Profile table, Pareto plot, and markdown data snapshot for the report."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_model  # noqa: E402
from scripts.run_experiments import build_run_tag  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.metrics import count_flops  # noqa: E402

FIG_DIR = ROOT / "figures"
OUT_ZH = ROOT / "REPORT_DATA_zh.md"
OUT_EN = ROOT / "REPORT_DATA.md"

# Aligned-budget configs from configs/experiments.yaml (profile only)
BUDGET_ROWS = [
    ("low", "vgg_baseline", {"width": 48}),
    ("low", "depthwise_small", {"width": 128}),
    ("low", "convnext_small", {"width": 32}),
    ("med", "vgg_baseline", {"width": 64}),
    ("med", "depthwise_small", {"width": 175}),
    ("med", "convnext_small", {"width": 52}),
    ("high", "vgg_baseline", {"width": 80}),
    ("high", "depthwise_small", {"width": 224}),
    ("high", "convnext_small", {"width": 64}),
]


def profile_defaults() -> list[dict]:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    inp = tuple(cfg.flops_input_size)
    rows = []
    for name in ["vgg_baseline", "depthwise_small", "convnext_small"]:
        m = build_model(name, width=64)
        s = count_flops(m, inp)
        rows.append({"model": name, "width": 64, **s})
    return rows


def profile_budget_table() -> list[dict]:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    inp = tuple(cfg.flops_input_size)
    rows = []
    for budget, model, kwargs in BUDGET_ROWS:
        m = build_model(model, num_classes=10, **kwargs)
        s = count_flops(m, inp)
        rows.append({"budget": budget, "model": model, **kwargs, **s})
    return rows


def load_all_results() -> list[dict]:
    rows: list[dict] = []
    for path in [ROOT / "logs" / "results.csv", ROOT / "logs" / "results.csv.bak"]:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_source"] = path.name
                rows.append(r)
    # Latest row per run_tag (by timestamp); prefer results.csv on tie
    by_tag: dict[str, dict] = {}
    for r in rows:
        tag = (r.get("run_tag") or r.get("model", "")).strip()
        if not tag:
            continue
        prev = by_tag.get(tag)
        if prev is None:
            by_tag[tag] = r
            continue
        ts, pts = r.get("timestamp", ""), prev.get("timestamp", "")
        if ts > pts or (ts == pts and r["_source"] == "results.csv"):
            by_tag[tag] = r
    return sorted(by_tag.values(), key=lambda x: (x.get("timestamp", ""), x.get("run_tag", "")))


def parse_epoch_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    """Parse trainer epoch log -> (epochs, train_acc%, val_acc%)."""
    epochs, train_accs, val_accs = [], [], []
    if not path.exists():
        return epochs, train_accs, val_accs
    import re

    pat = re.compile(
        r"\[(\d+)/\d+\].*train_acc=([\d.]+).*\| val_loss=[\d.]+ val_acc=([\d.]+)"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_accs.append(float(m.group(2)) * 100)
            val_accs.append(float(m.group(3)) * 100)
    return epochs, train_accs, val_accs


def plot_training_curves(log_path: Path, out: Path, title: str) -> None:
    epochs, train_accs, val_accs = parse_epoch_log(log_path)
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_accs, label="Train acc", color="#4477AA", linewidth=1.2)
    ax.plot(epochs, val_accs, label="Val acc", color="#EE6677", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _completed_tier_rows(rows: list[dict]) -> list[dict]:
    """200-epoch tier-aligned runs (Table 4) with test_acc."""
    tier_specs = [
        ("low", "vgg_baseline", "48", "VGG w=48"),
        ("low", "depthwise_small", "128", "DW w=128"),
        ("low", "convnext_small", "32", "CN w=32"),
        ("med", "vgg_baseline", "64", "VGG w=64"),
        ("med", "depthwise_small", "175", "DW w=175"),
        ("med", "convnext_small", "52", "CN w=52"),
        ("high", "vgg_baseline", "80", "VGG w=80"),
        ("high", "depthwise_small", "224", "DW w=224"),
        ("high", "convnext_small", "64", "CN w=64"),
    ]
    by_key: dict[tuple, dict] = {}
    for r in rows:
        if int(r.get("epochs", 0)) < 200 or not r.get("test_acc"):
            continue
        if r.get("kernel_size") or r.get("depths"):
            continue
        w = str(r.get("width", ""))
        b = r.get("budget", "")
        m = r.get("model", "")
        by_key[(b, m, w)] = r

    out = []
    for budget, model, width, label in tier_specs:
        r = by_key.get((budget, model, width))
        if r:
            out.append({**r, "label": label, "budget": budget})
    order = {"low": 0, "med": 1, "high": 2}
    return sorted(out, key=lambda x: (order.get(x["budget"], 9), x["model"]))


def plot_tier_bars(rows: list[dict], out: Path) -> None:
    pts = _completed_tier_rows(rows)
    if not pts:
        return
    labels = [p["label"] for p in pts]
    accs = [float(p["test_acc"]) * 100 for p in pts]
    colors = {
        "vgg_baseline": "#4477AA",
        "depthwise_small": "#EE6677",
        "convnext_small": "#228833",
    }
    bar_colors = [colors.get(p["model"], "#888") for p in pts]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(labels))
    ax.bar(x, accs, color=bar_colors, edgecolor="k", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("FLOPs-aligned tiers (completed 200-epoch runs)")
    ax.set_ylim(min(accs) - 2, max(accs) + 1)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.15, f"{a:.2f}%", ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_width_scaling(rows: list[dict], out: Path) -> None:
    """Scatter test acc vs FLOPs for width-scaled completed runs."""
    pts = []
    for r in rows:
        if int(r.get("epochs", 0)) < 200 or not r.get("test_acc") or not r.get("width"):
            continue
        tag = r.get("run_tag", "")
        if tag.startswith("_smoke"):
            continue
        if r.get("kernel_size") or r.get("depths"):
            continue
        pts.append(r)
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "vgg_baseline": "#4477AA",
        "depthwise_small": "#EE6677",
        "convnext_small": "#228833",
    }
    names = {
        "vgg_baseline": "VGG",
        "depthwise_small": "Depthwise",
        "convnext_small": "ConvNeXt",
    }
    for model in colors:
        sub = [r for r in pts if r["model"] == model]
        if not sub:
            continue
        xs = [int(r["flops"]) / 1e6 for r in sub]
        ys = [float(r["test_acc"]) * 100 for r in sub]
        ax.plot(xs, ys, "o-", label=names[model], color=colors[model], linewidth=1.5, markersize=7)
        for r in sub:
            ax.annotate(
                f"w={r['width']}",
                (int(r["flops"]) / 1e6, float(r["test_acc"]) * 100),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("FLOPs (M)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Width scaling (completed runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kernel_scan(rows: list[dict], out: Path) -> None:
    """Kernel size vs test accuracy for CN w=52 and DW w=175 (200-epoch runs)."""
    cn_pts: list[tuple[int, float, float]] = []
    dw_pts: list[tuple[int, float, float]] = []
    for r in rows:
        if int(r.get("epochs", 0)) < 200 or not r.get("test_acc"):
            continue
        tag = r.get("run_tag", "")
        if "convnext_small_w52" in tag and r.get("kernel_size"):
            cn_pts.append((int(r["kernel_size"]), int(r["flops"]) / 1e6, float(r["test_acc"]) * 100))
        elif tag == "convnext_small_w52_seed42_ep200":
            cn_pts.append((7, int(r["flops"]) / 1e6, float(r["test_acc"]) * 100))
        elif "depthwise_small_w175" in tag and r.get("kernel_size"):
            dw_pts.append((int(r["kernel_size"]), int(r["flops"]) / 1e6, float(r["test_acc"]) * 100))
        elif tag == "depthwise_small_w175_seed42_ep200":
            dw_pts.append((3, int(r["flops"]) / 1e6, float(r["test_acc"]) * 100))

    if not cn_pts and not dw_pts:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pts, title, color in [
        (axes[0], sorted(cn_pts), "ConvNeXt w=52", "#228833"),
        (axes[1], sorted(dw_pts), "Depthwise w=175", "#EE6677"),
    ]:
        if not pts:
            ax.set_visible(False)
            continue
        ks = [p[0] for p in pts]
        accs = [p[2] for p in pts]
        ax.plot(ks, accs, "o-", color=color, linewidth=2, markersize=8)
        for k, flops, acc in pts:
            ax.annotate(f"{acc:.1f}%", (k, acc), fontsize=8, xytext=(0, 6), textcoords="offset points", ha="center")
        ax.set_xlabel("Kernel size")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(title)
        ax.set_xticks(sorted(set(ks)))
        ax.grid(True, alpha=0.3)
    fig.suptitle("Kernel scan (med FLOPs region, 200 epoch)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_depth_scan(rows: list[dict], out: Path) -> None:
    """ConvNeXt depth variants: test acc vs FLOPs."""
    labels: list[str] = []
    flops_l: list[float] = []
    accs: list[float] = []
    specs = [
        ("convnext_small_w64_seed42_ep200", "d(2,2,2) w=64"),
        ("convnext_small_w64_d1-1-1_seed42_ep200", "d(1,1,1) w=64"),
        ("convnext_small_w43_d3-3-3_seed42_ep200", "d(3,3,3) w=43"),
        ("convnext_small_w52_seed42_ep200", "d(2,2,2) w=52"),
    ]
    by_tag = {r.get("run_tag", ""): r for r in rows if int(r.get("epochs", 0)) >= 200}
    for tag, label in specs:
        r = by_tag.get(tag)
        if not r or not r.get("test_acc"):
            continue
        labels.append(label)
        flops_l.append(int(r["flops"]) / 1e6)
        accs.append(float(r["test_acc"]) * 100)

    if len(labels) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#228833" if float(a) > 80 else "#CC6677" for a in accs]
    x = range(len(labels))
    ax.bar(x, accs, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("ConvNeXt depth scaling (200 epoch)")
    for i, (a, f) in enumerate(zip(accs, flops_l)):
        ax.text(i, a + 0.3, f"{a:.1f}%\n{f:.0f}M", ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pareto(rows: list[dict], out: Path) -> None:
    pts = []
    for r in rows:
        if not r.get("test_acc") or not r.get("flops"):
            continue
        try:
            ep = int(r.get("epochs", 0))
        except ValueError:
            continue
        if ep < 100:
            continue
        tag = r.get("run_tag", "")
        if tag.startswith("_smoke") or tag in ("vgg_baseline", "convnext_small"):
            if ep < 200:
                continue
        pts.append(r)
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "vgg_baseline": "#4477AA",
        "depthwise_small": "#EE6677",
        "convnext_small": "#228833",
    }
    for r in pts:
        flops = int(r["flops"]) / 1e6
        acc = float(r["test_acc"]) * 100
        label = (r.get("run_tag") or r["model"]).replace("_seed42_ep200", "")
        ax.scatter(
            flops,
            acc,
            s=90,
            c=colors.get(r["model"], "#888888"),
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
        )
        ax.annotate(
            label,
            (flops, acc),
            fontsize=7,
            alpha=0.85,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("FLOPs (M)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy–FLOPs (completed runs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fmt_pct(x: str | float | None) -> str:
    if x in ("", None):
        return "—"
    try:
        return f"{float(x) * 100:.2f}%"
    except ValueError:
        return "—"


def fmt_flops(x: str | int | None) -> str:
    if x in ("", None):
        return "—"
    return f"{int(x) / 1e6:.1f}M"


def build_markdown(lang: str = "zh") -> str:
    profiles = profile_defaults()
    budget_profiles = profile_budget_table()
    results = load_all_results()

    if lang == "zh":
        title = "# 报告实验数据（自动生成）"
        note = "> 由 `scripts/generate_report_assets.py` 生成；训练完成后重新运行以更新。"
        t2 = "## 表 2 — 默认 width=64（未对齐 FLOPs）"
        t4 = "## 表 4 — 三档位 FLOPs 对齐（profile，待训练填精度）"
        t_done = "## 已完成训练（含 test accuracy）"
        hdr_done = "| run_tag | epochs | budget | width | FLOPs | Best Val | Test | 备注 |"
        hdr_bud = "| Budget | Model | width | FLOPs | Params |"
    else:
        title = "# Report experiment data (auto-generated)"
        note = "> Generated by `scripts/generate_report_assets.py`; re-run after training."
        t2 = "## Table 2 — Default width=64 (unmatched FLOPs)"
        t4 = "## Table 4 — Three-tier FLOPs alignment (profile; accuracies pending)"
        t_done = "## Completed training runs"
        hdr_done = "| run_tag | epochs | budget | width | FLOPs | Best Val | Test | Note |"
        hdr_bud = "| Budget | Model | width | FLOPs | Params |"

    lines = [title, "", note, "", t2, "", "| Model | Params | FLOPs |", "|-------|--------|-------|"]
    for p in profiles:
        lines.append(f"| {p['model']} | {p['params']:,} | {fmt_flops(p['flops'])} |")

    lines += ["", t4, "", hdr_bud, "|--------|-------|-------|-------|--------|"]
    for p in budget_profiles:
        w = p.get("width", "")
        lines.append(
            f"| {p['budget']} | {p['model']} | {w} | {fmt_flops(p['flops'])} | {p['params']:,} |"
        )

    lines += ["", t_done, "", hdr_done, "|--------|--------|--------|-------|-------|----------|------|------|"]
    for r in results:
        note_cell = ""
        tag = r.get("run_tag", "")
        if tag.endswith("_ep200"):
            ck = ROOT / "checkpoints" / f"{tag}.pt"
            if ck.exists():
                import torch

                meta = torch.load(ck, map_location="cpu", weights_only=False)
                ep_done = int(meta.get("epoch", 200))
                if ep_done < 200:
                    note_cell = f"ckpt {ep_done}/200"
        lines.append(
            f"| {r.get('run_tag', r['model'])} | {r.get('epochs', '—')} | "
            f"{r.get('budget', '—')} | {r.get('width', '—')} | "
            f"{fmt_flops(r.get('flops', ''))} | {fmt_pct(r.get('best_val_acc'))} | "
            f"{fmt_pct(r.get('test_acc'))} | {note_cell} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_all_results()
    log_dir = ROOT / "logs" / "epochs"

    plot_pareto(results, FIG_DIR / "pareto_accuracy_flops.png")
    plot_training_curves(
        log_dir / "vgg_baseline_w64_seed42_ep200.log",
        FIG_DIR / "vgg_baseline_train_val.png",
        "VGG baseline (w=64) — train / val accuracy",
    )
    plot_tier_bars(results, FIG_DIR / "tier_test_accuracy.png")
    plot_width_scaling(results, FIG_DIR / "width_scaling_accuracy_flops.png")
    plot_kernel_scan(results, FIG_DIR / "kernel_scan_test_accuracy.png")
    plot_depth_scan(results, FIG_DIR / "depth_scan_test_accuracy.png")

    try:
        OUT_ZH.write_text(build_markdown("zh"), encoding="utf-8")
        OUT_EN.write_text(build_markdown("en"), encoding="utf-8")
        print(f"Wrote {OUT_ZH}")
        print(f"Wrote {OUT_EN}")
    except ModuleNotFoundError as e:
        print(f"Skip markdown tables (need torch): {e}")

    for name in (
        "pareto_accuracy_flops.png",
        "vgg_baseline_train_val.png",
        "tier_test_accuracy.png",
        "width_scaling_accuracy_flops.png",
        "kernel_scan_test_accuracy.png",
        "depth_scan_test_accuracy.png",
    ):
        p = FIG_DIR / name
        if p.exists():
            print(f"Figure: {p}")


if __name__ == "__main__":
    main()
