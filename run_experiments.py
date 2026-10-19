"""Reproduce every number and figure in the README.

    python run_experiments.py            # full run (~10-15 min on a laptop)
    python run_experiments.py --quick    # smaller GA for a fast check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.assignment import solve_assignment
from src.ndp import SIOUX_FALLS_CANDIDATES, BudgetNDP
from src.network import braess_network, load_tntp

ROOT = Path(__file__).parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# reference palette (light mode) — categorical slots 1-3 + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load_sioux_falls():
    return load_tntp(ROOT / "data/SiouxFalls_net.tntp", ROOT / "data/SiouxFalls_trips.tntp", "Sioux Falls")


def node_coords():
    xy = {}
    for line in (ROOT / "data/SiouxFalls_node.tntp").read_text().splitlines()[1:]:
        p = line.replace(";", "").split()
        if len(p) >= 3:
            xy[int(p[0]) - 1] = (float(p[1]), float(p[2]))
    return xy


# ---------------------------------------------------------------------------
def braess():
    rows = []
    for bridge in (False, True):
        r = solve_assignment(braess_network(bridge), gap_tol=1e-9)
        rows.append({"bridge": bridge, "flows": r.flow.round(1).tolist(), "avg_time_min": r.tstt / 4000})
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["Without bridge", "With bridge"]
    vals = [rows[0]["avg_time_min"], rows[1]["avg_time_min"]]
    bars = ax.bar(labels, vals, color=[BLUE, ORANGE], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f} min", ha="center", color=INK)
    ax.set_ylabel("Travel time per driver (min)")
    ax.set_ylim(0, 95)
    ax.grid(axis="x", visible=False)
    ax.set_title("Braess paradox: adding a free link makes everyone slower")
    fig.tight_layout()
    fig.savefig(FIG / "braess.png", dpi=160)
    plt.close(fig)
    return rows


def equilibrium(net):
    ue_gp = solve_assignment(net, objective="ue", method="gp", gap_tol=1e-10, max_iter=300)
    ue_fw = solve_assignment(net, objective="ue", method="cfw", gap_tol=1e-10, max_iter=1000)
    so = solve_assignment(net, objective="so", method="gp", gap_tol=1e-10, max_iter=300)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.semilogy(range(1, len(ue_gp.gap_history) + 1), ue_gp.gap_history, color=BLUE, lw=2)
    ax.semilogy(range(1, len(ue_fw.gap_history) + 1), ue_fw.gap_history, color=ORANGE, lw=2)
    ax.text(3, 1e-7, "Gradient projection", color=INK2)
    ax.text(40, 3e-3, "Conjugate Frank-Wolfe", color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative gap")
    ax.set_title("User-equilibrium convergence on Sioux Falls")
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png", dpi=160)
    plt.close(fig)

    return {
        "ue_tstt": ue_gp.tstt, "ue_gap": ue_gp.gap, "ue_iterations": ue_gp.iterations,
        "ue_beckmann": net.beckmann(ue_gp.flow),
        "cfw_gap_after_1000": ue_fw.gap,
        "so_tstt": so.tstt, "so_gap": so.gap,
        "price_of_anarchy": ue_gp.tstt / so.tstt,
    }, ue_gp


def plot_ndp_comparison(table):
    names = {"uniform": "Uniform split", "ioa": "Iterative opt.-assignment",
             "sndp": "System-optimal planner",
             "sndp+descent": "SO warm start + bi-level descent", "ga": "Genetic algorithm",
             "ga+descent": "GA + bi-level descent"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = [names[t["method"]] for t in table][::-1]
    vals = [t["reduction_pct"] for t in table][::-1]
    colors = [BLUE if t["method"] in ("ga", "ga+descent", "sndp+descent") else "#a9a8a2" for t in table][::-1]
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", color=INK)
    ax.set_xlabel("Reduction in total system travel time vs. no investment (%)")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Same budget, different planners")
    fig.tight_layout()
    fig.savefig(FIG / "ndp_comparison.png", dpi=160)
    plt.close(fig)



def network_design(net, quick):
    p = BudgetNDP(net, SIOUX_FALLS_CANDIDATES, budget_share=0.3, y_max=1.0)
    base = solve_assignment(net, gap_tol=1e-7).tstt
    results = [p.uniform(), p.ioa(), p.sndp()]
    ga, ga_best = p.genetic_algorithm(pop_size=12 if quick else 24, generations=10 if quick else 40)
    results.append(ga)
    results.append(p.descent(ga_best, name="ga+descent"))
    results.insert(3, p.descent(results[2].y, name="sndp+descent"))

    table = []
    for r in results:
        table.append({
            "method": r.method, "tstt": r.tstt, "reduction_pct": 100 * (base - r.tstt) / base,
            "budget_used_pct": 100 * r.cost / p.budget, "ue_solves": r.evaluations,
            "seconds": r.seconds, "y": dict(zip(SIOUX_FALLS_CANDIDATES, np.round(r.y, 3).tolist())),
        })

    plot_ndp_comparison(table)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(len(ga.history)), np.array(ga.history) / 1e6, color=BLUE, lw=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best TSTT (million veh·min)")
    ax.set_title("Genetic algorithm convergence (upper level)")
    fig.tight_layout()
    fig.savefig(FIG / "ga_convergence.png", dpi=160)
    plt.close(fig)

    return {"base_tstt": base, "budget": p.budget, "candidates": SIOUX_FALLS_CANDIDATES, "methods": table}, min(results, key=lambda r: r.tstt), p


def network_map(net, ue, best, p):
    xy = node_coords()
    after = solve_assignment(net, p.full_y(best.y), gap_tol=1e-7)
    vc = after.flow / net.eff_capacity(p.full_y(best.y))
    fig, ax = plt.subplots(figsize=(5.2, 6))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cmap = plt.get_cmap("Blues")
    for a in range(net.n_links):
        (x1, y1), (x2, y2) = xy[net.init[a]], xy[net.term[a]]
        expanded = a in set(p.cand) and best.y[list(p.cand).index(a)] > 1e-3
        ax.plot([x1, x2], [y1, y2], color=ORANGE if expanded else cmap(0.3 + 0.6 * min(vc[a] / 2.5, 1)),
                lw=4 if expanded else 1.5 + 2 * min(vc[a] / 2.5, 1), solid_capstyle="round", zorder=1)
    for n, (x, y) in xy.items():
        ax.scatter(x, y, s=150, color=SURFACE, edgecolor=INK2, zorder=2)
        ax.text(x, y, str(n + 1), ha="center", va="center", fontsize=7, zorder=3)
    ax.plot([], [], color=ORANGE, lw=4, label="Expanded link (best bi-level design)")
    ax.plot([], [], color=cmap(0.9), lw=3, label="Other links (darker = higher V/C)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=8)
    ax.set_title("Sioux Falls: optimised capacity expansion")
    fig.tight_layout()
    fig.savefig(FIG / "network_map.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    net = load_sioux_falls()
    summary = {"braess": braess()}
    print("Braess:", summary["braess"])
    summary["equilibrium"], ue = equilibrium(net)
    print("Equilibrium:", summary["equilibrium"])
    summary["ndp"], best, problem = network_design(net, args.quick)
    for t in summary["ndp"]["methods"]:
        print(f"{t['method']:>9}: TSTT={t['tstt']:,.0f}  -{t['reduction_pct']:.2f}%  "
              f"budget={t['budget_used_pct']:.0f}%  solves={t['ue_solves']}  {t['seconds']:.0f}s")
    network_map(net, ue, best, problem)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
