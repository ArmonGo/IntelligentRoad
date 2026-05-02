
from __future__ import annotations
import copy
import sys
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from minisim import TrafficSim, TileType

_TERRAIN_COLOUR: dict[TileType, tuple] = {
    TileType.GRASS: (0.56, 0.93, 0.56, 1.0),
    TileType.WATER: (0.53, 0.81, 0.98, 1.0),
    TileType.MOUNTAIN: (0.75, 0.75, 0.75, 1.0),
    TileType.BUILDING: (0.20, 0.20, 0.20, 1.0),
    TileType.ROAD: (1.00, 0.80, 0.30, 1.0),
    TileType.DEPOT_OUT: (0.20, 0.40, 0.90, 1.0),
    TileType.DEPOT_IN: (0.90, 0.20, 0.20, 1.0),
}


def _road_colour(cap: int, max_cap: int = 5) -> tuple:
    """Orange-red gradient by capacity fraction."""
    frac = min(1.0, cap / max(1, max_cap))
    return (1.0, 0.65 - 0.35 * frac, 0.10, 1.0)


def _legend_patches(include_flow: bool = False) -> list:
    patches = [
        mpatches.Patch(facecolor=_TERRAIN_COLOUR[TileType.GRASS][:3], label="Grass"),
        mpatches.Patch(facecolor=_TERRAIN_COLOUR[TileType.WATER][:3], label="Water"),
        mpatches.Patch(
            facecolor=_TERRAIN_COLOUR[TileType.MOUNTAIN][:3], label="Mountain"
        ),
        mpatches.Patch(
            facecolor=_TERRAIN_COLOUR[TileType.BUILDING][:3], label="Building"
        ),
        mpatches.Patch(facecolor=_road_colour(1)[:3], label="Road (low cap)"),
        mpatches.Patch(facecolor=_road_colour(5)[:3], label="Road (high cap)"),
        mpatches.Patch(
            facecolor=_TERRAIN_COLOUR[TileType.DEPOT_OUT][:3], label="Depot OUT"
        ),
        mpatches.Patch(
            facecolor=_TERRAIN_COLOUR[TileType.DEPOT_IN][:3], label="Depot IN"
        ),
    ]
    if include_flow:
        patches.append(
            mpatches.Patch(facecolor=(0.8, 0.0, 0.0, 0.5), label="Flow heatmap")
        )
    return patches



def plot_sim(
    sim: TrafficSim,
    title: str = "",
    occ: dict | None = None,
    tick: int | None = None,
    ax=None,
    show: bool = True,
) -> Any:
    if ax is None:
        fig, ax = plt.subplots(figsize=(sim.w * 0.6, sim.h * 0.6))

    # Terrain background
    img = np.ones((sim.h, sim.w, 4))
    for x in range(sim.w):
        for y in range(sim.h):
            tile = sim.grid[x][y]
            if tile in (TileType.ROAD, TileType.DEPOT_OUT, TileType.DEPOT_IN):
                img[y, x] = (1, 1, 1, 1)
            else:
                img[y, x] = _TERRAIN_COLOUR.get(tile, (1, 1, 1, 1))
    ax.imshow(
        img,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
        extent=(-0.5, sim.w - 0.5, sim.h - 0.5, -0.5),
    )

    # Road tiles
    max_cap = max((sim.road_capacity.get(p, 1) for p in sim.road_capacity), default=1)
    for pos, cap in sim.road_capacity.items():
        x, y = pos
        col = _road_colour(cap, max_cap)
        ax.add_patch(
            plt.Rectangle(
                (x - 0.5, y - 0.5), 1, 1, facecolor=col, edgecolor="none", zorder=2
            )
        )
        ax.text(
            x,
            y,
            str(cap),
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            zorder=4,
        )

    def _parse_pos(p):
        """Accept tuple (x,y) or JSON-stringified '(x, y)' key."""
        if isinstance(p, str):
            return tuple(int(v) for v in p.strip("()").split(","))
        return tuple(p)

    # Flow heatmap overlay
    if occ:
        if tick is not None:
            flow = occ.get(tick, occ.get(str(tick), {}))
        else:
            flow: dict = {}
            for t_occ in occ.values():
                for p, cnt in t_occ.items():
                    key = _parse_pos(p)
                    flow[key] = flow.get(key, 0) + cnt
        if flow:
            max_flow = max(flow.values())
            for pos, cnt in flow.items():
                x, y = _parse_pos(pos)
                alpha = 0.25 + 0.55 * (cnt / max(1, max_flow))
                ax.add_patch(
                    plt.Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor=(0.8, 0.0, 0.0, alpha),
                        edgecolor="none",
                        zorder=3,
                    )
                )

    # Depots
    for d in sim.init_depots.values():
        x, y = d.pos
        col = (0.2, 0.4, 0.9) if d.kind == "out" else (0.9, 0.2, 0.2)
        ax.add_patch(
            plt.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=col,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )
        )
        label = f"S{d.id}\n{d.amount}" if d.kind == "out" else f"D{d.id}\n{d.amount}"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
            zorder=6,
        )

    # Grid lines
    ax.set_xlim(-0.5, sim.w - 0.5)
    ax.set_ylim(sim.h - 0.5, -0.5)
    # ax.set_xticks([x + 0.5 for x in range(sim.w)])
    # ax.set_yticks([y + 0.5 for y in range(sim.h)])
    ax.set_xticks([])
    ax.set_yticks([])
    # hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(labelsize=6)
    ax.grid(True, color="gray", linewidth=0.3, zorder=1)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=12)

    # ax.legend(handles=_legend_patches(include_flow=bool(occ)),
    #          loc="upper right", fontsize=5, framealpha=0.8)

    if show:
        plt.tight_layout()
        plt.show()
    return ax

def show_result(
    method: str,
    sim_orig: TrafficSim,
    meta: dict,
    solved_sim: TrafficSim | None = None,
    sim_for_plot: TrafficSim | None = None,
) -> None:
    """Print result summary and show road visualisation."""
    print(f"  {method}")
    for k in ["makespan", "road_cost", "solve_status", "all_delivered", "solve_time_s"]:
        if k in meta:
            print(f"  {k:20s}: {meta[k]}")
    if "fitness_history" in meta:
        h = meta["fitness_history"]
        print(f"  {'fitness_history':20s}: start={h[0]}  end={h[-1]}  gens={len(h)-1}")
    for k in ["agent_iterations", "net_iterations", "route_iterations"]:
        if k in meta:
            print(f"  {k:20s}: {meta[k]}")

    plot_target = sim_for_plot or solved_sim or sim_orig
    plot_sim(
        plot_target,
        title=f"{method}  |  makespan={meta.get('makespan','?')}  "
        f"cost={meta.get('road_cost','?')}",
        occ=meta.get("occ"),
    )


def show_agent_result(
    method: str,
    sim_orig: TrafficSim,
    meta: dict,
    solved_sim: TrafficSim | None = None,
    show_best: bool = False,
) -> None:
    snap_best = meta.get("_best")
    snap_final = meta.get("_final")

    if snap_best is None and snap_final is None:
        show_result(method, sim_orig, meta, solved_sim=solved_sim)
        return

    snap_plot = snap_best if show_best else snap_final
    label = "best" if show_best else "final (submitted)"

    # Reconstruct a sim with the chosen snapshot's roads
    sim_plot = copy.deepcopy(sim_orig)
    for r in snap_plot.get("roads", []):
        pos = tuple(r["pos"])
        sim_plot.add_road(pos, r["capacity"])

    print(f"  {method}  [{label}]")
    for k in ["makespan", "road_cost", "all_delivered"]:
        v = snap_plot.get(k)
        if v is not None:
            print(f"  {k:20s}: {v}")
    for k in ["solve_status", "solve_time_s"]:
        if k in meta:
            print(f"  {k:20s}: {meta[k]}")

    plot_sim(
        sim_plot,
        title=(
            f"{method}  [{label}]  |  "
            f"makespan={snap_plot.get('makespan','?')}  "
            f"cost={snap_plot.get('road_cost','?')}"
        ),
        occ=meta.get("occ"),
    )

    if not (snap_best and snap_final):
        return

    try:
        from solvers import agent as agent_solver

        cmp = agent_solver.compare_solutions(meta)
    except Exception:
        return

    print()
    print("  --- best vs final comparison ---")
    print(f"  {'solutions_identical':25s}: {cmp['solutions_identical']}")
    print(f"  {'final_all_delivered':25s}: {cmp['final_all_delivered']}")
    print(
        f"  {'makespan  best -> final':25s}: {cmp['makespan_best']} -> {cmp['makespan_final']}"
        f"  (delta={cmp['makespan_delta']})"
    )
    if cmp["road_cost_best"] is not None or cmp["road_cost_final"] is not None:
        print(
            f"  {'road_cost best -> final':25s}: {cmp['road_cost_best']} -> {cmp['road_cost_final']}"
            f"  (delta={cmp['road_cost_delta']})"
        )
    print(
        f"  {'roads_identical':25s}: {cmp['roads_identical']}"
        f"  (only_in_best={len(cmp['roads_only_in_best'])}"
        f"  only_in_final={len(cmp['roads_only_in_final'])}"
        f"  cap_changed={len(cmp['roads_capacity_changed'])})"
    )
    print(
        f"  {'routes_identical':25s}: {cmp['routes_identical']}"
        f"  (cars best={cmp['cars_count_best']}  final={cmp['cars_count_final']})"
    )



def plot_car_paths(
    sim: TrafficSim,
    car_plan: list[dict],
    departures: dict[int, int],
    title: str = "",
    max_cars: int = 20,
) -> None:
    """Draw each car's path as a coloured line overlaid on the grid."""
    fig, ax = plt.subplots(figsize=(sim.w * 0.65, sim.h * 0.65))
    plot_sim(sim, title=title, ax=ax, show=False)

    cmap = plt.cm.get_cmap("tab20", min(max_cars, len(car_plan)))
    for i, entry in enumerate(car_plan[:max_cars]):
        path = entry.get("path", [])
        if len(path) < 2:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        dep = departures.get(i, 0)
        ax.plot(
            xs,
            ys,
            color=cmap(i),
            linewidth=1.5,
            alpha=0.75,
            zorder=7,
            label=f"car {i} @t{dep}",
        )

    ax.legend(fontsize=5, loc="upper right", ncol=2, framealpha=0.7)
    plt.tight_layout()
    plt.show()


def compare_table(results: dict) -> pd.DataFrame:
    # Names that identify a solver algorithm, not a model
    _SOLVER_NAMES = {"Heuristic", "CPSAT", "GA", "Joint"}

    rows = []
    for tag, entry in sorted(results.items()):
        meta = entry["meta"]

        # Last two underscore-delimited segments are map_size + budget
        # e.g. deepseek_Joint_small_sufficient → ['deepseek_Joint','small','sufficient']
        parts = tag.rsplit("_", 2)
        if len(parts) == 3:
            method_key, map_size, budget = parts
        else:
            method_key, map_size, budget = tag, "?", "?"

        # Split method_key into networker and router on the first underscore
        # e.g. deepseek_Joint → networker=deepseek, router=Joint
        mk_parts = method_key.split("_", 1)
        if len(mk_parts) == 2:
            networker, router = mk_parts
        else:
            networker, router = method_key, "?"

        # Detect which component (if any) is a model/agent short name
        model = next((x for x in [networker, router] if x not in _SOLVER_NAMES), None)

        rows.append(
            {
                "Tag": tag,
                "Method": method_key,
                "Map": map_size,
                "Budget": budget,
                "Network": networker,
                "Router": router,
                "Model": model or "-",
                "Makespan": meta.get("makespan"),
                "Road cost": meta.get("road_cost"),
                "Delivered": meta.get("all_delivered"),
                "Status": meta.get("solve_status"),
                "Tool calls": meta.get("tool_calls_cnt"),
                "Route runs": meta.get("inner_route_runs"),
                "Total calls": meta.get("total_tool_calls"),
                "Time (s)": meta.get("solve_time_s"),
            }
        )

    return pd.DataFrame(rows)


def plot_comparison(
    sims_meta: list[tuple[str, TrafficSim, dict]],
    sim_orig: TrafficSim | None = None,
    title: str = "",
    cols: int = 3,
    snap: str = "final",
) -> None:
    
    n = len(sims_meta)
    if n == 0:
        return
    cols = min(cols, n)
    rows_fig = (n + cols - 1) // cols

    sample_sim = sims_meta[0][1]
    fig, axes = plt.subplots(
        rows_fig,
        cols,
        figsize=(sample_sim.w * 0.55 * cols, sample_sim.h * 0.55 * rows_fig),
    )
    axes = np.array(axes).flatten()

    for i, (label, sim, meta) in enumerate(sims_meta):
        snap_final = meta.get("_final")
        is_agent = snap_final is not None

        if is_agent:
            # Pick the requested snapshot; fall back to best when final is absent
            chosen = snap_final if snap == "final" else {}
            snap_label = "submitted"
            makespan = chosen.get("makespan", "?")
            road_cost = chosen.get("road_cost", "?")
            occ = chosen.get("occ")

            # Reconstruct road network from snapshot onto a clean base sim
            if sim_orig is not None and chosen.get("roads"):
                sim_plot = copy.deepcopy(sim_orig)
                for r in chosen["roads"]:
                    sim_plot.add_road(tuple(r["pos"]), r["capacity"])
            else:
                sim_plot = sim  # fallback: use whatever sim was passed
        else:
            snap_label = None
            makespan = meta.get("makespan", "?")
            road_cost = meta.get("road_cost", "?")
            occ = meta.get("occ")
            sim_plot = sim

        subtitle = (
            f"{'_'.join(label.split('_')[:-2])}\n"
            f"makespan={makespan}  cost={road_cost}"
        )
        plot_sim(sim_plot, title=subtitle, occ=occ, ax=axes[i], show=False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if title:
        plt.suptitle(title, fontsize=18, y=1.01)
    plt.tight_layout()
    plt.show()
