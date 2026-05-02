
from __future__ import annotations
import copy
import sys
import os
import time
from collections import defaultdict
from typing import Any, Callable

# Allow import from parent directory when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minisim import TrafficSim, TileType, TERRAIN_ROAD_COST, TERRAIN_MAX_CAPACITY

_TILE_NAME: dict[TileType, str] = {
    TileType.GRASS: "grass",
    TileType.WATER: "water",
    TileType.MOUNTAIN: "mountain",
    TileType.BUILDING: "building",
    TileType.ROAD: "road",
    TileType.DEPOT_OUT: "depot_out",
    TileType.DEPOT_IN: "depot_in",
}

# Short codes used in the tiles list - reduces token count vs full names
_TILE_CODE: dict[TileType, str] = {
    TileType.GRASS: "G",
    TileType.WATER: "W",
    TileType.MOUNTAIN: "M",
    TileType.BUILDING: "B",
    TileType.ROAD: "R",
    TileType.DEPOT_OUT: "O",
    TileType.DEPOT_IN: "I",
}

_LEGEND = {
    "G": "grass",
    "W": "water",
    "M": "mountain",
    "B": "building (impassable, no roads)",
    "O": "depot_out (cars start here)",
    "I": "depot_in  (cars end here)",
    "R": "road (already built)",
}

_TERRAIN_RULES = {
    name: {"build_cost_per_cap": int(cost), "max_capacity": TERRAIN_MAX_CAPACITY[name]}
    for name, cost in TERRAIN_ROAD_COST.items()
    if cost != float("inf")
}
_TERRAIN_RULES["building"] = {"build_cost_per_cap": None, "max_capacity": 0}


def world_to_json(sim: TrafficSim) -> dict[str, Any]:
    """Serialise sim state to a compact, LLM-readable dict for prompting / evaluation.
    """
    # Flat tile list - row-major (y outer, x inner) for natural reading order
    tiles = []
    for y in range(sim.h):
        for x in range(sim.w):
            code = _TILE_CODE.get(sim.grid[x][y], "G")
            tiles.append([x, y, code])

    # Depots
    depots_out = []
    depots_in = []
    for d in sorted(sim.init_depots.values(), key=lambda d: d.id):
        entry = {"id": d.id, "pos": list(d.pos), "amount": d.amount}
        (depots_out if d.kind == "out" else depots_in).append(entry)

    # Roads (empty list if none built yet)
    roads = []
    for pos, cap in sorted(sim.road_capacity.items()):
        terrain = sim.road_terrain.get(pos, "grass")
        roads.append(
            {
                "pos": list(pos),
                "capacity": cap,
                "terrain": terrain,
                "cost": sim.calculate_road_cost(pos, cap),
            }
        )

    return {
        "width": sim.w,
        "height": sim.h,
        "budget": sim.budget,
        "legend": _LEGEND,
        "tiles": tiles,
        "depots_out": depots_out,
        "depots_in": depots_in,
        "roads": roads,
        "terrain_rules": _TERRAIN_RULES,
    }

def plan_to_occ(
    car_plan: list[dict],
    departures: dict[int, int],
) -> dict[int, dict]:
    """Convert individual car plans to aggregate occupancy by tick.

    Returns {tick: {pos: car_count}} covering all road tiles a car occupies.
    The final element of each path (depot_in position) is excluded because
    cars entering a depot are removed from the road network.

    tick = departure + path_step_index.
    """
    occ: dict = defaultdict(lambda: defaultdict(int))
    for idx, entry in enumerate(car_plan):
        d = departures.get(idx, 0)
        path = entry.get("path", [])
        for j, pos in enumerate(path[:-1]):  # exclude depot_in (last element)
            occ[d + j][tuple(pos)] += 1
    return {t: dict(pos_counts) for t, pos_counts in occ.items()}

def _total_road_cost(sim: TrafficSim) -> int:
    return sum(
        sim.calculate_road_cost(pos, cap) for pos, cap in sim.road_capacity.items()
    )

def evaluate(sim: TrafficSim, max_ticks: int = 10_000) -> dict[str, Any]:
    """Replay sim and record outcomes.  For validation / visualisation only.
    """
    if not sim.car_plan or sim.saved_departures is None:
        return {
            "makespan": None,
            "road_cost": _total_road_cost(sim),
            "all_delivered": False,
            "timed_out": False,
            "error": "no car_plan or departures set",
        }

    sim.reset_sim(sim.saved_departures)

    while not sim.is_done():
        if sim.tick_count >= max_ticks:
            return {
                "makespan": sim.tick_count,
                "road_cost": _total_road_cost(sim),
                "all_delivered": sim.all_delivered(),
                "timed_out": True,
            }
        sim.step()

    return {
        "makespan": sim.tick_count,
        "road_cost": _total_road_cost(sim),
        "all_delivered": sim.all_delivered(),
        "timed_out": False,
    }


def run_solver(
    solver_fn: Callable,
    sim: TrafficSim,
    **kwargs,
) -> tuple[dict[str, Any], TrafficSim]:
    """Deep-copy sim, run solver, return results + solved sim state.
    """
    sim_copy = copy.deepcopy(sim)
    t0 = time.perf_counter()
    solver_result = solver_fn(sim_copy, **kwargs)
    solve_time = time.perf_counter() - t0

    # Support both old 2-tuple and new 3-tuple returns
    if len(solver_result) == 3:
        car_plan, departures, meta = solver_result
    else:
        car_plan, departures = solver_result
        meta = {}

    sim_copy.car_plan = car_plan
    sim_copy.saved_departures = departures

    results: dict[str, Any] = dict(meta)
    results["solve_time_s"] = round(solve_time, 3)
    results.setdefault("makespan", None)
    results.setdefault("road_cost", _total_road_cost(sim_copy))
    results.setdefault("all_delivered", bool(car_plan))
    results.setdefault("solve_status", "UNKNOWN")
    results.setdefault("occ", None)
    return results, sim_copy

def validate_plan(
    sim: TrafficSim,
    car_plan: list[dict],
    departures: dict[int, int],
) -> tuple[bool, list[str]]:
    """Check car_plan + departures for consistency with sim.
    """
    errors: list[str] = []

    depots_out = {d.id: d for d in sim.init_depots.values() if d.kind == "out"}
    depots_in = {d.id: d for d in sim.init_depots.values() if d.kind == "in"}
    total_supply = sum(d.amount for d in depots_out.values())

    # Total car count
    if len(car_plan) != total_supply:
        errors.append(f"car_plan length {len(car_plan)} != total supply {total_supply}")

    assigned_out: dict[int, int] = defaultdict(int)
    assigned_in: dict[int, int] = defaultdict(int)

    for idx, entry in enumerate(car_plan):
        d_out_id = entry.get("depot_out_id")
        d_out_pos = entry.get("depot_out_pos")
        d_in_id = entry.get("depot_in_id")
        path = entry.get("path", [])

        # Depot out
        if d_out_id not in depots_out:
            errors.append(f"car[{idx}]: unknown depot_out_id {d_out_id}")
            continue
        d_out = depots_out[d_out_id]
        if tuple(d_out_pos) != d_out.pos:
            errors.append(f"car[{idx}]: depot_out_pos {d_out_pos} != {d_out.pos}")
        assigned_out[d_out_id] += 1

        # Depot in
        if d_in_id not in depots_in:
            errors.append(f"car[{idx}]: unknown depot_in_id {d_in_id}")
            continue
        d_in = depots_in[d_in_id]
        assigned_in[d_in_id] += 1

        # Path checks
        if not path:
            errors.append(f"car[{idx}]: empty path")
            continue
        if tuple(path[-1]) != d_in.pos:
            errors.append(
                f"car[{idx}]: path ends at {path[-1]} not depot_in pos {d_in.pos}"
            )

        # All tiles except last must be ROAD
        for step, pos in enumerate(path[:-1]):
            pos = tuple(pos)
            if not sim._in_bounds(pos):
                errors.append(f"car[{idx}]: path step {step} {pos} out of bounds")
            elif sim.grid[pos[0]][pos[1]] != TileType.ROAD:
                errors.append(f"car[{idx}]: path step {step} {pos} is not a road tile")

        # Departure tick
        if idx not in departures:
            errors.append(f"car[{idx}]: missing departure tick")
        elif departures[idx] < 0:
            errors.append(f"car[{idx}]: negative departure tick {departures[idx]}")

    # Supply per depot_out
    for d_id, d in depots_out.items():
        if assigned_out[d_id] > d.amount:
            errors.append(
                f"depot_out {d_id}: assigned {assigned_out[d_id]} but supply={d.amount}"
            )

    # Demand per depot_in (over-assignment only; under-assignment is allowed but warns)
    for d_id, d in depots_in.items():
        if assigned_in[d_id] > d.amount:
            errors.append(
                f"depot_in {d_id}: assigned {assigned_in[d_id]} but demand={d.amount}"
            )

    # Budget check
    if sim.initial_budget is not None:
        cost = _total_road_cost(sim)
        if cost > sim.initial_budget:
            errors.append(f"road_cost {cost} exceeds budget {sim.initial_budget}")
    return len(errors) == 0, errors
