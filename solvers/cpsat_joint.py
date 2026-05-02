from __future__ import annotations
import sys
import os
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minisim import (
    TrafficSim,
    TERRAIN_ROAD_COST,
    TERRAIN_MAX_CAPACITY,
    DIR4,
    Pos,
)


def _cap_levels(sim: TrafficSim, pos: Pos) -> list[int]:
    """Allowed capacity levels for a buildable tile ([] if not buildable)."""
    tile = sim.grid[pos[0]][pos[1]]
    terrain = sim._terrain_name(tile)
    max_cap = TERRAIN_MAX_CAPACITY.get(terrain, 0)
    if max_cap == 0 or TERRAIN_ROAD_COST.get(terrain, float("inf")) == float("inf"):
        return []
    return list(range(1, max_cap + 1))


def _road_adj(sim: TrafficSim, depot_pos: Pos) -> list[Pos]:
    """Non-depot, buildable tiles orthogonally adjacent to a depot tile."""
    depot_positions = {d.pos for d in sim.init_depots.values()}
    result = []
    x, y = depot_pos
    for dx, dy in DIR4:
        nb = (x + dx, y + dy)
        if not sim._in_bounds(nb):
            continue
        if nb in depot_positions:
            continue
        if _cap_levels(sim, nb):  # [] means building / unbuildable
            result.append(nb)
    return result


# CP-SAT joint solver — time-expanded flow


def solve_cpsat_joint(
    sim: TrafficSim,
    T_max: Optional[int] = None,
    time_limit_s: float = 120.0,
    minimize_cost: bool = True,
) -> tuple[list[dict], dict[int, int], dict]:
    """Time-expanded flow CP-SAT joint solver to find the optimal road network + departure schedule jointly.
    Modifies sim in-place with the chosen roads.
    sim : TrafficSim with terrain + depots, no pre-built roads
    T_max : time horizon (auto-estimated if None)
    time_limit_s : CP-SAT wall-clock budget (split equally across phases)
    minimize_cost : if True (default), run a second solve phase that fixes the
                    optimal makespan and minimises road cost as a secondary goal
    """
    from ortools.sat.python import cp_model

    depots_out = sorted(
        [d for d in sim.init_depots.values() if d.kind == "out"],
        key=lambda d: d.id,
    )
    depots_in = sorted(
        [d for d in sim.init_depots.values() if d.kind == "in"],
        key=lambda d: d.id,
    )
    if not depots_out or not depots_in:
        return [], {}

    N = sum(d.amount for d in depots_out)
    U = N  # max cars on any single tile
    all_dep_pos = {d.pos for d in sim.init_depots.values()}

    # Build sources / demands dicts  (s_id / d_id = TrafficSim depot id)
    sources = {
        d.id: {"adj": _road_adj(sim, d.pos), "stock": d.amount} for d in depots_out
    }
    demands = {
        d.id: {"adj": _road_adj(sim, d.pos), "demand": d.amount} for d in depots_in
    }

    # Warn if any depot has no road-adjacent tiles
    for d in depots_out + depots_in:
        adj = (sources if d.kind == "out" else demands)[d.id]["adj"]
        if not adj:
            print(
                f"[cpsat_joint] WARNING: depot {d.id} ({d.kind}) at {d.pos} "
                "has no buildable adjacent tile - problem may be infeasible",
                file=sys.stderr,
            )

    # Time horizon
    if T_max is None:
        max_manhattan = max(
            abs(do.pos[0] - di.pos[0]) + abs(do.pos[1] - di.pos[1])
            for do in depots_out
            for di in depots_in
        )
        T_max = max_manhattan + N + 10
    T = T_max

    cells = [(x, y) for x in range(sim.w) for y in range(sim.h)]

    # Per-tile allowed capacities and build costs
    allowed_caps: dict[Pos, list[int]] = {
        pos: ([] if pos in all_dep_pos else _cap_levels(sim, pos)) for pos in cells
    }
    build_cost: dict[tuple, int] = {
        (pos, k): sim.calculate_road_cost(pos, k)
        for pos in cells
        for k in allowed_caps[pos]
    }

    # Build model
    model = cp_model.CpModel()

    # --- Road / capacity variables ---
    built: dict[Pos, object] = {}
    road: dict[tuple, object] = {}  # (pos, k) -> BoolVar
    tile_cap: dict[Pos, object] = {}

    for pos in cells:
        caps = allowed_caps[pos]
        built[pos] = model.NewBoolVar(f"b_{pos[0]}_{pos[1]}")

        if caps:
            for k in caps:
                road[(pos, k)] = model.NewBoolVar(f"r_{pos[0]}_{pos[1]}_{k}")
            model.Add(sum(road[(pos, k)] for k in caps) == built[pos])

            tile_cap[pos] = model.NewIntVar(0, max(caps), f"c_{pos[0]}_{pos[1]}")
            model.Add(tile_cap[pos] == sum(k * road[(pos, k)] for k in caps))
        else:
            tile_cap[pos] = model.NewConstant(0)
            model.Add(built[pos] == 0)

    # Road cost expression — reused for budget constraint and phase-2 objective
    road_cost_terms = [
        build_cost[(pos, k)] * road[(pos, k)]
        for pos in cells
        for k in allowed_caps[pos]
    ]
    road_cost_expr = sum(road_cost_terms) if road_cost_terms else 0

    # Hard budget constraint
    if sim.initial_budget is not None and road_cost_terms:
        model.Add(road_cost_expr <= sim.initial_budget)

    # Time-expanded variables
    occ: dict[tuple, object] = {}
    wait: dict[tuple, object] = {}
    flow: dict[tuple, object] = {}
    spawn: dict[tuple, object] = {}
    arrive: dict[tuple, object] = {}

    for t in range(T + 1):
        for pos in cells:
            occ[(pos, t)] = model.NewIntVar(0, U, f"o_{pos[0]}_{pos[1]}_{t}")

    for t in range(T):
        for pos in cells:
            wait[(pos, t)] = model.NewIntVar(0, U, f"w_{pos[0]}_{pos[1]}_{t}")
            x, y = pos
            for dx, dy in DIR4:
                nb = (x + dx, y + dy)
                if sim._in_bounds(nb):
                    flow[(pos, nb, t)] = model.NewIntVar(
                        0, U, f"f_{pos[0]}_{pos[1]}_{nb[0]}_{nb[1]}_{t}"
                    )

    for s_id, sdata in sources.items():
        for u in sdata["adj"]:
            for t in range(T):
                spawn[(s_id, u, t)] = model.NewIntVar(
                    0, U, f"sp_{s_id}_{u[0]}_{u[1]}_{t}"
                )

    for d_id, ddata in demands.items():
        for u in ddata["adj"]:
            for t in range(T):
                arrive[(u, d_id, t)] = model.NewIntVar(
                    0, U, f"ar_{u[0]}_{u[1]}_{d_id}_{t}"
                )

    # Initial condition: grid starts empty
    for pos in cells:
        model.Add(occ[(pos, 0)] == 0)

    # Tile capacity respected at every tick
    for t in range(T + 1):
        for pos in cells:
            model.Add(occ[(pos, t)] <= tile_cap[pos])

    # Movement only on built tiles
    for t in range(T):
        for pos in cells:
            model.Add(wait[(pos, t)] <= U * built[pos])
            x, y = pos
            for dx, dy in DIR4:
                nb = (x + dx, y + dy)
                if sim._in_bounds(nb):
                    model.Add(flow[(pos, nb, t)] <= U * built[pos])
                    model.Add(flow[(pos, nb, t)] <= U * built[nb])

    # Spawn / arrive only on built adjacent tiles
    for s_id, sdata in sources.items():
        for u in sdata["adj"]:
            for t in range(T):
                model.Add(spawn[(s_id, u, t)] <= U * built[u])

    for d_id, ddata in demands.items():
        for u in ddata["adj"]:
            for t in range(T):
                model.Add(arrive[(u, d_id, t)] <= U * built[u])

    # Supply / demand satisfaction
    for s_id, sdata in sources.items():
        model.Add(
            sum(spawn[(s_id, u, t)] for u in sdata["adj"] for t in range(T))
            == sdata["stock"]
        )
    for d_id, ddata in demands.items():
        model.Add(
            sum(arrive[(u, d_id, t)] for u in ddata["adj"] for t in range(T))
            == ddata["demand"]
        )

    # Occupancy decomposition (conservation at each tile, each tick)
    # occ[u,t] = wait[u,t] + Σ_outflows + Σ_arrivals_to_demand
    demand_adj_set: dict[Pos, list[int]] = defaultdict(list)
    for d_id, ddata in demands.items():
        for u in ddata["adj"]:
            demand_adj_set[u].append(d_id)

    source_adj_set: dict[Pos, list[int]] = defaultdict(list)
    for s_id, sdata in sources.items():
        for u in sdata["adj"]:
            source_adj_set[u].append(s_id)

    for t in range(T):
        for pos in cells:
            x, y = pos
            out_flows = [
                flow[(pos, (x + dx, y + dy), t)]
                for dx, dy in DIR4
                if sim._in_bounds((x + dx, y + dy))
            ]
            arr_here = [arrive[(pos, d_id, t)] for d_id in demand_adj_set.get(pos, [])]
            model.Add(wait[(pos, t)] + sum(out_flows) + sum(arr_here) == occ[(pos, t)])

    # State transition
    for t in range(T):
        for pos in cells:
            x, y = pos
            in_flows = [
                flow[((x + dx, y + dy), pos, t)]
                for dx, dy in DIR4
                if sim._in_bounds((x + dx, y + dy))
                and ((x + dx, y + dy), pos, t) in flow
            ]
            spawns_here = [
                spawn[(s_id, pos, t)] for s_id in source_adj_set.get(pos, [])
            ]
            model.Add(
                occ[(pos, t + 1)] == wait[(pos, t)] + sum(in_flows) + sum(spawns_here)
            )

    # End condition: all cars have left the road by tick T
    for pos in cells:
        model.Add(occ[(pos, T)] == 0)

    # Structural cut: isolated road tiles are useless
    # Any non-depot-adjacent built tile must have at least one built neighbour.
    special_adj: set[Pos] = set()
    for sdata in sources.values():
        special_adj.update(sdata["adj"])
    for ddata in demands.values():
        special_adj.update(ddata["adj"])

    for pos in cells:
        if pos not in special_adj and pos not in all_dep_pos:
            x, y = pos
            nbrs = [
                (x + dx, y + dy) for dx, dy in DIR4 if sim._in_bounds((x + dx, y + dy))
            ]
            if nbrs:
                model.Add(sum(built[nb] for nb in nbrs) >= built[pos])

    # --- Makespan objective ---
    makespan = model.NewIntVar(0, T, "makespan")
    for t in range(T):
        has_arr_t = model.NewBoolVar(f"ha_{t}")
        total_arr = sum(
            arrive[(u, d_id, t)]
            for d_id, ddata in demands.items()
            for u in ddata["adj"]
            if (u, d_id, t) in arrive
        )
        model.Add(total_arr >= 1).OnlyEnforceIf(has_arr_t)
        model.Add(total_arr == 0).OnlyEnforceIf(has_arr_t.Not())
        model.Add(makespan >= t).OnlyEnforceIf(has_arr_t)

    model.Minimize(makespan)

    # Phase 1: minimise makespan
    solver = cp_model.CpSolver()
    phase1_limit = time_limit_s / 2 if minimize_cost else time_limit_s
    solver.parameters.max_time_in_seconds = phase1_limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers = 8
    solver.parameters.linearization_level = 2

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(
            f"[cpsat_joint] No solution (status={solver.StatusName(status)}), "
            "falling back to rule_based joint",
            file=sys.stderr,
        )
        from solvers.rule_based import solve_joint as _rb

        return _rb(sim)  # already returns 3-tuple

    opt_makespan = solver.Value(makespan)
    phase1_status = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    print(
        f"[cpsat_joint] Phase 1 ({phase1_status}): makespan={opt_makespan}  "
        f"wall={solver.WallTime():.2f}s",
        file=sys.stderr,
    )

    # Phase 2 (optional): fix makespan, minimise road cost

    if minimize_cost:
        model.Add(makespan <= opt_makespan)
        model.Minimize(road_cost_expr)
        solver.parameters.max_time_in_seconds = time_limit_s / 2
        status2 = solver.Solve(model)
        if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            status = status2
            print(
                f"[cpsat_joint] Phase 2 ({'OPTIMAL' if status2 == cp_model.OPTIMAL else 'FEASIBLE'}): "
                f"makespan={solver.Value(makespan)}  "
                f"road_cost={int(solver.ObjectiveValue())}  "
                f"wall={solver.WallTime():.2f}s",
                file=sys.stderr,
            )
        else:
            print(
                "[cpsat_joint] Phase 2 timed out — keeping Phase 1 solution",
                file=sys.stderr,
            )

    # Apply chosen roads to sim
    for pos in cells:
        for k in allowed_caps.get(pos, []):
            if solver.Value(road.get((pos, k), model.NewConstant(0))):
                ok, msg = sim.add_road(pos, k)
                if not ok:
                    print(
                        f"[cpsat_joint] WARNING: add_road({pos}, {k}) failed: {msg}",
                        file=sys.stderr,
                    )
                break  # at most one k is active per tile

    road_cost = sum(sim.calculate_road_cost(p, c) for p, c in sim.road_capacity.items())
    flow_makespan = solver.Value(
        makespan
    )  # always read from variable, not ObjectiveValue
    solve_status = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    print(
        f"[cpsat_joint] Final: makespan={flow_makespan}  "
        f"roads={len(sim.road_capacity)}  road_cost={road_cost}  "
        f"wall={solver.WallTime():.2f}s",
        file=sys.stderr,
    )

    # Aggregate occ directly from solver variables (flow-model tick numbering)
    occ_result: dict = {}
    for t in range(T + 1):
        tick_occ = {
            pos: solver.Value(occ[(pos, t)])
            for pos in cells
            if solver.Value(occ[(pos, t)]) > 0
        }
        if tick_occ:
            occ_result[t] = tick_occ

    meta = {
        # final solution need to +1: arrive[(u,d,t)] means car exits at tick t; depot entry is t+1.
        "makespan": flow_makespan + 1,
        "road_cost": road_cost,
        "occ": occ_result,
        "solve_status": solve_status,
        "all_delivered": True,
    }
    return [], {}, meta


# Alias matching the joint-solver interface
solve_joint = solve_cpsat_joint
